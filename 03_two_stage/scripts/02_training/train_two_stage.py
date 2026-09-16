"""전처리된 vehicle/license_plate 데이터로 YOLOv26n을 학습하고 매 epoch 검증한다."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "03_two_stage"
DEFAULT_DATA = PROJECT_DIR / "configs" / "license_plate_vehicle.yaml"
PRIMARY_MODEL = REPO_ROOT / "common" / "weights" / "pretrained" / "yolo26n.pt"
FALLBACK_MODEL = REPO_ROOT / "yolo26n.pt"
RUNS_DIR = PROJECT_DIR / "runs"
LOGS_DIR = PROJECT_DIR / "logs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv26n vehicle/license plate 학습+validation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--weights", type=Path, help="지정 가중치로 optimizer를 새로 시작")
    mode.add_argument("--resume", type=Path, help="last.pt에서 optimizer/epoch까지 복구")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--run-name", help="고정할 Ultralytics run 이름")
    parser.add_argument("--check-only", action="store_true", help="데이터와 설정만 검사")
    return parser.parse_args()


def default_model() -> Path:
    return PRIMARY_MODEL if PRIMARY_MODEL.is_file() else FALLBACK_MODEL


def make_logger(path: Path):
    logger = logging.getLogger(f"two_stage_train_{path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    yolo_handler = logging.FileHandler(path, encoding="utf-8")
    yolo_handler.setFormatter(logging.Formatter("[YOLO] %(message)s"))
    ULTRALYTICS_LOGGER.addHandler(yolo_handler)
    return logger, yolo_handler


def count_files(path: Path, extensions: set[str]) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in extensions)


def validate_dataset(data_yaml: Path, model_path: Path) -> tuple[Path, dict]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"데이터 YAML이 없습니다: {data_yaml}")
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 가중치가 없습니다: {model_path}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config.get("names")
    expected_names = {0: "plate", 1: "car"}
    normalized_names = {int(key): value for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(names or []))
    if normalized_names != expected_names:
        raise ValueError(f"class ID는 {expected_names}이어야 합니다: {normalized_names}")
    data_root = Path(config["path"])
    if not data_root.is_absolute():
        data_root = (data_yaml.parent / data_root).resolve()
    counts = {}
    for split in ("train", "val", "test"):
        image_dir = data_root / config[split]
        label_dir = data_root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"{split} images/labels 폴더가 없습니다: {image_dir}, {label_dir}")
        images = count_files(image_dir, IMAGE_EXTENSIONS)
        labels = count_files(label_dir, {".txt"})
        if images == 0 or images != labels:
            raise ValueError(f"{split} 이미지/라벨 개수 오류: images={images}, labels={labels}")
        image_stems = {path.stem for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
        label_stems = {path.stem for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"}
        if image_stems != label_stems:
            raise ValueError(
                f"{split} 이미지/라벨 stem 불일치: "
                f"missing_labels={sorted(image_stems - label_stems)[:3]}, "
                f"missing_images={sorted(label_stems - image_stems)[:3]}"
            )
        for label_path in label_dir.glob("*.txt"):
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5 or int(parts[0]) not in expected_names:
                    raise ValueError(f"YOLO 라벨 오류: {label_path}:{line_number}: {line}")
                values = tuple(map(float, parts[1:]))
                if not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
                    raise ValueError(f"bbox 범위 오류: {label_path}:{line_number}: {values}")
        counts[split] = {"images": images, "labels": labels}
    return data_root, counts


def check_resume(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", -1)
    if epoch < 0 or checkpoint.get("optimizer") is None:
        raise ValueError("optimizer가 없는 체크포인트입니다. --resume 대신 --weights를 사용하세요.")
    return epoch + 1


def metric_number(row: dict, key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else math.nan


def total_loss(row: dict, prefix: str) -> float:
    values = [metric_number(row, f"{prefix}/{name}") for name in ("box_loss", "cls_loss", "dfl_loss")]
    return math.nan if any(math.isnan(value) for value in values) else sum(values)


def read_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_loss_curve(results_csv: Path, output_path: Path) -> None:
    """epoch별 train/validation 총 loss와 구성 loss를 저장한다."""
    rows = read_results(results_csv)
    if not rows:
        return
    epochs = [int(float(row["epoch"])) for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(epochs, [total_loss(row, "train") for row in rows], marker="o", label="train total loss")
    axes[0].plot(epochs, [total_loss(row, "val") for row in rows], marker="o", label="validation total loss")
    axes[0].set(title="YOLOv26n Train vs Validation Loss", ylabel="box + class + DFL loss")
    axes[0].grid(alpha=0.35)
    axes[0].legend()

    colors = {"box_loss": "#1f77b4", "cls_loss": "#ff7f0e", "dfl_loss": "#2ca02c"}
    for loss_name, color in colors.items():
        axes[1].plot(epochs, [metric_number(row, f"train/{loss_name}") for row in rows],
                     color=color, linestyle="-", label=f"train {loss_name}")
        axes[1].plot(epochs, [metric_number(row, f"val/{loss_name}") for row in rows],
                     color=color, linestyle="--", label=f"val {loss_name}")
    axes[1].set(xlabel="epoch", ylabel="loss", title="Loss Components")
    axes[1].grid(alpha=0.35)
    axes[1].legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


class EpochRecorder:
    """매 epoch 결과를 로그에 기록하고 loss curve를 갱신한다."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_epoch = -1

    def on_fit_epoch_end(self, trainer) -> None:
        results_csv = Path(trainer.csv)
        rows = read_results(results_csv) if results_csv.is_file() else []
        if not rows:
            return
        row = rows[-1]
        epoch = int(float(row["epoch"]))
        if epoch == self.last_epoch:
            return
        self.last_epoch = epoch
        precision = metric_number(row, "metrics/precision(B)")
        recall = metric_number(row, "metrics/recall(B)")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        self.logger.info(
            f"[EPOCH] {epoch:03d}/{trainer.epochs:03d} "
            f"train_loss={total_loss(row, 'train'):.5f} val_loss={total_loss(row, 'val'):.5f} "
            f"P={precision:.5f} R={recall:.5f} F1={f1:.5f} "
            f"mAP50={metric_number(row, 'metrics/mAP50(B)'):.5f} "
            f"mAP50-95={metric_number(row, 'metrics/mAP50-95(B)'):.5f}"
        )
        save_loss_curve(results_csv, Path(trainer.save_dir) / "loss_curve.png")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{'check' if args.check_only else 'train'}_{timestamp}.log"
    logger, yolo_handler = make_logger(log_path)
    try:
        if args.epochs <= 0 or args.image_size <= 0 or args.workers < 0:
            raise ValueError("epochs/image-size는 양수, workers는 0 이상이어야 합니다.")
        if args.batch == 0 or args.batch < -1 or args.patience < 0:
            raise ValueError("batch는 -1 또는 양수, patience는 0 이상이어야 합니다.")
        data_yaml = args.data.resolve()
        model_path = (args.weights or args.resume or default_model()).resolve()
        data_root, counts = validate_dataset(data_yaml, model_path)
        resume_epoch = check_resume(model_path) if args.resume else None
        run_name = args.run_name or f"two_stage_yolo26n_{timestamp}"
        if args.resume:
            run_dir = model_path.parent.parent
            run_name = run_dir.name
        else:
            run_dir = RUNS_DIR / run_name
            if run_dir.exists() and not args.check_only:
                raise FileExistsError(f"run 폴더가 이미 있습니다: {run_dir}")

        logger.info(f"[INFO] time_local={datetime.now().astimezone().isoformat(timespec='seconds')}")
        logger.info(f"[INFO] time_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}")
        logger.info(f"[INFO] Python={platform.python_version()}, PyTorch={torch.__version__}, Ultralytics={ultralytics.__version__}")
        logger.info(f"[INFO] data={data_yaml}, data_root={data_root}, counts={counts}")
        logger.info(f"[INFO] model={model_path}, run={run_dir}, device={args.device}")
        logger.info(
            f"[INFO] epochs={args.epochs}, imgsz={args.image_size}, batch={args.batch}, "
            f"workers={args.workers}, patience={args.patience}, seed={args.seed}"
        )
        logger.info("[INFO] optimizer=SGD, lr0=0.002, lrf=0.01, val_conf=0.001, val_iou=0.70")
        if resume_epoch is not None:
            logger.info(f"[INFO] resume_from_completed_epoch={resume_epoch}")
        if args.check_only:
            logger.info("[RESULT] check-only passed")
            return
        if args.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 찾지 못했습니다. CPU는 --device cpu를 사용하세요.")

        model = YOLO(str(model_path))
        recorder = EpochRecorder(logger)
        model.add_callback("on_fit_epoch_end", recorder.on_fit_epoch_end)
        if args.resume:
            model.train(resume=True, epochs=args.epochs, patience=args.patience, device=args.device,
                        workers=args.workers, seed=args.seed, val=True, plots=True)
        else:
            model.train(
                data=str(data_yaml), epochs=args.epochs, patience=args.patience,
                imgsz=args.image_size, batch=args.batch, workers=args.workers, device=args.device,
                optimizer="SGD", lr0=0.002, lrf=0.01, momentum=0.937,
                weight_decay=0.0005, warmup_epochs=5.0, cos_lr=True,
                mosaic=0.20, mixup=0.0, scale=0.20, translate=0.05,
                fliplr=0.50, flipud=0.0, hsv_h=0.015, hsv_s=0.50, hsv_v=0.30,
                close_mosaic=10, amp=True, cache=False, val=True, conf=0.001, iou=0.70,
                plots=True, save=True, seed=args.seed, deterministic=True,
                project=str(RUNS_DIR), name=run_name, exist_ok=False, verbose=True,
            )
        results_csv = run_dir / "results.csv"
        loss_curve = run_dir / "loss_curve.png"
        if results_csv.is_file():
            save_loss_curve(results_csv, loss_curve)
        logger.info(f"[RESULT] best_model={run_dir / 'weights' / 'best.pt'}")
        logger.info(f"[RESULT] validation_metrics={results_csv}")
        logger.info(f"[RESULT] loss_curve={loss_curve}")
    except Exception:
        logger.exception("[ERROR] training failed")
        raise
    finally:
        ULTRALYTICS_LOGGER.removeHandler(yolo_handler)
        yolo_handler.close()


if __name__ == "__main__":
    main()
