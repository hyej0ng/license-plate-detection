"""
실행방법:
python 02_quarter/scripts/02_training/train_quarter.py \
  --device 2 \
  --epochs 40 \
  --batch 128 \
  --patience 0 \
  --seed 42

resume
python 02_quarter/scripts/02_training/train_quarter.py \
  --resume /home/hyejong/landing_pjt/02_quarter/runs/quarter_yolo26n_20260903-152325/weights/last.pt \
  --epochs 50 \
  --device 2
"""

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


# 1. 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUARTER_ROOT = PROJECT_ROOT / "02_quarter"
DATA_YAML = QUARTER_ROOT / "configs" / "quarter_data.yaml"
PRIMARY_MODEL_PATH = PROJECT_ROOT / "common" / "weights" / "pretrained" / "yolo26n.pt"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"
RUNS_DIR = QUARTER_ROOT / "runs"
LOGS_DIR = QUARTER_ROOT / "logs"


# 2. 학습 하이퍼파라미터: 기본값을 한곳에서 관리
EPOCHS = 50
PATIENCE = 0
IMAGE_SIZE = 640
BATCH_SIZE = 128
WORKERS = 8
DEVICE = "0" # gpu 번호

OPTIMIZER = "SGD"
INITIAL_LR = 0.001
FINAL_LR_RATIO = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 1.0
WARMUP_BIAS_LR = 0.01 #기본 설정값이 0.1이었음 너무 큼

MOSAIC = 0.20
MIXUP = 0.0
SCALE = 0.20
TRANSLATE = 0.05
HORIZONTAL_FLIP = 0.50
VERTICAL_FLIP = 0.0
HSV_H = 0.015
HSV_S = 0.50
HSV_V = 0.30

SEED = 42
VAL_CONFIDENCE = 0.001
VAL_NMS_IOU = 0.70
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def default_model_path() -> Path:
    return PRIMARY_MODEL_PATH if PRIMARY_MODEL_PATH.is_file() else FALLBACK_MODEL_PATH


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv26n quarter 학습 및 validation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--weights", type=Path, help="이 가중치에서 optimizer를 새로 시작")
    mode.add_argument("--resume", type=Path, help="last.pt의 optimizer/epoch까지 정확히 복구")
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="데이터/가중치/설정만 확인하고 실제 학습은 시작하지 않음",
    )
    return parser.parse_args()


def make_logger(log_path: Path):
    logger = logging.getLogger(f"quarter_training_{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    yolo_handler = logging.FileHandler(log_path, encoding="utf-8")
    yolo_handler.setFormatter(logging.Formatter("[YOLO] %(message)s"))
    ULTRALYTICS_LOGGER.addHandler(yolo_handler)
    return logger, yolo_handler


def count_files(folder: Path, extensions: set[str]) -> int:
    return sum(1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def check_dataset(data_yaml: Path, model_path: Path) -> tuple[Path, dict]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"데이터 YAML이 없습니다: {data_yaml}")
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 체크포인트가 없습니다: {model_path}")
    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_root = Path(config["path"])
    if not data_root.is_absolute():
        data_root = (data_yaml.parent / data_root).resolve()
    counts = {}
    for split in ("train", "val"):
        image_dir = data_root / config[split]
        label_dir = data_root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"{split} images/labels 폴더가 없습니다: {image_dir}, {label_dir}")
        image_count = count_files(image_dir, IMAGE_EXTENSIONS)
        label_count = count_files(label_dir, {".txt"})
        if image_count == 0 or image_count != label_count:
            raise ValueError(f"{split} 이미지/라벨 개수 오류: images={image_count}, labels={label_count}")
        counts[split] = (image_count, label_count)
    return data_root, counts


def check_resume_checkpoint(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", -1)
    if epoch < 0 or checkpoint.get("optimizer") is None:
        raise ValueError("optimizer가 제거된 체크포인트입니다. --resume 대신 --weights를 사용하세요.")
    return epoch + 1


def number(row: dict, key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else math.nan


def total_loss(row: dict, prefix: str) -> float:
    values = [number(row, f"{prefix}/{name}") for name in ("box_loss", "cls_loss", "dfl_loss")]
    return math.nan if any(math.isnan(value) for value in values) else sum(values)


def read_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_loss_graph(csv_path: Path, output_path: Path) -> None:
    rows = read_results(csv_path)
    epochs = [int(float(row["epoch"])) for row in rows]
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(epochs, [total_loss(row, "train") for row in rows], label="train total loss")
    axis.plot(epochs, [total_loss(row, "val") for row in rows], label="validation total loss")
    axis.set(title="YOLOv26n Quarter Loss", xlabel="epoch", ylabel="box + class + DFL loss")
    axis.grid(alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


class EpochRecorder:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_epoch = -1

    def on_fit_epoch_end(self, trainer) -> None:
        csv_path = Path(trainer.csv)
        rows = read_results(csv_path) if csv_path.is_file() else []
        if not rows:
            return
        row = rows[-1]
        epoch = int(float(row["epoch"]))
        if epoch == self.last_epoch:
            return
        self.last_epoch = epoch
        precision = number(row, "metrics/precision(B)")
        recall = number(row, "metrics/recall(B)")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        self.logger.info(
            f"[LOG] epoch {epoch:03d}/{trainer.epochs:03d} | train_loss {total_loss(row, 'train'):.4f} "
            f"| val_loss {total_loss(row, 'val'):.4f} | P {precision:.4f} | R {recall:.4f} "
            f"| F1 {f1:.4f} | mAP50 {number(row, 'metrics/mAP50(B)'):.4f} "
            f"| mAP50-95 {number(row, 'metrics/mAP50-95(B)'):.4f}"
        )
        save_loss_graph(csv_path, Path(trainer.save_dir) / "loss_curve.png")


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs <= 0 or arguments.image_size <= 0 or arguments.workers < 0:
        raise ValueError("epochs/image-size는 1 이상, workers는 0 이상이어야 합니다.")
    if arguments.batch == 0 or arguments.batch < -1 or arguments.patience < 0:
        raise ValueError("batch는 -1 또는 1 이상, patience는 0 이상이어야 합니다.")
    data_yaml = arguments.data.resolve()
    model_path = (arguments.weights or arguments.resume or default_model_path()).resolve()
    training_mode = "exact_resume" if arguments.resume else ("new_from_weights" if arguments.weights else "new")
    resume_epoch = check_resume_checkpoint(model_path) if arguments.resume else None

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if arguments.resume:
        run_dir = model_path.parent.parent
        run_name = run_dir.name
        log_path = LOGS_DIR / f"resume_{timestamp}.log"
    else:
        run_name = f"quarter_yolo26n_{timestamp}"
        run_dir = RUNS_DIR / run_name
        log_path = LOGS_DIR / f"train_{timestamp}.log"
    logger, yolo_handler = make_logger(log_path)

    try:
        data_root, counts = check_dataset(data_yaml, model_path)
        logger.info(f"[INFO] time_local: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        logger.info(f"[INFO] time_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
        logger.info(f"[INFO] Python={platform.python_version()}, PyTorch={torch.__version__}, Ultralytics={ultralytics.__version__}")
        logger.info(f"[INFO] mode={training_mode}, model={model_path}, data={data_yaml}, data_root={data_root}")
        logger.info(f"[INFO] train={counts['train']}, val={counts['val']}, run={run_dir}")
        if resume_epoch is not None:
            logger.info(f"[INFO] resume_from_completed_epoch={resume_epoch}")
        logger.info(
            f"[INFO] epochs={arguments.epochs}, patience={arguments.patience}, imgsz={arguments.image_size}, "
            f"batch={arguments.batch}, workers={arguments.workers}, device={arguments.device}, seed={arguments.seed}"
        )
        logger.info(
            f"[INFO] optimizer={OPTIMIZER}, lr0={INITIAL_LR}, lrf={FINAL_LR_RATIO}, momentum={MOMENTUM}, "
            f"weight_decay={WEIGHT_DECAY}, warmup_epochs={WARMUP_EPOCHS}, "
            f"warmup_bias_lr={WARMUP_BIAS_LR}, mosaic={MOSAIC}, scale={SCALE}, "
            f"conf={VAL_CONFIDENCE}, iou={VAL_NMS_IOU}"
        )
        if arguments.check_only:
            logger.info("[RESULT] check_only passed: dataset, labels, model checkpoint, and arguments are valid")
            return
        if arguments.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 찾지 못했습니다. CPU 학습은 --device cpu를 사용하세요.")

        model = YOLO(str(model_path))
        recorder = EpochRecorder(logger)
        model.add_callback("on_fit_epoch_end", recorder.on_fit_epoch_end)
        if arguments.resume:
            model.train(
                resume=True,
                epochs=arguments.epochs,
                patience=arguments.patience,
                device=arguments.device,
                workers=arguments.workers,
                seed=arguments.seed,
                val=True,
                plots=True,
            )
        else:
            model.train(
                data=str(data_yaml), epochs=arguments.epochs, patience=arguments.patience,
                imgsz=arguments.image_size, batch=arguments.batch, workers=arguments.workers,
                device=arguments.device, optimizer=OPTIMIZER, lr0=INITIAL_LR,
                lrf=FINAL_LR_RATIO, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                warmup_epochs=WARMUP_EPOCHS, warmup_bias_lr=WARMUP_BIAS_LR,
                cos_lr=True, mosaic=MOSAIC, mixup=MIXUP,
                scale=SCALE, translate=TRANSLATE, fliplr=HORIZONTAL_FLIP,
                flipud=VERTICAL_FLIP, hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
                close_mosaic=10, amp=True, cache=False, val=True, conf=VAL_CONFIDENCE,
                iou=VAL_NMS_IOU, plots=True, save=True, seed=arguments.seed,
                deterministic=True, project=str(RUNS_DIR), name=run_name,
                exist_ok=False, verbose=True,
            )
        logger.info(f"[RESULT] best_model: {run_dir / 'weights' / 'best.pt'}")
        logger.info(f"[RESULT] last_model: {run_dir / 'weights' / 'last.pt'}")
        logger.info(f"[RESULT] validation_metrics: {run_dir / 'results.csv'}")
    except Exception:
        logger.exception("[ERROR] Quarter training failed")
        raise
    finally:
        ULTRALYTICS_LOGGER.removeHandler(yolo_handler)
        yolo_handler.close()


if __name__ == "__main__":
    main()
