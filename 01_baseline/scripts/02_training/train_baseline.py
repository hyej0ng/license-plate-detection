"""
실행 방법: 
cd /path/to/landing_pjt
conda activate yolo
python 01_baseline/scripts/02_training/train_baseline.py
"""

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


# 화면이 없는 서버에서도 그래프를 파일로 저장할 수 있게 한다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. 경로 설정
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_YAML = PROJECT_ROOT / "01_baseline" / "configs" / "license_plate.yaml"
MODEL_PATH = PROJECT_ROOT / "common" / "weights" / "pretrained" / "yolo26n.pt"
RUNS_DIR = PROJECT_ROOT / "01_baseline" / "runs"
LOGS_DIR = PROJECT_ROOT / "01_baseline" / "logs"


# =============================================================================
# 2. 학습 설정 - 하이퍼파라미터는 여기에서 변경한다.
# =============================================================================

EPOCHS = 50
PATIENCE = 0  # 0: early stopping을 사용하지 않고 EPOCHS까지 학습
IMAGE_SIZE = 640
# 864로했더니 미탐이 가장 작았다
BATCH_SIZE = 32  # -1: GPU 메모리에 맞게 Ultralytics가 자동 결정
# 32
WORKERS = 8
DEVICE = 4  # n번째 GPU를 사용하겠단 뜻. CPU를 쓸 때는 "cpu"로 변경

OPTIMIZER = "SGD"
INITIAL_LR = 0.002
# 다들 0.05~0.02
FINAL_LR_RATIO = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 5.0

# 번호판이 작으므로 객체를 더 작게 만드는 강한 증강은 줄였다.
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


def make_logger(log_path):
    """터미널과 파일에 같은 사용자 로그를 남긴다."""
    logger = logging.getLogger(f"baseline_training_{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Ultralytics가 출력하는 모델 구조와 학습 메시지도 같은 파일에 저장한다.
    yolo_file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    yolo_file_handler.setFormatter(logging.Formatter("[YOLO] %(message)s"))
    ULTRALYTICS_LOGGER.addHandler(yolo_file_handler)

    return logger, yolo_file_handler


def count_files(folder, extensions):
    """폴더 바로 아래에서 지정한 확장자의 파일 수를 센다."""
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def check_dataset_and_count(model_path):
    """학습 전에 YAML과 train/val/test 이미지·라벨 경로를 확인한다."""
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"데이터 YAML이 없습니다: {DATA_YAML}")
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 체크포인트가 없습니다: {model_path}")

    with DATA_YAML.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    data_root = Path(data_config["path"])
    if not data_root.is_absolute():
        data_root = (DATA_YAML.parent / data_root).resolve()

    split_counts = {}

    for split in ("train", "val", "test"):
        image_dir = data_root / data_config[split]
        label_dir = data_root / "labels" / split

        if not image_dir.is_dir():
            raise FileNotFoundError(f"{split} 이미지 폴더가 없습니다: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"{split} 라벨 폴더가 없습니다: {label_dir}")

        image_count = count_files(image_dir, IMAGE_EXTENSIONS)
        label_count = count_files(label_dir, {".txt"})

        if image_count == 0:
            raise ValueError(f"{split} 이미지가 없습니다: {image_dir}")
        if image_count != label_count:
            raise ValueError(
                f"{split} 이미지/라벨 수가 다릅니다: "
                f"images={image_count:,}, labels={label_count:,}"
            )

        split_counts[split] = (image_count, label_count)

    return data_root, split_counts


def parse_arguments():
    """새 학습, 가중치 이어 학습, 정확한 resume 중 하나를 선택한다."""
    parser = argparse.ArgumentParser(description="YOLOv26n baseline 학습")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--weights",
        type=Path,
        help="지정한 .pt 가중치에서 optimizer를 새로 만들어 학습",
    )
    mode.add_argument(
        "--resume",
        type=Path,
        help="중간에 중단된 last.pt에서 optimizer와 epoch까지 그대로 복구",
    )
    return parser.parse_args()


def check_resume_checkpoint(checkpoint_path):
    """정확한 resume에 필요한 epoch와 optimizer가 남아 있는지 확인한다."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_epoch = checkpoint.get("epoch", -1)
    optimizer = checkpoint.get("optimizer")

    if saved_epoch < 0 or optimizer is None:
        raise ValueError(
            "이 체크포인트는 정상 종료 후 optimizer가 제거되어 정확한 resume가 불가능합니다. "
            "--resume 대신 --weights를 사용해 새 학습으로 이어가세요."
        )

    return saved_epoch + 1


def number(row, key):
    """CSV 값을 float로 바꾼다. 값이 없으면 NaN을 반환한다."""
    value = row.get(key)
    if value is None or value == "":
        return math.nan
    return float(value)


def total_loss(row, prefix):
    """YOLO의 box, class, DFL loss를 더해 보기 쉬운 총 loss를 만든다."""
    keys = [
        f"{prefix}/box_loss",
        f"{prefix}/cls_loss",
        f"{prefix}/dfl_loss",
    ]
    values = [number(row, key) for key in keys]

    if any(math.isnan(value) for value in values):
        return math.nan
    return sum(values)


def read_results(csv_path):
    """Ultralytics가 epoch마다 저장한 results.csv를 읽는다."""
    with csv_path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_loss_graph(csv_path, graph_path):
    """현재 epoch까지 누적된 train/validation 총 loss 그래프를 저장한다."""
    rows = read_results(csv_path)
    epochs = [int(float(row["epoch"])) for row in rows]
    train_losses = [total_loss(row, "train") for row in rows]
    val_losses = [total_loss(row, "val") for row in rows]

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(epochs, train_losses, marker="o", markersize=4, label="train total loss")
    axis.plot(epochs, val_losses, marker="o", markersize=4, label="validation total loss")
    axis.set_title("YOLOv26n loss by epoch")
    axis.set_xlabel("epoch")
    axis.set_ylabel("box loss + class loss + DFL loss")
    axis.grid(alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(graph_path, dpi=150)
    plt.close(figure)


class EpochRecorder:
    """매 epoch 종료 시 한 줄 로그와 누적 loss 그래프를 만든다."""

    def __init__(self, logger):
        self.logger = logger
        self.last_epoch = 0

    def on_fit_epoch_end(self, trainer):
        csv_path = Path(trainer.csv)
        if not csv_path.is_file():
            return

        rows = read_results(csv_path)
        if not rows:
            return

        row = rows[-1]
        epoch = int(float(row["epoch"]))

        # 학습 마지막의 추가 validation callback에서는 같은 epoch를 중복 기록하지 않는다.
        if epoch == self.last_epoch:
            return
        self.last_epoch = epoch

        train_loss = total_loss(row, "train")
        val_loss = total_loss(row, "val")
        precision = number(row, "metrics/precision(B)")
        recall = number(row, "metrics/recall(B)")
        map50 = number(row, "metrics/mAP50(B)")
        map50_95 = number(row, "metrics/mAP50-95(B)")
        learning_rate = number(row, "lr/pg0")
        progress = epoch / trainer.epochs * 100
        f1_score = 0.0
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)

        self.logger.info(
            f"[LOG] {progress:5.1f}% | epoch {epoch:03d}/{trainer.epochs:03d} "
            f"| train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
            f"| P {precision:.4f} | R {recall:.4f} | F1 {f1_score:.4f} "
            f"| mAP50 {map50:.4f} | mAP50-95 {map50_95:.4f} "
            f"| lr {learning_rate:.3e}"
        )

        graph_path = Path(trainer.save_dir) / "loss_curve.png"
        save_loss_graph(csv_path, graph_path)


def log_start_information(
    logger,
    log_path,
    run_dir,
    data_root,
    split_counts,
    training_mode,
    model_path,
    resume_epoch,
):
    """학습 재현에 필요한 환경과 설정을 로그 맨 앞에 기록한다."""
    local_time = datetime.now().astimezone()
    utc_time = datetime.now(timezone.utc)

    logger.info(f"[INFO] Logging to: {log_path}")
    logger.info(f"[INFO] time_local: {local_time.isoformat(timespec='seconds')}")
    logger.info(f"[INFO] time_utc:   {utc_time.isoformat(timespec='seconds')}")
    logger.info(f"[INFO] Python: {platform.python_version()}")
    logger.info(f"[INFO] PyTorch: {torch.__version__}")
    logger.info(f"[INFO] Ultralytics: {ultralytics.__version__}")
    logger.info(f"[INFO] data_yaml: {DATA_YAML}")
    logger.info(f"[INFO] data_root: {data_root}")
    logger.info(f"[INFO] training_mode: {training_mode}")
    logger.info(f"[INFO] model_checkpoint: {model_path}")
    if resume_epoch is not None:
        logger.info(f"[INFO] resume_from_completed_epoch: {resume_epoch}")
    logger.info(f"[INFO] run_directory: {run_dir}")

    for split, counts in split_counts.items():
        logger.info(
            f"[INFO] {split}: images={counts[0]:,}, labels={counts[1]:,}"
        )

    if torch.cuda.is_available():
        gpu_index = DEVICE if isinstance(DEVICE, int) else 0
        gpu_name = torch.cuda.get_device_name(gpu_index)
        gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)
        logger.info(f"[INFO] device: cuda:{gpu_index} ({gpu_name}, {gpu_memory:.1f} GB)")
    else:
        logger.info("[INFO] device: CUDA is not available")

    if training_mode == "exact_resume":
        logger.info("[INFO] hyperparameters: restored from the resume checkpoint")
        logger.info(
            f"[INFO] resume overrides: patience={PATIENCE}, device={DEVICE}, "
            f"workers={WORKERS}, val=True, plots=True"
        )
    else:
        logger.info(
            f"[INFO] epochs={EPOCHS}, patience={PATIENCE}, imgsz={IMAGE_SIZE}, "
            f"batch={BATCH_SIZE}, workers={WORKERS}"
        )
        logger.info(
            f"[INFO] optimizer={OPTIMIZER}, lr0={INITIAL_LR}, lrf={FINAL_LR_RATIO}, "
            f"momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}, warmup_epochs={WARMUP_EPOCHS}"
        )
        logger.info(
            f"[INFO] mosaic={MOSAIC}, mixup={MIXUP}, scale={SCALE}, "
            f"translate={TRANSLATE}, fliplr={HORIZONTAL_FLIP}, flipud={VERTICAL_FLIP}"
        )
    logger.info(f"[INFO] early_stopping: disabled (patience={PATIENCE})")
    logger.info(
        f"[INFO] validation thresholds: confidence={VAL_CONFIDENCE}, "
        f"NMS IoU={VAL_NMS_IOU}"
    )
    logger.info("[INFO] Starting training")


def main():
    arguments = parse_arguments()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now().astimezone()
    timestamp = start_time.strftime("%Y%m%d-%H%M%S")

    training_mode = "new"
    model_path = MODEL_PATH
    resume_epoch = None

    if arguments.weights is not None:
        training_mode = "new_from_weights"
        model_path = arguments.weights.resolve()
    elif arguments.resume is not None:
        training_mode = "exact_resume"
        model_path = arguments.resume.resolve()

    if training_mode == "exact_resume":
        run_dir = model_path.parent.parent
        run_name = run_dir.name
        log_path = LOGS_DIR / f"resume_{timestamp}.log"
    else:
        run_name = f"baseline_yolo26n_{timestamp}"
        run_dir = RUNS_DIR / run_name
        log_path = LOGS_DIR / f"train_{timestamp}.log"

    logger, yolo_file_handler = make_logger(log_path)

    try:
        data_root, split_counts = check_dataset_and_count(model_path)

        if training_mode == "exact_resume":
            resume_epoch = check_resume_checkpoint(model_path)

        log_start_information(
            logger,
            log_path,
            run_dir,
            data_root,
            split_counts,
            training_mode,
            model_path,
            resume_epoch,
        )

        if not torch.cuda.is_available() and DEVICE != "cpu":
            raise RuntimeError(
                "CUDA GPU를 찾지 못했습니다. nvidia-smi와 PyTorch CUDA 설치를 확인하세요. "
                "CPU 학습이 필요하면 DEVICE를 'cpu'로 변경하세요."
            )

        model = YOLO(str(model_path))
        recorder = EpochRecorder(logger)
        model.add_callback("on_fit_epoch_end", recorder.on_fit_epoch_end)

        if training_mode == "exact_resume":
            # 원래 실행의 optimizer, 학습률 scheduler, epoch, 결과 폴더를 복구한다.
            model.train(
                resume=True,
                patience=PATIENCE,
                device=DEVICE,
                workers=WORKERS,
                val=True,
                plots=True,
            )
        else:
            model.train(
                data=str(DATA_YAML),
                epochs=EPOCHS,
                patience=PATIENCE,
                imgsz=IMAGE_SIZE,
                batch=BATCH_SIZE,
                workers=WORKERS,
                device=DEVICE,
                optimizer=OPTIMIZER,
                lr0=INITIAL_LR,
                lrf=FINAL_LR_RATIO,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                warmup_epochs=WARMUP_EPOCHS,
                cos_lr=True,
                mosaic=MOSAIC,
                mixup=MIXUP,
                scale=SCALE,
                translate=TRANSLATE,
                fliplr=HORIZONTAL_FLIP,
                flipud=VERTICAL_FLIP,
                hsv_h=HSV_H,
                hsv_s=HSV_S,
                hsv_v=HSV_V,
                close_mosaic=10,
                amp=True,
                cache=False,
                val=True,
                conf=VAL_CONFIDENCE,
                iou=VAL_NMS_IOU,
                plots=True,
                save=True,
                seed=SEED,
                deterministic=True,
                project=str(RUNS_DIR),
                name=run_name,
                exist_ok=False,
                verbose=True,
            )

        elapsed = datetime.now().astimezone() - start_time
        logger.info(f"[INFO] Training finished. elapsed_time: {elapsed}")
        logger.info(f"[INFO] Best model: {run_dir / 'weights' / 'best.pt'}")
        logger.info(f"[INFO] Validation metrics: {run_dir / 'results.csv'}")
        logger.info(f"[INFO] Loss graph: {run_dir / 'loss_curve.png'}")

    except Exception:
        logger.exception("[ERROR] Training failed")
        raise
    finally:
        ULTRALYTICS_LOGGER.removeHandler(yolo_file_handler)
        yolo_file_handler.close()


if __name__ == "__main__":
    main()
