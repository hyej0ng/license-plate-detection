"""
실행 방법: 
cd /path/to/landing_pjt
conda activate yolo
python 01_baseline/scripts/03_inference/inference_baseline.py \
  --model 01_baseline/runs/baseline_yolo26n_20260820-142748_30epoch/weights/best.pt \
  --conf 0.001
  
"""

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO


# 1. 기본 경로와 inference 설정

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "01_baseline"
    / "runs"
    / "baseline_yolo26n_20260811-100138"
    / "weights"
    / "best.pt"
)
DEFAULT_SOURCE = PROJECT_ROOT / "01_baseline" / "data" / "preprocessed" / "images" / "test"
OUTPUT_ROOT = PROJECT_ROOT / "01_baseline" / "results" / "predictions"

IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.70
BATCH_SIZE = 16
DEVICE = 0  # 첫 번째 GPU. CPU를 사용하려면 명령에서 --device cpu 입력

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_arguments():
    """모델, 입력, threshold와 장치를 명령행에서 선택한다."""
    parser = argparse.ArgumentParser(description="YOLOv26n baseline 이미지 inference")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="사용할 .pt 모델 경로 (기본값: 완료된 baseline run의 best.pt)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="이미지 한 장 또는 이미지 폴더 (기본값: 전처리된 test 이미지 폴더)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="최소 confidence threshold (기본값: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=NMS_IOU_THRESHOLD,
        help="후처리 IoU 인자 (기본값: 0.70, YOLOv26 end-to-end 모드에서는 NMS에 사용되지 않음)",
    )
    parser.add_argument(
        "--device",
        default=str(DEVICE),
        help="추론 장치: 0, 1 또는 cpu (기본값: 0)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
        help="inference batch 크기 (기본값: 16)",
    )
    return parser.parse_args()


def make_logger(log_path):
    """터미널과 파일에 같은 메시지를 기록한다."""
    logger = logging.getLogger(f"baseline_inference_{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def find_images(source):
    """입력이 이미지면 한 장, 폴더면 폴더 아래의 모든 이미지를 반환한다."""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"지원하지 않는 이미지 확장자입니다: {source}")
        return [source.resolve()]

    if source.is_dir():
        image_paths = sorted(
            path.resolve()
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"폴더에 이미지가 없습니다: {source}")
        return image_paths

    raise FileNotFoundError(f"입력 이미지 또는 폴더가 없습니다: {source}")


def check_duplicate_output_names(image_paths):
    """서로 다른 입력 이미지가 같은 출력 파일명을 덮어쓰지 않도록 확인한다."""
    names = [path.name for path in image_paths]
    if len(names) != len(set(names)):
        raise ValueError(
            "입력 폴더 안에 파일명이 같은 이미지가 있습니다. "
            "중복 파일명을 정리하거나 한 폴더씩 inference하세요."
        )


def write_yolo_txt(txt_path, detections):
    """class, 정규화 bbox, confidence를 YOLO TXT 형식으로 저장한다."""
    lines = []
    for detection in detections:
        lines.append(
            f"{detection['class_id']} "
            f"{detection['cx']:.6f} {detection['cy']:.6f} "
            f"{detection['width']:.6f} {detection['height']:.6f} "
            f"{detection['confidence']:.6f}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def get_detections(result):
    """Ultralytics 결과를 저장하기 쉬운 dict 목록으로 변환한다."""
    detections = []
    image_height, image_width = result.orig_shape

    if result.boxes is None:
        return detections

    pixel_boxes = result.boxes.xyxy.cpu().tolist()
    normalized_boxes = result.boxes.xywhn.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    class_ids = result.boxes.cls.cpu().tolist()

    for index in range(len(pixel_boxes)):
        xmin, ymin, xmax, ymax = pixel_boxes[index]
        center_x, center_y, width, height = normalized_boxes[index]
        class_id = int(class_ids[index])

        detections.append(
            {
                "detection_index": index + 1,
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": float(confidences[index]),
                "xmin": float(xmin),
                "ymin": float(ymin),
                "xmax": float(xmax),
                "ymax": float(ymax),
                "cx": float(center_x),
                "cy": float(center_y),
                "width": float(width),
                "height": float(height),
                "image_width": image_width,
                "image_height": image_height,
            }
        )

    return detections


def log_start(logger, model_path, source, output_dir, image_count, arguments):
    """재현에 필요한 실행 환경과 threshold를 로그 앞부분에 기록한다."""
    logger.info(f"[INFO] time_local: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    logger.info(f"[INFO] time_utc:   {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    logger.info(f"[INFO] PyTorch: {torch.__version__}")
    logger.info(f"[INFO] Ultralytics: {ultralytics.__version__}")
    logger.info(f"[INFO] model: {model_path}")
    logger.info(f"[INFO] source: {source}")
    logger.info(f"[INFO] image_count: {image_count:,}")
    logger.info(f"[INFO] output_directory: {output_dir}")
    logger.info(
        f"[INFO] imgsz={IMAGE_SIZE}, confidence={arguments.conf}, "
        f"IoU_argument={arguments.iou}, batch={arguments.batch}, device={arguments.device}"
    )


def main():
    arguments = parse_arguments()
    model_path = arguments.model.resolve()
    source = arguments.source.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    if not 0.0 <= arguments.conf <= 1.0:
        raise ValueError("--conf는 0~1 사이여야 합니다.")
    if not 0.0 <= arguments.iou <= 1.0:
        raise ValueError("--iou는 0~1 사이여야 합니다.")
    if arguments.batch <= 0:
        raise ValueError("--batch는 1 이상이어야 합니다.")

    image_paths = find_images(source)
    check_duplicate_output_names(image_paths)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_ROOT / f"baseline_inference_{timestamp}"
    annotated_dir = output_dir / "annotated"
    labels_dir = output_dir / "labels"
    annotated_dir.mkdir(parents=True, exist_ok=False)
    labels_dir.mkdir(parents=True, exist_ok=False)

    logger = make_logger(output_dir / "inference.log")
    log_start(logger, model_path, source, output_dir, len(image_paths), arguments)

    model = YOLO(str(model_path))
    end_to_end = bool(getattr(model.model, "end2end", False))
    logger.info(f"[INFO] model_end_to_end: {end_to_end}")
    if end_to_end:
        logger.info(
            "[INFO] YOLOv26 end-to-end 출력은 NMS-free이므로 --iou는 "
            "서로 다른 bbox 제거에 사용되지 않습니다."
        )

    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMAGE_SIZE,
        conf=arguments.conf,
        iou=arguments.iou,
        batch=arguments.batch,
        device=arguments.device,
        stream=True,
        verbose=False,
        save=False,
    )

    detection_csv_path = output_dir / "detections.csv"
    summary_csv_path = output_dir / "image_summary.csv"
    detection_fields = [
        "image_name",
        "detection_index",
        "class_id",
        "class_name",
        "confidence",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "cx",
        "cy",
        "width",
        "height",
        "image_width",
        "image_height",
    ]

    processed_count = 0
    total_detection_count = 0

    with detection_csv_path.open("w", newline="", encoding="utf-8") as detection_file:
        with summary_csv_path.open("w", newline="", encoding="utf-8") as summary_file:
            detection_writer = csv.DictWriter(detection_file, fieldnames=detection_fields)
            summary_writer = csv.DictWriter(
                summary_file,
                fieldnames=["image_name", "detection_count"],
            )
            detection_writer.writeheader()
            summary_writer.writeheader()

            for image_path, result in zip(image_paths, results):
                # 경로 목록을 batch inference하면 Ultralytics가 result.path를
                # image0.jpg처럼 바꿀 수 있으므로 실제 입력 파일명을 직접 사용한다.
                image_name = image_path.name
                detections = get_detections(result)

                for detection in detections:
                    detection_writer.writerow({"image_name": image_name, **detection})

                summary_writer.writerow(
                    {"image_name": image_name, "detection_count": len(detections)}
                )

                write_yolo_txt(labels_dir / f"{Path(image_name).stem}.txt", detections)

                annotated_image = result.plot()
                saved = cv2.imwrite(str(annotated_dir / image_name), annotated_image)
                if not saved:
                    raise IOError(f"시각화 이미지 저장 실패: {annotated_dir / image_name}")

                processed_count += 1
                total_detection_count += len(detections)

                if processed_count == 1 or processed_count % 25 == 0:
                    logger.info(
                        f"[LOG] {processed_count:04d}/{len(image_paths):04d} "
                        f"| image={image_name} | detections={len(detections)}"
                    )

    zero_detection_count = 0
    with summary_csv_path.open("r", encoding="utf-8") as summary_file:
        for row in csv.DictReader(summary_file):
            if int(row["detection_count"]) == 0:
                zero_detection_count += 1

    logger.info("[INFO] Inference finished")
    logger.info(f"[INFO] processed_images: {processed_count:,}")
    logger.info(f"[INFO] total_detections: {total_detection_count:,}")
    logger.info(f"[INFO] zero_detection_images: {zero_detection_count:,}")
    logger.info(f"[INFO] annotated_images: {annotated_dir}")
    logger.info(f"[INFO] YOLO_prediction_labels: {labels_dir}")
    logger.info(f"[INFO] detection_CSV: {detection_csv_path}")
    logger.info(f"[INFO] image_summary_CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
