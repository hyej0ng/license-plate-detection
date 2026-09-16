"""차량 탐지 → 차량 crop → 번호판 탐지 → 원본 좌표 복원의 2단계 inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "03_two_stage"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from two_stage_utils import make_stage_logger  # noqa: E402


DEFAULT_SOURCE = PROJECT_DIR / "data" / "preprocessed" / "images" / "test"
OUTPUT_ROOT = PROJECT_DIR / "results" / "predictions"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PLATE_CLASS_ID = 0
CAR_CLASS_ID = 1
CLASS_NAMES = {PLATE_CLASS_ID: "plate", CAR_CLASS_ID: "car"}
COLORS = {PLATE_CLASS_ID: (0, 255, 0), CAR_CLASS_ID: (255, 120, 0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv26n vehicle-first two-stage inference")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--conf", "--save-conf", dest="save_conf", type=float, default=0.001,
        help="저장할 vehicle/plate 최소 confidence; AP 계산을 위해 0.001 권장",
    )
    parser.add_argument(
        "--vehicle-crop-conf", type=float, default=0.25,
        help="2단계 번호판 탐지를 실행할 vehicle의 최소 confidence",
    )
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO predict의 IoU 인자")
    parser.add_argument("--global-nms-iou", type=float, default=0.50, help="복원된 plate 간 global NMS IoU")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="차량 crop 추론 batch")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int, help="흐름 확인용 이미지 수 제한")
    return parser.parse_args()


def find_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source.resolve()]
    if source.is_dir():
        paths = sorted(
            path.resolve() for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if paths:
            names = [path.name for path in paths]
            if len(names) != len(set(names)):
                raise ValueError("입력에 중복 이미지 파일명이 있습니다.")
            return paths
    raise FileNotFoundError(f"입력 이미지를 찾지 못했습니다: {source}")


def box_iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def global_nms(detections: list[dict], threshold: float) -> list[dict]:
    """여러 vehicle crop에서 복원된 동일 번호판을 confidence 순으로 통합한다."""
    remaining = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        current_box = tuple(current[key] for key in ("xmin", "ymin", "xmax", "ymax"))
        remaining = [
            candidate for candidate in remaining
            if box_iou(current_box, tuple(candidate[key] for key in ("xmin", "ymin", "xmax", "ymax"))) <= threshold
        ]
    return kept


def make_row(
    image_name: str,
    class_id: int,
    confidence: float,
    box: tuple[float, ...],
    image_width: int,
    image_height: int,
    stage: str,
    parent_vehicle_index: int | None,
) -> dict:
    x1, y1, x2, y2 = box
    return {
        "image_name": image_name, "detection_index": 0,
        "class_id": class_id, "class_name": CLASS_NAMES[class_id], "confidence": confidence,
        "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
        "cx": ((x1 + x2) / 2) / image_width, "cy": ((y1 + y2) / 2) / image_height,
        "width": (x2 - x1) / image_width, "height": (y2 - y1) / image_height,
        "image_width": image_width, "image_height": image_height,
        "stage": stage, "parent_vehicle_index": parent_vehicle_index if parent_vehicle_index is not None else "",
    }


def vehicle_rows_from_result(result, image_name: str, image_width: int, image_height: int) -> list[dict]:
    rows = []
    if result.boxes is None:
        return rows
    for xyxy, confidence, class_value in zip(
        result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()
    ):
        if int(class_value) != CAR_CLASS_ID:
            continue
        x1, y1, x2, y2 = xyxy
        x1, x2 = max(0.0, min(x1, image_width)), max(0.0, min(x2, image_width))
        y1, y2 = max(0.0, min(y1, image_height)), max(0.0, min(y2, image_height))
        if x2 > x1 and y2 > y1:
            rows.append(make_row(image_name, CAR_CLASS_ID, float(confidence), (x1, y1, x2, y2),
                                 image_width, image_height, "vehicle_full_image", None))
    return rows


def detect_plates_in_crops(
    model: YOLO,
    image: np.ndarray,
    image_name: str,
    vehicles: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict], int]:
    height, width = image.shape[:2]
    crops = []
    crop_metadata = []
    for vehicle_index, vehicle in enumerate(vehicles, 1):
        if vehicle["confidence"] < args.vehicle_crop_conf:
            continue
        x1 = max(0, int(math.floor(vehicle["xmin"])))
        y1 = max(0, int(math.floor(vehicle["ymin"])))
        x2 = min(width, int(math.ceil(vehicle["xmax"])))
        y2 = min(height, int(math.ceil(vehicle["ymax"])))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(image[y1:y2, x1:x2].copy())
        crop_metadata.append((vehicle_index, x1, y1))
    if not crops:
        return [], 0

    results = model.predict(
        source=crops, imgsz=args.image_size, conf=args.save_conf, iou=args.iou,
        batch=args.batch, device=args.device, verbose=False, save=False,
    )
    plates = []
    for (vehicle_index, offset_x, offset_y), result in zip(crop_metadata, results):
        if result.boxes is None:
            continue
        for xyxy, confidence, class_value in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()
        ):
            if int(class_value) != PLATE_CLASS_ID:
                continue
            x1, y1, x2, y2 = xyxy
            original_box = (
                max(0.0, min(width, x1 + offset_x)), max(0.0, min(height, y1 + offset_y)),
                max(0.0, min(width, x2 + offset_x)), max(0.0, min(height, y2 + offset_y)),
            )
            if original_box[2] > original_box[0] and original_box[3] > original_box[1]:
                plates.append(make_row(image_name, PLATE_CLASS_ID, float(confidence), original_box,
                                       width, height, "plate_vehicle_crop", vehicle_index))
    return global_nms(plates, args.global_nms_iou), len(crops)


def draw_detections(image: np.ndarray, rows: list[dict]) -> np.ndarray:
    output = image.copy()
    for row in rows:
        x1, y1, x2, y2 = (int(round(row[key])) for key in ("xmin", "ymin", "xmax", "ymax"))
        color = COLORS[row["class_id"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"{row['class_name']} {row['confidence']:.2f}",
                    (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return output


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    logger, _ = make_stage_logger(PROJECT_DIR, "inference", timestamp)
    for name, value in (("save-conf", args.save_conf), ("vehicle-crop-conf", args.vehicle_crop_conf),
                        ("iou", args.iou), ("global-nms-iou", args.global_nms_iou)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name}는 0~1 사이여야 합니다.")
    if args.batch <= 0 or args.image_size <= 0 or (args.max_samples is not None and args.max_samples <= 0):
        raise ValueError("batch/image-size/max-samples는 양수여야 합니다.")
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"모델이 없습니다: {model_path}")
    images = find_images(args.source.resolve())
    if args.max_samples is not None:
        images = images[: args.max_samples]
    output = (args.output_dir or OUTPUT_ROOT / f"two_stage_inference_{timestamp}").resolve()
    if output.exists():
        raise FileExistsError(f"출력 폴더가 이미 있습니다: {output}")
    annotated, labels = output / "annotated", output / "labels"
    annotated.mkdir(parents=True)
    labels.mkdir(parents=True)

    config = {
        "method": "vehicle_crop_two_stage", "model": str(model_path),
        "source": str(args.source.resolve()), "image_count": len(images), "image_size": args.image_size,
        "confidence": args.save_conf, "vehicle_crop_confidence": args.vehicle_crop_conf,
        "iou_argument": args.iou, "global_plate_nms_iou": args.global_nms_iou,
        "batch": args.batch, "device": args.device, "limited_run": args.max_samples is not None,
    }
    logger.info(f"[INFO] config={config}")
    logger.info(f"[INFO] output={output}")
    (output / "inference_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    detection_fields = [
        "image_name", "detection_index", "class_id", "class_name", "confidence",
        "xmin", "ymin", "xmax", "ymax", "cx", "cy", "width", "height",
        "image_width", "image_height", "stage", "parent_vehicle_index",
    ]
    summary_fields = [
        "image_name", "image_width", "image_height", "detection_count",
        "vehicle_count", "cropped_vehicle_count", "plate_count",
    ]
    vehicle_model = YOLO(str(model_path))
    plate_model = YOLO(str(model_path))
    normalized_names = {int(key): str(value) for key, value in vehicle_model.names.items()}
    if normalized_names != CLASS_NAMES:
        raise ValueError(f"모델 class는 {CLASS_NAMES}이어야 합니다: {normalized_names}")
    vehicle_results = vehicle_model.predict(
        source=[str(path) for path in images], imgsz=args.image_size, conf=args.save_conf,
        iou=args.iou, batch=args.batch, device=args.device, stream=True, verbose=False, save=False,
    )

    total_vehicles = total_crops = total_plates = 0
    with (output / "detections.csv").open("w", newline="", encoding="utf-8") as detection_file, \
         (output / "image_summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
        detection_writer = csv.DictWriter(detection_file, fieldnames=detection_fields)
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        detection_writer.writeheader()
        summary_writer.writeheader()
        for number, (image_path, vehicle_result) in enumerate(zip(images, vehicle_results), 1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")
            height, width = image.shape[:2]
            vehicles = vehicle_rows_from_result(vehicle_result, image_path.name, width, height)
            plates, crop_count = detect_plates_in_crops(plate_model, image, image_path.name, vehicles, args)
            rows = vehicles + plates
            for index, row in enumerate(rows, 1):
                row["detection_index"] = index
            detection_writer.writerows(rows)
            summary_writer.writerow({
                "image_name": image_path.name, "image_width": width, "image_height": height,
                "detection_count": len(rows), "vehicle_count": len(vehicles),
                "cropped_vehicle_count": crop_count, "plate_count": len(plates),
            })
            label_lines = [
                f"{row['class_id']} {row['cx']:.6f} {row['cy']:.6f} {row['width']:.6f} "
                f"{row['height']:.6f} {row['confidence']:.6f}" for row in rows
            ]
            (labels / f"{image_path.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
            if not cv2.imwrite(str(annotated / image_path.name), draw_detections(image, rows)):
                raise IOError(f"시각화 저장 실패: {image_path.name}")
            total_vehicles += len(vehicles)
            total_crops += crop_count
            total_plates += len(plates)
            if number == 1 or number % 100 == 0:
                logger.info(f"[LOG] inference {number:,}/{len(images):,}")
    logger.info(f"[RESULT] output={output}")
    logger.info(f"[RESULT] images={len(images):,}, vehicles={total_vehicles:,}, "
                f"vehicle_crops={total_crops:,}, plates_after_nms={total_plates:,}")


if __name__ == "__main__":
    main()
