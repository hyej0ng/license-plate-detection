"""
원본 Test 이미지를 메모리에서 4분할하고 원본 좌표 Global NMS를 수행함
실행 방법:
python 02_quarter/scripts/03_inference/inference_quarter.py \
  --model /home/hyejong/landing_pjt/02_quarter/runs/quarter_yolo26n_20260903-152325/weights/best.pt \
  --data-root /mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset \
  --margin-ratio 0.05 \
  --confidence 0.25 \
  --nms-iou 0.50 \
  --image-size 640 \
  --device 6
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
QUARTER_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(QUARTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(QUARTER_SCRIPTS_DIR))

from quarter_utils import (  # noqa: E402
    IMAGE_EXTENSIONS,
    box_from_letterbox,
    build_stem_index,
    crop_image,
    dataset_split_roots,
    global_nms,
    letterbox,
    local_box_to_original,
    make_margin_quarters,
    validate_ratio,
)


# 1. 경로와 inference 하이퍼파라미터
PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUARTER_ROOT = PROJECT_ROOT / "02_quarter"
DEFAULT_DATA_ROOT = Path(
    os.environ.get("LICENSE_PLATE_DATA_ROOT", "/mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset")
)
OUTPUT_ROOT = QUARTER_ROOT / "results" / "predictions"

IMAGE_SIZE = 640
MARGIN_RATIO = 0.05
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.50
DEVICE = "0"


def latest_quarter_model() -> Path | None:
    candidates = sorted((QUARTER_ROOT / "runs").glob("quarter_yolo26n_*/weights/best.pt"))
    return candidates[-1] if candidates else None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv26n 마진 기반 4분할 Test inference")
    parser.add_argument("--model", type=Path, default=latest_quarter_model())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source", type=Path, help="별도 이미지 한 장/폴더. 생략하면 원본 Test")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--margin-ratio", type=float, default=MARGIN_RATIO)
    parser.add_argument("--confidence", "--conf", dest="confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU_THRESHOLD)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--max-samples", type=int, help="앞에서 N장만 inference(흐름 검증용)")
    return parser.parse_args()


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"quarter_inference_{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def find_images(source: Path | None, data_root: Path) -> list[Path]:
    if source is None:
        image_root, _ = dataset_split_roots(data_root, "test")
        return [path for _, path in sorted(build_stem_index(image_root, IMAGE_EXTENSIONS).items())]
    source = source.resolve()
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        images = sorted(path.resolve() for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            stems = [path.stem for path in images]
            if len(stems) != len(set(stems)):
                raise ValueError("source 안에 동일 stem 이미지가 있어 CSV 연결이 모호합니다.")
            return images
    raise FileNotFoundError(f"입력 이미지를 찾지 못했습니다: {source}")


def draw_detections(image, detections: list[dict], title: str) -> object:
    canvas = image.copy()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = (int(round(value)) for value in detection["box"])
        color = (0, 255, 0) if title == "after_global_nms" else (0, 180, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{index} {detection['confidence']:.2f} q{detection['crop_index']}"
        cv2.putText(canvas, text, (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{title}: {len(detections)} boxes", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def detection_row(image_name: str, detection_index: int, detection: dict, width: int, height: int) -> dict:
    x1, y1, x2, y2 = detection["box"]
    return {
        "image_name": image_name,
        "detection_index": detection_index,
        "class_id": detection["class_id"],
        "class_name": detection["class_name"],
        "confidence": detection["confidence"],
        "xmin": x1,
        "ymin": y1,
        "xmax": x2,
        "ymax": y2,
        "image_width": width,
        "image_height": height,
        "crop_index": detection["crop_index"],
        "crop_name": detection["crop_name"],
    }


def main() -> None:
    arguments = parse_arguments()
    if arguments.model is None:
        raise FileNotFoundError("quarter best.pt가 없습니다. --model로 가중치를 지정하세요.")
    model_path = arguments.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    validate_ratio("margin_ratio", arguments.margin_ratio, upper_exclusive=0.5)
    validate_ratio("confidence_threshold", arguments.confidence)
    validate_ratio("nms_iou_threshold", arguments.nms_iou)
    if arguments.image_size <= 0:
        raise ValueError("--image-size는 1 이상이어야 합니다.")
    if arguments.max_samples is not None and arguments.max_samples <= 0:
        raise ValueError("--max-samples는 1 이상이어야 합니다.")

    data_root = arguments.data_root.resolve()
    image_paths = find_images(arguments.source, data_root)
    if arguments.max_samples is not None:
        image_paths = image_paths[: arguments.max_samples]
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = arguments.output_root.resolve() / f"quarter_inference_{timestamp}"
    before_dir = output_dir / "before_nms"
    annotated_dir = output_dir / "annotated"
    before_dir.mkdir(parents=True, exist_ok=False)
    annotated_dir.mkdir(parents=True, exist_ok=False)
    logger = make_logger(output_dir / "inference.log")
    logger.info(f"[INFO] time_local: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    logger.info(f"[INFO] time_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    logger.info(f"[INFO] PyTorch={torch.__version__}, Ultralytics={ultralytics.__version__}")
    logger.info(f"[INFO] model={model_path}, source={arguments.source or 'raw Test'}, images={len(image_paths):,}")
    logger.info(
        f"[INFO] margin_ratio={arguments.margin_ratio}, image_size={arguments.image_size}, "
        f"confidence={arguments.confidence}, global_nms_iou={arguments.nms_iou}, device={arguments.device}"
    )

    model = YOLO(str(model_path))
    before_rows = []
    final_rows = []
    summary_rows = []
    for image_number, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")
        image_height, image_width = image.shape[:2]
        regions = make_margin_quarters(image_width, image_height, arguments.margin_ratio)
        model_inputs = []
        metas = []
        for region in regions:
            processed, meta = letterbox(crop_image(image, region), arguments.image_size)
            model_inputs.append(processed)
            metas.append(meta)

        results = model.predict(
            source=model_inputs,
            imgsz=arguments.image_size,
            conf=arguments.confidence,
            iou=arguments.nms_iou,
            batch=4,
            device=arguments.device,
            verbose=False,
            save=False,
            stream=False,
        )
        candidates = []
        for region, meta, result in zip(regions, metas, results):
            if result.boxes is None:
                continue
            for box, confidence, class_id_value in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                local_box = box_from_letterbox(tuple(float(value) for value in box), meta)
                if local_box is None:
                    continue
                original_box = local_box_to_original(local_box, region, image_width, image_height)
                if original_box is None:
                    continue
                class_id = int(class_id_value)
                candidates.append(
                    {
                        "box": original_box,
                        "confidence": float(confidence),
                        "class_id": class_id,
                        "class_name": str(result.names[class_id]),
                        "crop_index": region.index,
                        "crop_name": region.name,
                    }
                )
        final_detections = global_nms(candidates, arguments.nms_iou)
        for index, detection in enumerate(candidates, start=1):
            before_rows.append(detection_row(image_path.name, index, detection, image_width, image_height))
        for index, detection in enumerate(final_detections, start=1):
            final_rows.append(detection_row(image_path.name, index, detection, image_width, image_height))
        summary_rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "before_nms_count": len(candidates),
                "detection_count": len(final_detections),
                "image_width": image_width,
                "image_height": image_height,
            }
        )
        before_image = draw_detections(image, candidates, "before_global_nms")
        final_image = draw_detections(image, final_detections, "after_global_nms")
        if not cv2.imwrite(str(before_dir / image_path.name), before_image):
            raise IOError(f"NMS 전 이미지 저장 실패: {image_path.name}")
        if not cv2.imwrite(str(annotated_dir / image_path.name), final_image):
            raise IOError(f"최종 이미지 저장 실패: {image_path.name}")
        if image_number == 1 or image_number % 25 == 0 or image_number == len(image_paths):
            logger.info(
                f"[LOG] {image_number:04d}/{len(image_paths):04d} | {image_path.name} "
                f"| before={len(candidates)} after={len(final_detections)}"
            )

    fields = [
        "image_name", "detection_index", "class_id", "class_name", "confidence",
        "xmin", "ymin", "xmax", "ymax", "image_width", "image_height", "crop_index", "crop_name",
    ]
    for path, rows in ((output_dir / "detections_before_nms.csv", before_rows), (output_dir / "detections.csv", final_rows)):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "image_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    config = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model_path),
        "data_root": str(data_root),
        "source": str(arguments.source.resolve()) if arguments.source else None,
        "image_count": len(image_paths),
        "margin_ratio": arguments.margin_ratio,
        "margin_definition": "each inner edge expands by original_dimension * margin_ratio",
        "image_size": arguments.image_size,
        "confidence_threshold": arguments.confidence,
        "nms_iou_threshold": arguments.nms_iou,
        "device": arguments.device,
    }
    (output_dir / "inference_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[RESULT] output_directory: {output_dir}")
    logger.info(f"[RESULT] detections_after_nms: {len(final_rows):,}")
    logger.info(f"[RESULT] detections_before_nms: {len(before_rows):,}")


if __name__ == "__main__":
    main()
