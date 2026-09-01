"""
실행방법:
python 02_quarter/scripts/01_preprocessing/preprocess_quarter.py \
  --splits train val \
  --margin-ratio 0.05 \
  --min-visible-ratio 0.50 \
  --image-size 640
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


SCRIPT_DIR = Path(__file__).resolve().parent
QUARTER_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(QUARTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(QUARTER_SCRIPTS_DIR))

from quarter_utils import (  # noqa: E402
    CLASS_ID,
    box_to_letterbox,
    box_to_yolo,
    clip_box,
    crop_image,
    extract_boxes_from_json,
    intersect_box,
    letterbox,
    load_json,
    make_margin_quarters,
    matching_pairs,
    validate_ratio,
    yolo_to_box,
)


# 1. 경로와 기본 하이퍼파라미터
PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUARTER_ROOT = PROJECT_ROOT / "02_quarter"
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "LICENSE_PLATE_DATA_ROOT",
        "/mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset",
    )
)
DEFAULT_OUTPUT_ROOT = QUARTER_ROOT / "data" / "preprocessed"
DEFAULT_VISUALIZATION_ROOT = QUARTER_ROOT / "data" / "visualization" / "crops"
LOGS_ROOT = QUARTER_ROOT / "logs"

MARGIN_RATIO = 0.05
MIN_VISIBLE_RATIO = 0.50
IMAGE_SIZE = 640
VISUALIZE_COUNT = 5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="마진 기반 4분할 Train/Val 전처리")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--visualization-root", type=Path, default=DEFAULT_VISUALIZATION_ROOT)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--margin-ratio", type=float, default=MARGIN_RATIO)
    parser.add_argument("--min-visible-ratio", type=float, default=MIN_VISIBLE_RATIO)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--visualize-count", type=int, default=VISUALIZE_COUNT)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="split마다 앞에서 N장만 처리(좌표 검증용). 생략하면 전체 처리",
    )
    return parser.parse_args()


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"quarter_preprocess_{path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def draw_split_overview(image, boxes, regions, output_path: Path) -> None:
    canvas = image.copy()
    colors = [(255, 80, 80), (80, 180, 255), (180, 80, 255), (80, 220, 120)]
    for region, color in zip(regions, colors):
        cv2.rectangle(canvas, (region.x1, region.y1), (region.x2 - 1, region.y2 - 1), color, 3)
        cv2.putText(
            canvas,
            f"q{region.index} {region.name}",
            (region.x1 + 12, region.y1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(canvas, f"GT {index}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"분할 시각화 저장 실패: {output_path}")


def draw_letterbox_boxes(image, yolo_boxes, output_path: Path) -> None:
    canvas = image.copy()
    for index, yolo_box in enumerate(yolo_boxes, start=1):
        box = yolo_to_box(yolo_box, canvas.shape[1], canvas.shape[0])
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"plate {index}", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"crop 시각화 저장 실패: {output_path}")


def process_sample(
    stem: str,
    image_path: Path,
    json_path: Path,
    split: str,
    arguments: argparse.Namespace,
    visualize: bool,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")
    image_height, image_width = image.shape[:2]
    boxes = []
    for raw_box in extract_boxes_from_json(load_json(json_path)):
        clipped = clip_box(raw_box, image_width, image_height)
        if clipped is not None:
            boxes.append(clipped)

    regions = make_margin_quarters(image_width, image_height, arguments.margin_ratio)
    if visualize:
        draw_split_overview(
            image,
            boxes,
            regions,
            arguments.visualization_root / split / f"{stem}_split.jpg",
        )

    retained_instances = 0
    boxes_seen_in_any_crop: set[int] = set()
    empty_crops = 0
    for region in regions:
        crop = crop_image(image, region)
        processed_image, meta = letterbox(crop, arguments.image_size)
        yolo_boxes = []
        for box_index, original_box in enumerate(boxes):
            local_box, visible_ratio = intersect_box(original_box, region)
            if local_box is None or visible_ratio < arguments.min_visible_ratio:
                continue
            transformed = box_to_letterbox(local_box, meta)
            if transformed is None:
                continue
            yolo_box = box_to_yolo(transformed, arguments.image_size, arguments.image_size)
            if not (
                0.0 <= yolo_box[0] <= 1.0
                and 0.0 <= yolo_box[1] <= 1.0
                and 0.0 < yolo_box[2] <= 1.0
                and 0.0 < yolo_box[3] <= 1.0
            ):
                raise ValueError(f"YOLO 좌표 범위 오류: {stem} q{region.index} {yolo_box}")
            yolo_boxes.append(yolo_box)
            retained_instances += 1
            boxes_seen_in_any_crop.add(box_index)

        output_stem = f"{stem}_q{region.index}"
        image_output = arguments.output_root / "images" / split / f"{output_stem}.jpg"
        label_output = arguments.output_root / "labels" / split / f"{output_stem}.txt"
        image_output.parent.mkdir(parents=True, exist_ok=True)
        label_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_output), processed_image):
            raise IOError(f"전처리 이미지 저장 실패: {image_output}")
        lines = [
            f"{CLASS_ID} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
            for cx, cy, width, height in yolo_boxes
        ]
        label_output.write_text("\n".join(lines), encoding="utf-8")
        if not yolo_boxes:
            empty_crops += 1
        if visualize:
            draw_letterbox_boxes(
                processed_image,
                yolo_boxes,
                arguments.visualization_root / split / f"{output_stem}_letterbox.jpg",
            )

    return {
        "source_boxes": len(boxes),
        "retained_label_instances": retained_instances,
        "source_boxes_not_retained": len(boxes) - len(boxes_seen_in_any_crop),
        "generated_crops": 4,
        "empty_crops": empty_crops,
    }


def process_split(split: str, arguments: argparse.Namespace, logger: logging.Logger, error_writer) -> dict:
    pairs = matching_pairs(arguments.data_root, split)
    total_pairs = len(pairs)
    if arguments.max_samples is not None:
        pairs = pairs[: arguments.max_samples]
    stats = {
        "available_pairs": total_pairs,
        "attempted_images": len(pairs),
        "successful_images": 0,
        "failed_images": 0,
        "source_boxes": 0,
        "retained_label_instances": 0,
        "source_boxes_not_retained": 0,
        "generated_crops": 0,
        "empty_crops": 0,
    }
    logger.info(f"[INFO] {split}: available_pairs={total_pairs:,}, selected={len(pairs):,}")
    for number, (stem, image_path, json_path) in enumerate(pairs, start=1):
        try:
            sample_stats = process_sample(
                stem,
                image_path,
                json_path,
                split,
                arguments,
                visualize=number <= arguments.visualize_count,
            )
            stats["successful_images"] += 1
            for key, value in sample_stats.items():
                stats[key] += value
        except Exception as error:  # 한 파일 오류가 전체 처리를 중단하지 않게 기록
            stats["failed_images"] += 1
            error_writer.writerow(
                {
                    "split": split,
                    "stem": stem,
                    "image_path": image_path,
                    "json_path": json_path,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            logger.exception(f"[ERROR] {split}/{stem} 전처리 실패")
        if number == 1 or number % 1000 == 0 or number == len(pairs):
            logger.info(f"[LOG] {split} {number:,}/{len(pairs):,}")
    return stats


def main() -> None:
    arguments = parse_arguments()
    arguments.data_root = arguments.data_root.resolve()
    arguments.output_root = arguments.output_root.resolve()
    arguments.visualization_root = arguments.visualization_root.resolve()
    validate_ratio("margin_ratio", arguments.margin_ratio, upper_exclusive=0.5)
    validate_ratio("min_visible_ratio", arguments.min_visible_ratio)
    if arguments.image_size <= 0:
        raise ValueError("--image-size는 1 이상이어야 합니다.")
    if arguments.visualize_count < 0:
        raise ValueError("--visualize-count는 0 이상이어야 합니다.")
    if arguments.max_samples is not None and arguments.max_samples <= 0:
        raise ValueError("--max-samples는 1 이상이어야 합니다.")

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_ROOT / f"preprocess_{timestamp}.log"
    error_path = LOGS_ROOT / f"preprocess_errors_{timestamp}.csv"
    stats_path = arguments.output_root / f"preprocessing_stats_{timestamp}.json"
    logger = make_logger(log_path)
    logger.info(f"[INFO] time_local: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    logger.info(f"[INFO] time_utc:   {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    logger.info(f"[INFO] data_root: {arguments.data_root}")
    logger.info(f"[INFO] output_root: {arguments.output_root}")
    logger.info(
        f"[INFO] splits={arguments.splits}, margin_ratio={arguments.margin_ratio}, "
        f"min_visible_ratio={arguments.min_visible_ratio}, image_size={arguments.image_size}"
    )

    all_stats = {}
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with error_path.open("w", newline="", encoding="utf-8") as error_file:
        fields = ["split", "stem", "image_path", "json_path", "error_type", "message"]
        error_writer = csv.DictWriter(error_file, fieldnames=fields)
        error_writer.writeheader()
        for split in arguments.splits:
            all_stats[split] = process_split(split, arguments, logger, error_writer)

    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(arguments.data_root),
        "output_root": str(arguments.output_root),
        "margin_ratio": arguments.margin_ratio,
        "margin_definition": "each inner edge expands by original_dimension * margin_ratio",
        "min_visible_ratio": arguments.min_visible_ratio,
        "image_size": arguments.image_size,
        "max_samples_per_split": arguments.max_samples,
        "splits": all_stats,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[RESULT] preprocessing_stats: {stats_path}")
    logger.info(f"[RESULT] error_csv: {error_path}")
    for split, stats in all_stats.items():
        logger.info(f"[RESULT] {split}: {stats}")


if __name__ == "__main__":
    main()
