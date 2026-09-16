"""Roboflow YOLO export를 640 letterbox 및 고정 class ID로 전처리한다."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from two_stage_utils import (  # noqa: E402
    IMAGE_EXTENSIONS,
    image_index,
    make_stage_logger,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


DEFAULT_RAW = PROJECT_DIR / "data" / "raw" / "roboflow_v2"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "preprocessed"
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "license_plate_vehicle.yaml"
TARGET_NAMES = {0: "plate", 1: "car"}
COLORS = {0: (0, 255, 0), 1: (255, 120, 0)}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_names(value) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    raise ValueError("원본 data.yaml의 names가 list 또는 dict가 아닙니다.")


def class_remap(source_names: dict[int, str]) -> dict[int, int]:
    aliases = {"licenseplate": 0, "plate": 0, "vehicle": 1, "car": 1}
    remap = {}
    for source_id, source_name in source_names.items():
        normalized = normalize_name(source_name)
        if normalized not in aliases:
            raise ValueError(f"지원하지 않는 원본 클래스입니다: {source_id}: {source_name}")
        remap[source_id] = aliases[normalized]
    if set(remap.values()) != {0, 1}:
        raise ValueError(f"license plate와 vehicle 두 클래스가 모두 필요합니다: {source_names}")
    return remap


def resolve_split(source_yaml: Path, config: dict, split: str) -> tuple[Path, Path]:
    candidates = ("val", "valid") if split == "val" else (split,)
    value = next((config[key] for key in candidates if key in config), None)
    if value is None:
        fallback = source_yaml.parent / ("valid" if split == "val" else split) / "images"
        image_dir = fallback.resolve()
    else:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"여러 이미지 경로는 지원하지 않습니다: {split}={value}")
            value = value[0]
        image_dir = Path(value)
        if not image_dir.is_absolute():
            image_dir = (source_yaml.parent / image_dir).resolve()
    if not image_dir.is_dir():
        fallback_name = "valid" if split == "val" else split
        local_fallback = (source_yaml.parent / fallback_name / "images").resolve()
        if local_fallback.is_dir():
            image_dir = local_fallback
    label_dir = image_dir.parent / "labels" if image_dir.name == "images" else None
    if label_dir is None or not label_dir.is_dir():
        fallback_name = "valid" if split == "val" else split
        label_dir = (source_yaml.parent / fallback_name / "labels").resolve()
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"{split} images/labels를 찾지 못했습니다: {image_dir}, {label_dir}")
    return image_dir, label_dir


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, scale, left, top


def read_source_labels(
    path: Path,
    remap: dict[int, int],
) -> tuple[list[tuple[int, tuple[float, ...]]], list[dict]]:
    boxes = []
    rejected = []
    label_text = None
    for attempt in range(3):
        try:
            label_text = path.read_text(encoding="utf-8")
            break
        except FileNotFoundError:
            if attempt == 2:
                raise FileNotFoundError(f"이미지와 같은 stem의 라벨이 없습니다: {path}") from None
            time.sleep(0.25)
    assert label_text is not None
    for line_number, line in enumerate(label_text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"탐지 bbox가 아닌 YOLO 라벨입니다: {path}:{line_number}")
        source_id = int(parts[0])
        if source_id not in remap:
            raise ValueError(f"data.yaml에 없는 class ID: {path}:{line_number}: {source_id}")
        values = tuple(map(float, parts[1:]))
        if not all(0 <= value <= 1 for value in values):
            raise ValueError(f"잘못된 normalized bbox: {path}:{line_number}: {values}")
        if values[2] <= 0 or values[3] <= 0:
            rejected.append({
                "path": str(path),
                "line": line_number,
                "source_class_id": source_id,
                "target_class_id": remap[source_id],
                "values": values,
                "reason": "zero_width_or_height",
            })
            continue
        boxes.append((remap[source_id], yolo_to_xyxy(*values)))
    return boxes, rejected


def draw(image: np.ndarray, boxes: list[tuple[int, tuple[float, ...]]]) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    for class_id, box in boxes:
        x1, y1, x2, y2 = box
        points = tuple(int(round(value)) for value in (x1 * width, y1 * height, x2 * width, y2 * height))
        cv2.rectangle(result, points[:2], points[2:], COLORS[class_id], 2)
        cv2.putText(result, TARGET_NAMES[class_id], (points[0], max(18, points[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 1, cv2.LINE_AA)
    return result


def process_image(
    image_path: Path,
    label_path: Path,
    output_image: Path,
    output_label: Path,
    remap: dict[int, int],
    image_size: int,
) -> tuple[
    list[tuple[int, tuple[float, ...]]],
    list[tuple[int, tuple[float, ...]]],
    list[dict],
]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")
    height, width = image.shape[:2]
    source_boxes, rejected = read_source_labels(label_path, remap)
    output, scale, left, top = letterbox(image, image_size)
    transformed = []
    lines = []
    for class_id, normalized_box in source_boxes:
        x1n, y1n, x2n, y2n = normalized_box
        x1, y1 = x1n * width, y1n * height
        x2, y2 = x2n * width, y2n * height
        x1 = max(0.0, min(float(image_size), x1 * scale + left))
        y1 = max(0.0, min(float(image_size), y1 * scale + top))
        x2 = max(0.0, min(float(image_size), x2 * scale + left))
        y2 = max(0.0, min(float(image_size), y2 * scale + top))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"letterbox 후 bbox가 사라졌습니다: {image_path}: {normalized_box}")
        new_box = (x1 / image_size, y1 / image_size, x2 / image_size, y2 / image_size)
        cx, cy, box_width, box_height = xyxy_to_yolo(new_box)
        transformed.append((class_id, new_box))
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {box_width:.6f} {box_height:.6f}")
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image), output):
        raise IOError(f"이미지 저장 실패: {output_image}")
    output_label.write_text("\n".join(lines), encoding="utf-8")
    return source_boxes, transformed, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roboflow YOLO 데이터 letterbox 전처리")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-samples", type=int, help="split별 처리 제한(동작 확인 전용)")
    parser.add_argument("--visualize-count", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100, help="진행 로그를 남길 이미지 간격")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="이미 완료된 이미지/라벨 쌍을 검증 후 건너뛰고 중단 지점부터 재개",
    )
    parser.add_argument(
        "--test-from-val-ratio",
        type=float,
        default=0.5,
        help="원본 test가 없을 때 valid 중 test로 사용할 비율(default: 0.5)",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="valid→val/test 분할 시드")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    logger, _ = make_stage_logger(PROJECT_DIR, "preprocess", timestamp)
    if (
        args.image_size <= 0
        or args.visualize_count < 0
        or args.log_interval <= 0
        or (args.max_samples is not None and args.max_samples <= 0)
    ):
        raise ValueError("image-size/max-samples/log-interval은 양수, visualize-count는 0 이상이어야 합니다.")
    if not 0.0 < args.test_from_val_ratio < 1.0:
        raise ValueError("test-from-val-ratio는 0과 1 사이여야 합니다.")
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    source_yaml = raw_root / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Roboflow data.yaml이 없습니다: {source_yaml}")
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source_names = parse_names(config.get("names"))
    remap = class_remap(source_names)
    logger.info(f"[INFO] raw_root={raw_root}")
    logger.info(f"[INFO] output_root={output_root}")
    logger.info(f"[INFO] image_size={args.image_size}, splits={list(args.splits)}")
    logger.info(f"[INFO] class remap: {[(key, source_names[key], value) for key, value in remap.items()]}")

    split_sources: dict[str, tuple[Path, Path, set[str] | None]] = {}
    valid_partition = None
    if "test" in args.splits:
        try:
            test_image_dir, test_label_dir = resolve_split(source_yaml, config, "test")
            split_sources["test"] = (test_image_dir, test_label_dir, None)
        except FileNotFoundError:
            valid_image_dir, valid_label_dir = resolve_split(source_yaml, config, "val")
            valid_stems = sorted(image_index(valid_image_dir))
            if len(valid_stems) < 2:
                raise ValueError("test를 만들 원본 valid 이미지가 2장 미만입니다.")
            random.Random(args.split_seed).shuffle(valid_stems)
            test_count = max(1, min(len(valid_stems) - 1, round(len(valid_stems) * args.test_from_val_ratio)))
            test_stems = set(valid_stems[:test_count])
            val_stems = set(valid_stems[test_count:])
            split_sources["test"] = (valid_image_dir, valid_label_dir, test_stems)
            split_sources["val"] = (valid_image_dir, valid_label_dir, val_stems)
            valid_partition = {
                "reason": "source test split missing",
                "source_split": "valid",
                "seed": args.split_seed,
                "test_ratio": args.test_from_val_ratio,
                "val_images": len(val_stems),
                "test_images": len(test_stems),
            }
            logger.warning(
                "[WARNING] 원본 test split이 없어 valid를 고정 시드로 val/test로 나눕니다: "
                f"val={len(val_stems)}, test={len(test_stems)}, seed={args.split_seed}"
            )

    stats = {
        "source": str(raw_root),
        "output": str(output_root),
        "image_size": args.image_size,
        "limited_run": args.max_samples is not None,
        "valid_partition": valid_partition,
        "splits": {},
    }
    for split in args.splits:
        if split in split_sources:
            image_dir, label_dir, allowed_stems = split_sources[split]
        else:
            image_dir, label_dir = resolve_split(source_yaml, config, split)
            allowed_stems = None
        images = image_index(image_dir)
        if allowed_stems is not None:
            images = {stem: path for stem, path in images.items() if stem in allowed_stems}
        if args.max_samples is not None:
            images = dict(list(sorted(images.items()))[: args.max_samples])
        if not images:
            raise ValueError(f"{split} 이미지가 없습니다: {image_dir}")
        output_images = output_root / "images" / split
        output_labels = output_root / "labels" / split
        for existing_output in (output_images, output_labels):
            if not args.resume and existing_output.exists() and any(existing_output.iterdir()):
                raise FileExistsError(f"기존 전처리 결과를 덮어쓰지 않습니다: {existing_output}")
        class_counts: Counter[int] = Counter()
        empty_count = 0
        resumed_count = 0
        rejected_boxes = []
        for number, (stem, image_path) in enumerate(sorted(images.items()), 1):
            label_path = label_dir / f"{stem}.txt"
            output_image = output_images / image_path.name
            output_label = output_labels / f"{stem}.txt"
            if args.resume and output_image.is_file() and output_label.is_file():
                existing_image = cv2.imread(str(output_image))
                if existing_image is None or existing_image.shape[:2] != (args.image_size, args.image_size):
                    raise ValueError(f"재개 대상 이미지 크기 오류: {output_image}")
                existing_boxes, existing_rejected = read_source_labels(output_label, {0: 0, 1: 1})
                if existing_rejected:
                    raise ValueError(f"기존 전처리 라벨에 퇴화 bbox가 있습니다: {output_label}")
                source_boxes, source_rejected = read_source_labels(label_path, remap)
                if Counter(class_id for class_id, _ in existing_boxes) != Counter(
                    class_id for class_id, _ in source_boxes
                ):
                    raise ValueError(f"원본과 기존 전처리 라벨의 class 개수가 다릅니다: {output_label}")
                for item in source_rejected:
                    rejected_boxes.append(item)
                    logger.warning(
                        "[WARNING] 퇴화 bbox 제외(재개 검증): "
                        f"{item['path']}:{item['line']} class={item['source_class_id']} "
                        f"values={item['values']} reason={item['reason']}"
                    )
                class_counts.update(class_id for class_id, _ in existing_boxes)
                empty_count += not existing_boxes
                resumed_count += 1
                if number == 1 or number % args.log_interval == 0:
                    logger.info(f"[LOG] {split}: {number:,}/{len(images):,} (resume skip={resumed_count:,})")
                continue
            source_boxes, transformed, rejected = process_image(
                image_path, label_path, output_image,
                output_label, remap, args.image_size,
            )
            for item in rejected:
                rejected_boxes.append(item)
                logger.warning(
                    "[WARNING] 퇴화 bbox 제외: "
                    f"{item['path']}:{item['line']} class={item['source_class_id']} "
                    f"values={item['values']} reason={item['reason']}"
                )
            class_counts.update(class_id for class_id, _ in transformed)
            empty_count += not transformed
            if number <= args.visualize_count:
                raw_image = cv2.imread(str(image_path))
                processed_image = cv2.imread(str(output_images / image_path.name))
                raw_vis = output_root / "visualization" / "raw_bbox" / split / image_path.name
                processed_vis = output_root / "visualization" / "processed_bbox" / split / image_path.name
                raw_vis.parent.mkdir(parents=True, exist_ok=True)
                processed_vis.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(raw_vis), draw(raw_image, source_boxes))
                cv2.imwrite(str(processed_vis), draw(processed_image, transformed))
            if number == 1 or number % args.log_interval == 0:
                logger.info(f"[LOG] {split}: {number:,}/{len(images):,}")
        stats["splits"][split] = {
            "images": len(images), "labels": len(images), "empty_labels": empty_count,
            "resumed_images": resumed_count,
            "rejected_degenerate_boxes": len(rejected_boxes),
            "rejected_boxes": rejected_boxes,
            "boxes": {TARGET_NAMES[key]: class_counts[key] for key in TARGET_NAMES},
        }

    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_config = {
        "path": str(output_root), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": TARGET_NAMES,
    }
    args.config_output.write_text(yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8")
    stats_path = output_root / f"preprocessing_stats_{timestamp}.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[RESULT] dataset_yaml: {args.config_output.resolve()}")
    logger.info(f"[RESULT] stats: {stats_path}")
    logger.info(f"[RESULT] visualization: {output_root / 'visualization'}")
    if args.max_samples is not None:
        logger.warning("[WARNING] max-samples로 만든 데이터는 파이프라인 점검용이며 최종 학습에 사용하지 마세요.")


if __name__ == "__main__":
    main()
