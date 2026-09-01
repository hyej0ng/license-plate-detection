"""4분할 전처리와 추론이 함께 사용하는 좌표/파일 유틸리티."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_ID = 0


@dataclass(frozen=True)
class CropRegion:
    """원본 이미지 좌표계의 한 crop 영역. 끝 좌표는 슬라이싱처럼 미포함이다."""

    index: int
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def box(self) -> tuple[float, float, float, float]:
        return float(self.x1), float(self.y1), float(self.x2), float(self.y2)


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_left: int
    pad_top: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int


def validate_ratio(name: str, value: float, *, upper_exclusive: float | None = None) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name}는 0~1 사이여야 합니다: {value}")
    if upper_exclusive is not None and value >= upper_exclusive:
        raise ValueError(f"{name}는 {upper_exclusive}보다 작아야 합니다: {value}")


def make_margin_quarters(
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> list[CropRegion]:
    """중앙 경계 양쪽을 원본 크기×margin_ratio만큼 확장한 4개 crop을 만든다.

    예: margin_ratio=0.05이면 세로 중앙선 좌우에 각각 이미지 폭의 5%가
    포함되어, 좌/우 crop은 전체 폭의 10%만큼 서로 겹친다.
    """
    if image_width <= 1 or image_height <= 1:
        raise ValueError("이미지 가로와 세로는 2픽셀 이상이어야 합니다.")
    validate_ratio("margin_ratio", margin_ratio, upper_exclusive=0.5)

    mid_x = image_width // 2
    mid_y = image_height // 2
    margin_x = int(round(image_width * margin_ratio))
    margin_y = int(round(image_height * margin_ratio))
    left_end = min(image_width, mid_x + margin_x)
    right_start = max(0, mid_x - margin_x)
    top_end = min(image_height, mid_y + margin_y)
    bottom_start = max(0, mid_y - margin_y)

    return [
        CropRegion(0, "top_left", 0, 0, left_end, top_end),
        CropRegion(1, "top_right", right_start, 0, image_width, top_end),
        CropRegion(2, "bottom_left", 0, bottom_start, left_end, image_height),
        CropRegion(3, "bottom_right", right_start, bottom_start, image_width, image_height),
    ]


def crop_image(image: np.ndarray, region: CropRegion) -> np.ndarray:
    crop = image[region.y1 : region.y2, region.x1 : region.x2]
    if crop.size == 0:
        raise ValueError(f"빈 crop이 생성되었습니다: {region}")
    return crop


def clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    clipped = (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersect_box(
    box: tuple[float, float, float, float],
    region: CropRegion,
) -> tuple[tuple[float, float, float, float] | None, float]:
    """원본 bbox와 crop의 교집합을 crop 로컬 좌표 및 원본 면적 대비 비율로 반환."""
    ix1 = max(box[0], float(region.x1))
    iy1 = max(box[1], float(region.y1))
    ix2 = min(box[2], float(region.x2))
    iy2 = min(box[3], float(region.y2))
    intersection = (ix1, iy1, ix2, iy2)
    original_area = box_area(box)
    intersection_area = box_area(intersection)
    if original_area <= 0.0 or intersection_area <= 0.0:
        return None, 0.0
    local = (
        ix1 - region.x1,
        iy1 - region.y1,
        ix2 - region.x1,
        iy2 - region.y1,
    )
    return local, intersection_area / original_area


def letterbox(
    image: np.ndarray,
    image_size: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxMeta]:
    if image_size <= 0:
        raise ValueError("image_size는 1 이상이어야 합니다.")
    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("이미지 크기가 올바르지 않습니다.")

    scale = min(image_size / source_width, image_size / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_width = image_size - resized_width
    pad_height = image_size - resized_height
    pad_left = pad_width // 2
    pad_top = pad_height // 2
    output = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_height - pad_top,
        pad_left,
        pad_width - pad_left,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    meta = LetterboxMeta(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        source_width=source_width,
        source_height=source_height,
        target_width=image_size,
        target_height=image_size,
    )
    return output, meta


def box_to_letterbox(
    box: tuple[float, float, float, float],
    meta: LetterboxMeta,
) -> tuple[float, float, float, float] | None:
    transformed = (
        box[0] * meta.scale + meta.pad_left,
        box[1] * meta.scale + meta.pad_top,
        box[2] * meta.scale + meta.pad_left,
        box[3] * meta.scale + meta.pad_top,
    )
    return clip_box(transformed, meta.target_width, meta.target_height)


def box_from_letterbox(
    box: tuple[float, float, float, float],
    meta: LetterboxMeta,
) -> tuple[float, float, float, float] | None:
    restored = (
        (box[0] - meta.pad_left) / meta.scale,
        (box[1] - meta.pad_top) / meta.scale,
        (box[2] - meta.pad_left) / meta.scale,
        (box[3] - meta.pad_top) / meta.scale,
    )
    return clip_box(restored, meta.source_width, meta.source_height)


def local_box_to_original(
    box: tuple[float, float, float, float],
    region: CropRegion,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    original = (
        box[0] + region.x1,
        box[1] + region.y1,
        box[2] + region.x1,
        box[3] + region.y1,
    )
    return clip_box(original, image_width, image_height)


def box_to_yolo(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2.0 / image_width,
        (y1 + y2) / 2.0 / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def yolo_to_box(
    yolo_box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    cx, cy, width, height = yolo_box
    cx *= image_width
    cy *= image_height
    width *= image_width
    height *= image_height
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def load_json(json_path: Path) -> dict:
    try:
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except UnicodeDecodeError:
        with json_path.open("r", encoding="cp949") as file:
            return json.load(file)


def extract_boxes_from_json(data: dict) -> list[tuple[float, float, float, float]]:
    """baseline과 동일하게 Learning_Data_Info.annotations의 번호판 bbox를 읽는다."""
    boxes: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    annotations = data.get("Learning_Data_Info", {}).get("annotations", [])
    for group in annotations:
        for plate in group.get("license_plate", []):
            bbox = plate.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, width, height = (float(value) for value in bbox)
            if width <= 0.0 or height <= 0.0:
                continue
            box = (x, y, x + width, y + height)
            if box not in seen:
                seen.add(box)
                boxes.append(box)
    return boxes


def dataset_split_roots(data_root: Path, split: str) -> tuple[Path, Path]:
    paths = {
        "train": (
            data_root / "data" / "Train" / "01.원천데이터",
            data_root / "data" / "Train" / "02.라벨링데이터",
        ),
        "val": (
            data_root / "data" / "Validation" / "01.원천데이터",
            data_root / "data" / "Validation" / "02.라벨링데이터",
        ),
        "test": (
            data_root / "Test" / "01.원천데이터",
            data_root / "Test" / "02.라벨링데이터",
        ),
    }
    if split not in paths:
        raise ValueError(f"지원하지 않는 split입니다: {split}")
    return paths[split]


def build_stem_index(root: Path, extensions: Iterable[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {root}")
    extensions = {extension.lower() for extension in extensions}
    paths_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            paths_by_stem[path.stem].append(path)
    duplicates = {stem: paths for stem, paths in paths_by_stem.items() if len(paths) > 1}
    if duplicates:
        stem, paths = next(iter(duplicates.items()))
        raise ValueError(f"동일 stem 파일이 여러 개입니다: {stem} -> {paths}")
    return {stem: paths[0] for stem, paths in paths_by_stem.items()}


def matching_pairs(data_root: Path, split: str) -> list[tuple[str, Path, Path]]:
    image_root, label_root = dataset_split_roots(data_root, split)
    images = build_stem_index(image_root, IMAGE_EXTENSIONS)
    labels = build_stem_index(label_root, {".json"})
    stems = sorted(set(images) & set(labels))
    if not stems:
        raise FileNotFoundError(f"{split} 이미지-JSON 쌍이 없습니다.")
    return [(stem, images[stem], labels[stem]) for stem in stems]


def calculate_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    intersection = (
        max(box_a[0], box_b[0]),
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3]),
    )
    intersection_area = box_area(intersection)
    union = box_area(box_a) + box_area(box_b) - intersection_area
    return intersection_area / union if union > 0.0 else 0.0


def global_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """원본 좌표계에서 class별 confidence greedy NMS를 수행한다."""
    validate_ratio("nms_iou_threshold", iou_threshold)
    kept: list[dict] = []
    pending = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    while pending:
        best = pending.pop(0)
        kept.append(best)
        survivors = []
        for candidate in pending:
            same_class = candidate["class_id"] == best["class_id"]
            overlap = calculate_iou(candidate["box"], best["box"])
            if not same_class or overlap < iou_threshold:
                survivors.append(candidate)
        pending = survivors
    return kept
