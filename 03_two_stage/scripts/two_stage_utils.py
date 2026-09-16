"""03_two_stage 파이프라인에서 공통으로 쓰는 좌표와 파일 유틸리티."""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def make_stage_logger(project_dir: Path, stage: str, timestamp: str | None = None):
    """각 단계를 터미널과 03_two_stage/logs에 동시에 기록한다."""
    timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    log_dir = project_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}_{timestamp}.log"
    logger = logging.getLogger(f"two_stage_{stage}_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def log_uncaught_exception(exception_type, exception, traceback) -> None:
        logger.error(
            "[ERROR] uncaught exception",
            exc_info=(exception_type, exception, traceback),
        )

    sys.excepthook = log_uncaught_exception
    logger.info(f"[INFO] log={log_path}")
    return logger, log_path


def image_index(root: Path) -> dict[str, Path]:
    """하위 이미지를 stem으로 인덱싱하고 충돌 시 조용히 덮어쓰지 않는다."""
    by_stem: dict[str, list[Path]] = defaultdict(list)
    if not root.is_dir():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {root}")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            by_stem[path.stem].append(path)
    duplicates = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if duplicates:
        stem, paths = next(iter(duplicates.items()))
        raise ValueError(f"동일 stem 이미지가 여러 개입니다: {stem}: {paths}")
    return {stem: paths[0] for stem, paths in by_stem.items()}


def yolo_to_xyxy(cx: float, cy: float, width: float, height: float) -> tuple[float, ...]:
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def xyxy_to_yolo(box: tuple[float, ...]) -> tuple[float, ...]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def read_yolo_labels(path: Path, with_confidence: bool = False) -> list[dict]:
    expected = 6 if with_confidence else 5
    boxes = []
    if not path.is_file():
        raise FileNotFoundError(f"라벨 파일이 없습니다: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != expected:
            raise ValueError(f"YOLO 라벨 열 개수 오류: {path}:{line_number} ({len(parts)}개)")
        class_id = int(parts[0])
        cx, cy, width, height = map(float, parts[1:5])
        values = (cx, cy, width, height)
        if not all(0.0 <= value <= 1.0 for value in values) or width <= 0 or height <= 0:
            raise ValueError(f"정규화 bbox 범위 오류: {path}:{line_number}: {values}")
        item = {"class_id": class_id, "box": yolo_to_xyxy(*values)}
        if with_confidence:
            item["confidence"] = float(parts[5])
        boxes.append(item)
    return boxes
