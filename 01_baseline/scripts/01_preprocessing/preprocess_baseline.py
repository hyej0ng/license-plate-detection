import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# 01. 기본 경로 및 설정

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = PROJECT_ROOT / "data" / "preprocessed"
RAW_BBOX_VIS_ROOT = PROJECT_ROOT / "data" / "visualization" / "raw_bbox"
PROCESSED_BBOX_VIS_ROOT = PROJECT_ROOT / "data" / "visualization" / "processed_bbox"

TARGET_WIDTH = 640
TARGET_HEIGHT = 640

CLASS_ID = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SPLIT_PATHS: dict[str, dict[str, Path]] = {}


def configure_split_paths(data_root: Path) -> None:
    """로컬 원본 데이터 루트를 기준으로 split별 경로를 설정한다."""
    SPLIT_PATHS.clear()
    SPLIT_PATHS.update(
        {
            "train": {
                "image_root": data_root / "data" / "Train" / "01.원천데이터",
                "label_root": data_root / "data" / "Train" / "02.라벨링데이터",
            },
            "val": {
                "image_root": data_root / "data" / "Validation" / "01.원천데이터",
                "label_root": data_root / "data" / "Validation" / "02.라벨링데이터",
            },
            "test": {
                "image_root": data_root / "Test" / "01.원천데이터",
                "label_root": data_root / "Test" / "02.라벨링데이터",
            },
        }
    )


# 02. JSON 파일 읽기

def load_json(json_path: Path):
    try:
        with json_path.open(mode="r", encoding="utf-8") as file:
            return json.load(file) 
    except UnicodeDecodeError:
        with json_path.open(mode="r", encoding="cp949") as file:
            return json.load(file)


# 03. 실제 JSON 구조에서 번호판 bbox 추출

def extract_boxes_from_json(data): # data의 타입: dict[str, Any]

    boxes = []  # boxes 타입: list[tuple[float, float, float, float]]
    seen_boxes = set()

    learning_info = data.get("Learning_Data_Info", {})
    annotations = learning_info.get("annotations", [])

    for annotation_group in annotations:
        license_plates = annotation_group.get("license_plate", [])

        for plate in license_plates:
            bbox = plate.get("bbox")

            if bbox is None or not isinstance(bbox, list) or len(bbox) != 4: 
                continue   

            x, y, width, height = (float(v) for v in bbox) 

            if width <= 0 or height <= 0:
                continue

            box = (x, y, x + width, y + height)

            # 원본 JSON에 완전히 같은 bbox가 반복된 경우 한 번만 사용한다.
            # 서로 다른 bbox가 일부 겹치는 경우는 제거하지 않는다.
            if box in seen_boxes:
                continue

            seen_boxes.add(box)
            boxes.append(box)

    return boxes


# 04. 파일 stem 기준 인덱스 생성

def build_image_index(image_root: Path) -> dict[str, Path]:
    """image_root 아래의 모든 이미지를 stem 기준으로 인덱싱한다."""
    paths_by_stem: dict[str, list[Path]] = defaultdict(list)

    if not image_root.exists():
        raise FileNotFoundError(f"이미지 루트가 없습니다:\n{image_root}")

    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths_by_stem[path.stem].append(path)

    duplicate_stems = {
        stem: paths for stem, paths in paths_by_stem.items() if len(paths) > 1
    }

    if duplicate_stems:
        example_stem, example_paths = next(iter(duplicate_stems.items()))
        raise ValueError(
            "동일한 stem을 가진 이미지가 여러 개 있습니다.\n"
            f"예시 stem: {example_stem}\n"
            f"경로: {example_paths}"
        )

    return {stem: paths[0] for stem, paths in paths_by_stem.items()}


def build_json_index(label_root: Path) -> dict[str, Path]:
    """label_root 아래의 모든 JSON을 stem 기준으로 인덱싱한다."""
    if not label_root.exists():
        raise FileNotFoundError(f"라벨 루트가 없습니다:\n{label_root}")

    paths_by_stem: dict[str, list[Path]] = defaultdict(list)

    for path in label_root.rglob("*.json"):
        if path.is_file():
            paths_by_stem[path.stem].append(path)

    duplicate_stems = {
        stem: paths for stem, paths in paths_by_stem.items() if len(paths) > 1
    }

    if duplicate_stems:
        example_stem, example_paths = next(iter(duplicate_stems.items()))
        raise ValueError(
            "동일한 stem을 가진 JSON이 여러 개 있습니다.\n"
            f"예시 stem: {example_stem}\n"
            f"경로: {example_paths}"
        )

    return {stem: paths[0] for stem, paths in paths_by_stem.items()}


# 05. 이미지와 JSON 쌍 찾기

def get_matching_pairs(split_name: str) -> list[tuple[str, Path, Path]]:
    """
    split의 이미지와 JSON을 stem 기준으로 연결한다.

    반환: [(stem, image_path, json_path), ...]
    """
    if split_name not in SPLIT_PATHS:
        raise ValueError(f"잘못된 split입니다: {split_name}")

    image_root = SPLIT_PATHS[split_name]["image_root"]
    label_root = SPLIT_PATHS[split_name]["label_root"]

    image_index = build_image_index(image_root)
    json_index = build_json_index(label_root)

    image_stems = set(image_index)
    json_stems = set(json_index)

    matching_stems = sorted(image_stems & json_stems)
    image_only_stems = image_stems - json_stems
    json_only_stems = json_stems - image_stems

    print("=" * 70)
    print(f"Split: {split_name}")
    print(f"이미지 수        : {len(image_stems):,}")
    print(f"JSON 수          : {len(json_stems):,}")
    print(f"정상 쌍 수       : {len(matching_stems):,}")
    print(f"JSON 없는 이미지 : {len(image_only_stems):,}")
    print(f"이미지 없는 JSON : {len(json_only_stems):,}")
    print("=" * 70)

    return [(stem, image_index[stem], json_index[stem]) for stem in matching_stems]


def find_one_pair(
    split_name: str,
    requested_stem: str | None = None,
) -> tuple[str, Path, Path]:
    """
    테스트용 이미지와 JSON 한 쌍을 찾는다.

    requested_stem이 있으면 해당 파일을 찾고, 없으면 첫 번째 정상 쌍을 반환한다.
    """
    pairs = get_matching_pairs(split_name)

    if not pairs:
        raise FileNotFoundError(f"{split_name}에서 이미지-JSON 쌍을 찾지 못했습니다.")

    if requested_stem is None:
        return pairs[0]

    for stem, image_path, json_path in pairs:
        if stem == requested_stem:
            return stem, image_path, json_path

    raise FileNotFoundError(f"요청한 stem을 찾지 못했습니다:\n{requested_stem}")


# 06. 전처리 전 원본 bbox 시각화

def draw_boxes_on_original_image(
    image_path: Path,
    json_path: Path,
    output_path: Path,
) -> int:
    """
    resize 전 원본 이미지에 JSON bbox를 그린다.

    이 단계에서는 resize/좌표 변환/정규화를 하지 않는다.
    목적: JSON bbox 해석이 올바른지 눈으로 확인.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다:\n{image_path}")

    image_height, image_width = image.shape[:2]

    data = load_json(json_path)
    boxes = extract_boxes_from_json(data)

    valid_box_count = 0

    for box_index, box in enumerate(boxes, start=1):
        xmin, ymin, xmax, ymax = box

        # 이미지 경계를 넘어가지 않도록 제한
        xmin = max(0.0, min(xmin, float(image_width - 1)))
        ymin = max(0.0, min(ymin, float(image_height - 1)))
        xmax = max(0.0, min(xmax, float(image_width - 1)))
        ymax = max(0.0, min(ymax, float(image_height - 1)))

        if xmax <= xmin or ymax <= ymin:
            print(f"[경고] 잘못된 원본 bbox: {box}")
            continue

        x1, y1, x2, y2 = (int(round(v)) for v in (xmin, ymin, xmax, ymax))

        # 번호판 전체 bbox를 초록색으로 표시
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)

        label_text = f"plate {box_index}"
        text_y = max(y1 - 8, 20)

        cv2.putText(
            image,
            label_text,
            (x1, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.65,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        valid_box_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = cv2.imwrite(str(output_path), image)

    if not saved:
        raise IOError(f"이미지 저장 실패:\n{output_path}")

    print(f"원본 bbox 시각화 저장: {output_path}")
    print(f"그린 bbox 개수: {valid_box_count}")

    return valid_box_count


# 07. Letterbox resize

def letterbox_resize(
    image: np.ndarray,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> tuple[np.ndarray, float, int, int]:
    """
    이미지의 가로세로 비율을 유지하면서 resize하고,
    남는 부분에 회색 padding을 추가한다.

    반환: (letterboxed_image, scale, pad_left, pad_top)
    """
    original_height, original_width = image.shape[:2]

    if original_width <= 0:
        raise ValueError("이미지 너비가 0 이하입니다.")

    if original_height <= 0:
        raise ValueError("이미지 높이가 0 이하입니다.")

    scale = min(target_width / original_width, target_height / original_height)

    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    padding_width = target_width - resized_width
    padding_height = target_height - resized_height

    pad_left = padding_width // 2
    pad_right = padding_width - pad_left

    pad_top = padding_height // 2
    pad_bottom = padding_height - pad_top

    letterboxed_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return letterboxed_image, scale, pad_left, pad_top


# 08. Letterbox에 맞춰 bbox 좌표 변환

def transform_bbox_for_letterbox(
    bbox: tuple[float, float, float, float],
    scale: float,
    pad_left: int,
    pad_top: int,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> tuple[float, float, float, float] | None:
    """원본 bbox를 640x640 letterbox 이미지 좌표로 변환한다."""
    xmin, ymin, xmax, ymax = bbox

    new_xmin = xmin * scale + pad_left
    new_ymin = ymin * scale + pad_top
    new_xmax = xmax * scale + pad_left
    new_ymax = ymax * scale + pad_top

    new_xmin = max(0.0, min(new_xmin, float(target_width)))
    new_ymin = max(0.0, min(new_ymin, float(target_height)))
    new_xmax = max(0.0, min(new_xmax, float(target_width)))
    new_ymax = max(0.0, min(new_ymax, float(target_height)))

    if new_xmax <= new_xmin or new_ymax <= new_ymin:
        return None

    return new_xmin, new_ymin, new_xmax, new_ymax


# 09. bbox를 YOLO 형식으로 정규화

def bbox_to_yolo_format(
    bbox: tuple[float, float, float, float],
    image_width: int = TARGET_WIDTH,
    image_height: int = TARGET_HEIGHT,
) -> tuple[float, float, float, float]:
    """
    xmin, ymin, xmax, ymax를 center_x, center_y, width, height로 바꾸고,
    이미지 크기로 나눠 0~1 범위로 정규화한다.
    """
    xmin, ymin, xmax, ymax = bbox

    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    return (
        center_x / image_width,
        center_y / image_height,
        bbox_width / image_width,
        bbox_height / image_height,
    )


def is_valid_yolo_box(yolo_box: tuple[float, float, float, float]) -> bool:
    """YOLO 좌표가 정상적인 0~1 범위인지 확인한다."""
    center_x, center_y, width, height = yolo_box

    if not 0.0 <= center_x <= 1.0:
        return False
    if not 0.0 <= center_y <= 1.0:
        return False
    if not 0.0 < width <= 1.0:
        return False
    if not 0.0 < height <= 1.0:
        return False

    return True


# 10. YOLO 정규화 좌표를 픽셀 bbox로 복원

def yolo_to_pixel_bbox(
    yolo_box: tuple[float, float, float, float],
    image_width: int = TARGET_WIDTH,
    image_height: int = TARGET_HEIGHT,
) -> tuple[int, int, int, int]:
    """시각화 검증을 위해 YOLO 정규화 좌표를 다시 픽셀 좌표로 변환한다."""
    center_x, center_y, width, height = yolo_box

    center_x_pixel = center_x * image_width
    center_y_pixel = center_y * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height

    xmin = center_x_pixel - width_pixel / 2.0
    ymin = center_y_pixel - height_pixel / 2.0
    xmax = center_x_pixel + width_pixel / 2.0
    ymax = center_y_pixel + height_pixel / 2.0

    return tuple(int(round(v)) for v in (xmin, ymin, xmax, ymax))


# 11. 이미지 한 장 전체 전처리

def preprocess_one_sample(
    image_path: Path,
    json_path: Path,
    output_image_path: Path,
    output_label_path: Path,
) -> list[tuple[float, float, float, float]]:
    """
    이미지 한 장과 JSON 하나를 전처리한다.

    수행 작업:
        1. 이미지 읽기
        2. JSON에서 bbox 추출
        3. 640x640 letterbox resize
        4. bbox 좌표 변환
        5. YOLO 형식 정규화
        6. 이미지 저장
        7. TXT 라벨 저장

    반환: 저장한 YOLO bbox 목록
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다:\n{image_path}")

    original_height, original_width = image.shape[:2]

    data = load_json(json_path)
    original_boxes = extract_boxes_from_json(data)

    resized_image, scale, pad_left, pad_top = letterbox_resize(
        image=image,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )

    yolo_boxes: list[tuple[float, float, float, float]] = []
    yolo_lines: list[str] = []

    for original_box in original_boxes:
        xmin, ymin, xmax, ymax = original_box

        # 원본 bbox가 이미지 바깥으로 나간 경우 제한
        xmin = max(0.0, min(xmin, float(original_width)))
        ymin = max(0.0, min(ymin, float(original_height)))
        xmax = max(0.0, min(xmax, float(original_width)))
        ymax = max(0.0, min(ymax, float(original_height)))

        clipped_box = (xmin, ymin, xmax, ymax)

        transformed_box = transform_bbox_for_letterbox(
            bbox=clipped_box,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )

        if transformed_box is None:
            continue

        yolo_box = bbox_to_yolo_format(bbox=transformed_box)

        if not is_valid_yolo_box(yolo_box):
            continue

        yolo_boxes.append(yolo_box)

        center_x, center_y, bbox_width, bbox_height = yolo_box

        yolo_lines.append(
            f"{CLASS_ID} {center_x:.6f} {center_y:.6f} "
            f"{bbox_width:.6f} {bbox_height:.6f}"
        )

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)

    image_saved = cv2.imwrite(str(output_image_path), resized_image)

    if not image_saved:
        raise IOError(f"전처리 이미지 저장 실패:\n{output_image_path}")

    output_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    return yolo_boxes


# 12. 전처리된 이미지와 YOLO 라벨 시각화

def draw_processed_boxes(
    processed_image_path: Path,
    yolo_boxes: list[tuple[float, float, float, float]],
    output_path: Path,
) -> None:
    """
    640x640 전처리 이미지 위에 YOLO bbox를 다시 그린다.

    이 결과를 원본 bbox 시각화와 비교하면 resize와 좌표 변환이 정확한지 확인할 수 있다.
    """
    image = cv2.imread(str(processed_image_path))

    if image is None:
        raise ValueError(f"전처리 이미지를 읽지 못했습니다:\n{processed_image_path}")

    for box_index, yolo_box in enumerate(yolo_boxes, start=1):
        x1, y1, x2, y2 = yolo_to_pixel_bbox(yolo_box)

        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        cv2.putText(
            image,
            f"plate {box_index}",
            (x1, max(y1 - 5, 20)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = cv2.imwrite(str(output_path), image)

    if not saved:
        raise IOError(f"시각화 이미지 저장 실패:\n{output_path}")

    print(f"전처리 후 bbox 시각화 저장: {output_path}")


# 13. 원본 bbox 한 장 테스트

def visualize_raw_one(split_name: str, requested_stem: str | None) -> None:
    """이미지와 JSON 한 쌍을 찾아 전처리 전 원본 bbox를 시각화한다."""
    stem, image_path, json_path = find_one_pair(
        split_name=split_name,
        requested_stem=requested_stem,
    )

    output_path = RAW_BBOX_VIS_ROOT / split_name / f"{stem}_raw_bbox.jpg"

    print()
    print("[원본 bbox 시각화]")
    print(f"이미지: {image_path}")
    print(f"JSON  : {json_path}")
    print(f"출력  : {output_path}")

    draw_boxes_on_original_image(
        image_path=image_path,
        json_path=json_path,
        output_path=output_path,
    )


# 14. 이미지 한 장 전처리 테스트

def preprocess_one_test(split_name: str, requested_stem: str | None) -> None:
    """
    이미지 한 장을 전처리하고, 전처리 이미지와 TXT 라벨을 저장한다.
    동시에 전처리 후 bbox 시각화 이미지도 저장한다.
    """
    stem, image_path, json_path = find_one_pair(
        split_name=split_name,
        requested_stem=requested_stem,
    )

    output_image_path = OUTPUT_ROOT / "test_one" / "images" / f"{stem}.jpg"
    output_label_path = OUTPUT_ROOT / "test_one" / "labels" / f"{stem}.txt"
    processed_vis_path = (
        PROCESSED_BBOX_VIS_ROOT / split_name / f"{stem}_processed_bbox.jpg"
    )

    yolo_boxes = preprocess_one_sample(
        image_path=image_path,
        json_path=json_path,
        output_image_path=output_image_path,
        output_label_path=output_label_path,
    )

    draw_processed_boxes(
        processed_image_path=output_image_path,
        yolo_boxes=yolo_boxes,
        output_path=processed_vis_path,
    )

    print()
    print("[한 장 전처리 완료]")
    print(f"원본 이미지   : {image_path}")
    print(f"원본 JSON     : {json_path}")
    print(f"전처리 이미지 : {output_image_path}")
    print(f"YOLO 라벨     : {output_label_path}")
    print(f"번호판 수     : {len(yolo_boxes)}")

    if output_label_path.exists():
        print()
        print("[생성된 TXT 내용]")
        print(output_label_path.read_text(encoding="utf-8"))


# 15. Split 하나 전체 전처리

def process_split(split_name: str) -> None:
    """train, val 또는 test 전체를 전처리한다."""
    pairs = get_matching_pairs(split_name)

    output_image_dir = OUTPUT_ROOT / "images" / split_name
    output_label_dir = OUTPUT_ROOT / "labels" / split_name

    success_count = 0
    error_count = 0
    total_box_count = 0
    empty_label_count = 0

    for index, (stem, image_path, json_path) in enumerate(pairs, start=1):
        output_image_path = output_image_dir / f"{stem}.jpg"
        output_label_path = output_label_dir / f"{stem}.txt"

        try:
            yolo_boxes = preprocess_one_sample(
                image_path=image_path,
                json_path=json_path,
                output_image_path=output_image_path,
                output_label_path=output_label_path,
            )
        except Exception as error:
            error_count += 1

            print()
            print(f"[전처리 오류] {stem}")
            print(f"이미지: {image_path}")
            print(f"JSON  : {json_path}")
            print(f"오류  : {error}")

            continue

        success_count += 1
        total_box_count += len(yolo_boxes)

        if not yolo_boxes:
            empty_label_count += 1

        if index % 100 == 0:
            print(f"{split_name} 진행: {index:,}/{len(pairs):,}")

    print()
    print("=" * 70)
    print(f"{split_name} 전처리 완료")
    print(f"성공 이미지 수 : {success_count:,}")
    print(f"오류 이미지 수 : {error_count:,}")
    print(f"전체 번호판 수 : {total_box_count:,}")
    print(f"빈 라벨 파일 수: {empty_label_count:,}")
    print("=" * 70)


# 16. 전체 Train / Validation / Test 전처리

def preprocess_all() -> None:
    """train, val, test를 순서대로 전처리한다."""
    for split_name in ("train", "val", "test"):
        process_split(split_name)


# 17. 명령행 인자

def parse_args() -> argparse.Namespace:
    """실행할 작업과 split, 파일 stem을 입력받는다."""
    parser = argparse.ArgumentParser(
        description="차량 번호판 YOLO baseline 전처리"
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("LICENSE_PLATE_DATA_ROOT"),
        help=(
            "원본 데이터셋 루트. 생략하면 LICENSE_PLATE_DATA_ROOT 환경변수를 사용"
        ),
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["visualize_raw", "preprocess_one", "preprocess_split", "preprocess_all"],
        help=(
            "visualize_raw: 전처리 전 bbox 확인, "
            "preprocess_one: 한 장 전처리, "
            "preprocess_split: 선택한 split만 전체 전처리, "
            "preprocess_all: 전체 전처리"
        ),
    )

    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="한 장 테스트에 사용할 split",
    )

    parser.add_argument(
        "--stem",
        default=None,
        help="확장자를 제외한 파일 이름. 생략하면 첫 번째 정상 쌍을 사용",
    )

    args = parser.parse_args()

    if args.data_root is None:
        parser.error(
            "--data-root를 지정하거나 LICENSE_PLATE_DATA_ROOT 환경변수를 설정하세요."
        )

    args.data_root = args.data_root.expanduser().resolve()
    return args


# 18. 프로그램 시작 지점

def main() -> None:
    args = parse_args()
    configure_split_paths(args.data_root)

    if args.mode == "visualize_raw":
        visualize_raw_one(split_name=args.split, requested_stem=args.stem,)

    elif args.mode == "preprocess_one":
        preprocess_one_test(split_name=args.split, requested_stem=args.stem,)

    elif args.mode == "preprocess_split":
        process_split(args.split)

    elif args.mode == "preprocess_all":
        preprocess_all()

    else:
        raise ValueError(f"지원하지 않는 mode입니다: {args.mode}")


if __name__ == "__main__":
    main()
