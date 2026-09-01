"""Global NMS 예측을 JSON 또는 YOLO 정답과 원본 이미지 단위로 평가한다."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
QUARTER_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(QUARTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(QUARTER_SCRIPTS_DIR))

from quarter_utils import (  # noqa: E402
    IMAGE_EXTENSIONS,
    build_stem_index,
    calculate_iou,
    clip_box,
    dataset_split_roots,
    extract_boxes_from_json,
    load_json,
    matching_pairs,
    validate_ratio,
    yolo_to_box,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUARTER_ROOT = PROJECT_ROOT / "02_quarter"
DEFAULT_DATA_ROOT = Path(
    os.environ.get("LICENSE_PLATE_DATA_ROOT", "/mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset")
)
METRICS_ROOT = QUARTER_ROOT / "results" / "metrics"
ERRORS_ROOT = QUARTER_ROOT / "results" / "errors"

CONFIDENCE_THRESHOLD = 0.25
MATCH_IOU_THRESHOLD = 0.50


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NMS 이후 예측을 IoU 일대일 매칭으로 평가 (quarter/baseline 공용)"
    )
    parser.add_argument("--predictions", type=Path, required=True, help="detections.csv 또는 이를 포함한 inference 폴더")
    parser.add_argument("--confidence", "--conf", dest="confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--match-iou", type=float, default=MATCH_IOU_THRESHOLD)
    parser.add_argument("--ground-truth-format", choices=("json", "yolo"), default="json")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="JSON 모드의 원본 데이터 루트")
    parser.add_argument("--image-root", type=Path, help="YOLO 모드의 이미지 폴더")
    parser.add_argument("--ground-truth-root", type=Path, help="YOLO 모드의 labels/test 폴더")
    return parser.parse_args()


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"quarter_evaluation_{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def resolve_prediction_files(path: Path) -> tuple[Path, Path | None, Path]:
    path = path.resolve()
    if path.is_dir():
        csv_path = path / "detections.csv"
        summary_path = path / "image_summary.csv"
        return csv_path, summary_path if summary_path.is_file() else None, path
    if path.is_file():
        summary = path.parent / "image_summary.csv"
        return path, summary if summary.is_file() else None, path.parent
    raise FileNotFoundError(f"예측 경로가 없습니다: {path}")


def read_predictions(path: Path, confidence: float) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"예측 CSV가 없습니다: {path}")
    predictions: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_name", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"예측 CSV 필수 열이 없습니다: {sorted(required)}")
        for row in reader:
            score = float(row["confidence"])
            if score < confidence:
                continue
            predictions[row["image_name"]].append(
                {
                    "class_id": int(row["class_id"]),
                    "confidence": score,
                    "box": tuple(float(row[name]) for name in ("xmin", "ymin", "xmax", "ymax")),
                }
            )
    for detections in predictions.values():
        detections.sort(key=lambda item: item["confidence"], reverse=True)
    return predictions


def read_summary_names(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as file:
        return [row["image_name"] for row in csv.DictReader(file)]


def load_json_ground_truth(data_root: Path, selected_names: set[str] | None = None) -> dict[str, dict]:
    ground_truth = {}
    for stem, image_path, json_path in matching_pairs(data_root, "test"):
        if selected_names is not None and image_path.name not in selected_names:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")
        height, width = image.shape[:2]
        boxes = []
        for box in extract_boxes_from_json(load_json(json_path)):
            clipped = clip_box(box, width, height)
            if clipped is not None:
                boxes.append({"class_id": 0, "box": clipped})
        ground_truth[image_path.name] = {"image_path": image_path, "boxes": boxes}
    return ground_truth


def load_yolo_ground_truth(
    image_root: Path,
    label_root: Path,
    selected_names: set[str] | None = None,
) -> dict[str, dict]:
    images = build_stem_index(image_root.resolve(), IMAGE_EXTENSIONS)
    labels = build_stem_index(label_root.resolve(), {".txt"})
    ground_truth = {}
    for stem in sorted(set(images) & set(labels)):
        image_path = images[stem]
        if selected_names is not None and image_path.name not in selected_names:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")
        height, width = image.shape[:2]
        boxes = []
        for line_number, line in enumerate(labels[stem].read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"YOLO 정답 형식 오류: {labels[stem]}:{line_number}")
            boxes.append(
                {
                    "class_id": int(parts[0]),
                    "box": yolo_to_box(tuple(float(value) for value in parts[1:]), width, height),
                }
            )
        ground_truth[image_path.name] = {"image_path": image_path, "boxes": boxes}
    return ground_truth


def match_one_image(predictions: list[dict], ground_truths: list[dict], threshold: float):
    """confidence가 높은 예측부터 미사용 정답 중 IoU가 가장 큰 하나와 매칭."""
    used_ground_truths: set[int] = set()
    matches = []
    for prediction_index, prediction in enumerate(predictions):
        best_index = None
        best_iou = 0.0
        for ground_truth_index, ground_truth in enumerate(ground_truths):
            if ground_truth_index in used_ground_truths:
                continue
            if prediction["class_id"] != ground_truth["class_id"]:
                continue
            iou = calculate_iou(prediction["box"], ground_truth["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = ground_truth_index
        is_tp = best_index is not None and best_iou >= threshold
        if is_tp:
            used_ground_truths.add(best_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": best_index if is_tp else None,
                "is_true_positive": is_tp,
                "iou": best_iou if is_tp else 0.0,
            }
        )
    false_negatives = [index for index in range(len(ground_truths)) if index not in used_ground_truths]
    return matches, false_negatives


def draw_box(image, box, color, text) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def save_errors(image_path: Path, ground_truths, predictions, matches, false_negatives, fp_dir: Path, fn_dir: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"오류 시각화 이미지를 읽지 못했습니다: {image_path}")
    for index, ground_truth in enumerate(ground_truths):
        draw_box(image, ground_truth["box"], (0, 165, 255) if index in false_negatives else (0, 255, 0), "FN" if index in false_negatives else "GT")
    has_fp = False
    for match in matches:
        prediction = predictions[match["prediction_index"]]
        if match["is_true_positive"]:
            draw_box(image, prediction["box"], (255, 0, 0), f"TP {prediction['confidence']:.2f} IoU {match['iou']:.2f}")
        else:
            has_fp = True
            draw_box(image, prediction["box"], (0, 0, 255), f"FP {prediction['confidence']:.2f}")
    if has_fp and not cv2.imwrite(str(fp_dir / image_path.name), image):
        raise IOError(f"FP 이미지 저장 실패: {image_path.name}")
    if false_negatives and not cv2.imwrite(str(fn_dir / image_path.name), image):
        raise IOError(f"FN 이미지 저장 실패: {image_path.name}")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    arguments = parse_arguments()
    validate_ratio("confidence_threshold", arguments.confidence)
    validate_ratio("matching_iou_threshold", arguments.match_iou)
    prediction_csv, summary_csv, prediction_run = resolve_prediction_files(arguments.predictions)
    predictions_by_image = read_predictions(prediction_csv, arguments.confidence)
    summary_names = read_summary_names(summary_csv)

    selected_names = set(summary_names) if summary_names is not None else None
    if arguments.ground_truth_format == "json":
        ground_truth_by_image = load_json_ground_truth(arguments.data_root.resolve(), selected_names)
    else:
        if arguments.image_root is None or arguments.ground_truth_root is None:
            raise ValueError("YOLO 모드는 --image-root와 --ground-truth-root가 필요합니다.")
        ground_truth_by_image = load_yolo_ground_truth(
            arguments.image_root,
            arguments.ground_truth_root,
            selected_names,
        )

    image_names = summary_names or sorted(ground_truth_by_image)
    missing = [name for name in image_names if name not in ground_truth_by_image]
    if missing:
        raise FileNotFoundError(f"정답에서 찾지 못한 예측 이미지가 있습니다. 예: {missing[:3]}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    evaluation_name = f"quarter_evaluation_{timestamp}"
    metrics_dir = METRICS_ROOT / evaluation_name
    fp_dir = ERRORS_ROOT / "false_positive" / evaluation_name
    fn_dir = ERRORS_ROOT / "false_negative" / evaluation_name
    metrics_dir.mkdir(parents=True, exist_ok=False)
    fp_dir.mkdir(parents=True, exist_ok=False)
    fn_dir.mkdir(parents=True, exist_ok=False)
    logger = make_logger(metrics_dir / "evaluation.log")
    logger.info(f"[INFO] time_local: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    logger.info(f"[INFO] time_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    logger.info(f"[INFO] prediction_run={prediction_run}, prediction_csv={prediction_csv}")
    logger.info(f"[INFO] ground_truth_format={arguments.ground_truth_format}, images={len(image_names):,}")
    logger.info(f"[INFO] confidence={arguments.confidence}, match_iou={arguments.match_iou}")

    per_image_rows = []
    match_rows = []
    totals = {"gt": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
    matched_ious = []
    for number, image_name in enumerate(image_names, start=1):
        item = ground_truth_by_image[image_name]
        ground_truths = item["boxes"]
        predictions = predictions_by_image.get(image_name, [])
        matches, false_negatives = match_one_image(predictions, ground_truths, arguments.match_iou)
        tp = sum(match["is_true_positive"] for match in matches)
        fp = len(predictions) - tp
        fn = len(false_negatives)
        image_ious = [match["iou"] for match in matches if match["is_true_positive"]]
        matched_ious.extend(image_ious)
        totals["gt"] += len(ground_truths)
        totals["predictions"] += len(predictions)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        per_image_rows.append(
            {
                "image_name": image_name,
                "ground_truths": len(ground_truths),
                "predictions": len(predictions),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": safe_divide(tp, tp + fp),
                "recall": safe_divide(tp, tp + fn),
                "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
                "mean_iou": sum(image_ious) / len(image_ious) if image_ious else 0.0,
            }
        )
        for match in matches:
            prediction = predictions[match["prediction_index"]]
            match_rows.append(
                {
                    "image_name": image_name,
                    "result": "TP" if match["is_true_positive"] else "FP",
                    "prediction_index": match["prediction_index"] + 1,
                    "ground_truth_index": match["ground_truth_index"] + 1 if match["ground_truth_index"] is not None else "",
                    "confidence": prediction["confidence"],
                    "iou": match["iou"],
                }
            )
        for ground_truth_index in false_negatives:
            match_rows.append(
                {
                    "image_name": image_name,
                    "result": "FN",
                    "prediction_index": "",
                    "ground_truth_index": ground_truth_index + 1,
                    "confidence": "",
                    "iou": 0.0,
                }
            )
        if fp or fn:
            save_errors(item["image_path"], ground_truths, predictions, matches, false_negatives, fp_dir, fn_dir)
        if number == 1 or number % 50 == 0 or number == len(image_names):
            logger.info(f"[LOG] evaluated {number:04d}/{len(image_names):04d}")

    precision = safe_divide(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_divide(totals["tp"], totals["tp"] + totals["fn"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    metrics = {
        "prediction_run": str(prediction_run),
        "ground_truth_format": arguments.ground_truth_format,
        "confidence_threshold": arguments.confidence,
        "match_iou_threshold": arguments.match_iou,
        "test_images": len(image_names),
        "ground_truth_count": totals["gt"],
        "prediction_count": totals["predictions"],
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "detection_rate_percent": recall * 100.0,
        "mean_iou_of_true_positives": mean_iou,
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(metrics_dir / "metrics.csv", [{"metric": key, "value": value} for key, value in metrics.items()], ["metric", "value"])
    write_csv(metrics_dir / "per_image_metrics.csv", per_image_rows, list(per_image_rows[0].keys()))
    write_csv(
        metrics_dir / "prediction_matches.csv",
        match_rows,
        ["image_name", "result", "prediction_index", "ground_truth_index", "confidence", "iou"],
    )
    logger.info(f"[RESULT] TP={totals['tp']}, FP={totals['fp']}, FN={totals['fn']}")
    logger.info(f"[RESULT] Precision={precision:.6f}, Recall={recall:.6f}, F1={f1:.6f}, mean_IoU={mean_iou:.6f}")
    logger.info(f"[RESULT] metrics={metrics_dir}")
    logger.info(f"[RESULT] false_positive_images={fp_dir}")
    logger.info(f"[RESULT] false_negative_images={fn_dir}")


if __name__ == "__main__":
    main()
