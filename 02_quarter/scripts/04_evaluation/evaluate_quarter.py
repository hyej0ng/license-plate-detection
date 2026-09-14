"""Global NMS 예측을 JSON 또는 YOLO 정답과 원본 이미지 단위로 평가함
실행방법:
python 02_quarter/scripts/04_evaluation/evaluate_quarter.py \
  --predictions /home/hyejong/landing_pjt/02_quarter/results/predictions/quarter_inference_20260908-162018 \
  --ground-truth-format json \
  --data-root /mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset \
  --confidence 0.25 \
  --match-iou 0.50
"""

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
import matplotlib
import numpy as np


# 화면이 없는 서버에서도 결과 그래프를 PNG로 저장한다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


def read_predictions(path: Path) -> dict[str, list[dict]]:
    """저장된 모든 예측을 이미지별 confidence 내림차순으로 읽는다."""
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


def read_image_summary(path: Path | None) -> list[dict[str, str]] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"image_name", "image_width", "image_height"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"요약 CSV 필수 열이 없습니다: {sorted(required)}")
    return rows


def load_json_ground_truth(
    data_root: Path,
    selected_names: set[str] | None = None,
    image_sizes: dict[str, tuple[int, int]] | None = None,
) -> dict[str, dict]:
    ground_truth = {}
    for stem, image_path, json_path in matching_pairs(data_root, "test"):
        if selected_names is not None and image_path.name not in selected_names:
            continue
        if image_sizes is not None and image_path.name in image_sizes:
            width, height = image_sizes[image_path.name]
        else:
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


def calculate_ap(
    all_image_data: dict[str, dict],
    iou_threshold: float,
    total_ground_truths: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """저장된 전체 예측을 confidence 순으로 나열해 101-point interpolated AP를 계산한다."""
    ranked_predictions = []
    for image_name, image_data in all_image_data.items():
        for prediction in image_data["predictions"]:
            ranked_predictions.append(
                {
                    "image_name": image_name,
                    "prediction": prediction,
                }
            )
    ranked_predictions.sort(
        key=lambda item: item["prediction"]["confidence"],
        reverse=True,
    )

    used_ground_truths = {image_name: set() for image_name in all_image_data}
    true_positives = []
    false_positives = []
    for item in ranked_predictions:
        image_name = item["image_name"]
        prediction = item["prediction"]
        ground_truths = all_image_data[image_name]["ground_truths"]
        best_index = None
        best_iou = 0.0
        for ground_truth_index, ground_truth in enumerate(ground_truths):
            if ground_truth_index in used_ground_truths[image_name]:
                continue
            if prediction["class_id"] != ground_truth["class_id"]:
                continue
            iou = calculate_iou(prediction["box"], ground_truth["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = ground_truth_index

        if best_index is not None and best_iou >= iou_threshold:
            used_ground_truths[image_name].add(best_index)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    if not ranked_predictions or total_ground_truths == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recalls = cumulative_tp / total_ground_truths
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = [
        float(precisions[recalls >= recall_point].max())
        if np.any(recalls >= recall_point)
        else 0.0
        for recall_point in recall_points
    ]
    return float(np.mean(interpolated_precisions)), recalls, precisions


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_pr_curve(recalls: np.ndarray, precisions: np.ndarray, ap50: float, output_path: Path) -> None:
    """저장된 예측의 Precision-Recall curve를 저장한다."""
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(recalls, precisions, linewidth=2, label=f"AP@0.50 = {ap50:.4f}")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Quarter Test Precision-Recall Curve")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_evaluation_summary(
    metrics: dict,
    recalls: np.ndarray,
    precisions: np.ndarray,
    output_path: Path,
) -> None:
    """AP/mAP와 PR curve를 포함한 최종 test 요약 그림을 저장한다."""
    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    metric_axis = figure.add_subplot(grid[0, 0])

    metric_names = ["Precision", "Recall", "F1", "mIoU(TP)", "AP50", "mAP50-95"]
    metric_values = [
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["mean_iou_of_true_positives"],
        metrics["AP50_from_saved_predictions"],
        metrics["mAP50_95_from_saved_predictions"],
    ]
    metric_bars = metric_axis.bar(metric_names, metric_values, color="#2878B5")
    metric_axis.set_ylim(0.0, 1.05)
    metric_axis.set_ylabel("score")
    metric_axis.set_title("Main test metrics")
    metric_axis.grid(axis="y", alpha=0.3)
    metric_axis.set_axisbelow(True)
    metric_axis.tick_params(axis="x", rotation=25)
    for bar, value in zip(metric_bars, metric_values):
        metric_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    count_axis = figure.add_subplot(grid[0, 1])
    count_names = ["TP", "FP", "FN"]
    count_values = [
        metrics["true_positives"],
        metrics["false_positives"],
        metrics["false_negatives"],
    ]
    count_bars = count_axis.bar(
        count_names,
        count_values,
        color=["#2CA02C", "#D62728", "#FF8C00"],
    )
    count_axis.set_ylabel("number of boxes")
    count_axis.set_title("Detection counts")
    count_axis.grid(axis="y", alpha=0.3)
    count_axis.set_axisbelow(True)
    for bar, value in zip(count_bars, count_values):
        count_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    pr_axis = figure.add_subplot(grid[1, :])
    pr_axis.plot(recalls, precisions, linewidth=2, color="#2878B5")
    pr_axis.set_xlim(0.0, 1.0)
    pr_axis.set_ylim(0.0, 1.05)
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_title(
        f"Precision-Recall curve  |  AP50={metrics['AP50_from_saved_predictions']:.3f}"
    )
    pr_axis.grid(alpha=0.3)

    figure.suptitle(
        "YOLOv26n Quarter - Final Test Evaluation\n"
        f"images={metrics['test_images']:,}, GT={metrics['ground_truth_count']:,}, "
        f"confidence>={metrics['confidence_threshold']:.2f}, "
        f"match IoU>={metrics['match_iou_threshold']:.2f}",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.01,
        "AP/mAP uses all predictions available in the saved detections.csv.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    figure.tight_layout(rect=[0.0, 0.035, 1.0, 0.93])
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    validate_ratio("confidence_threshold", arguments.confidence)
    validate_ratio("matching_iou_threshold", arguments.match_iou)
    prediction_csv, summary_csv, prediction_run = resolve_prediction_files(arguments.predictions)
    all_predictions_by_image = read_predictions(prediction_csv)
    summary_rows = read_image_summary(summary_csv)
    summary_names = (
        [row["image_name"] for row in summary_rows]
        if summary_rows is not None
        else None
    )
    image_sizes = (
        {
            row["image_name"]: (int(row["image_width"]), int(row["image_height"]))
            for row in summary_rows
        }
        if summary_rows is not None
        else None
    )

    selected_names = set(summary_names) if summary_names is not None else None
    if arguments.ground_truth_format == "json":
        ground_truth_by_image = load_json_ground_truth(
            arguments.data_root.resolve(),
            selected_names,
            image_sizes,
        )
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
    all_image_data = {}
    totals = {"gt": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
    matched_ious = []
    all_saved_confidences = []
    for number, image_name in enumerate(image_names, start=1):
        item = ground_truth_by_image[image_name]
        ground_truths = item["boxes"]
        all_predictions = all_predictions_by_image.get(image_name, [])
        predictions = [
            prediction
            for prediction in all_predictions
            if prediction["confidence"] >= arguments.confidence
        ]
        all_saved_confidences.extend(
            prediction["confidence"] for prediction in all_predictions
        )
        all_image_data[image_name] = {
            "ground_truths": ground_truths,
            "predictions": all_predictions,
        }
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

    average_precisions = {}
    ap50_recalls = np.array([0.0])
    ap50_precisions = np.array([0.0])
    for iou_threshold in np.arange(0.50, 0.96, 0.05):
        average_precision, recalls, precisions = calculate_ap(
            all_image_data,
            float(iou_threshold),
            totals["gt"],
        )
        average_precisions[f"AP@{iou_threshold:.2f}"] = average_precision
        if np.isclose(iou_threshold, 0.50):
            ap50_recalls = recalls
            ap50_precisions = precisions
    ap50 = average_precisions["AP@0.50"]
    map50_95 = float(np.mean(list(average_precisions.values())))

    metrics = {
        "prediction_run": str(prediction_run),
        "ground_truth_format": arguments.ground_truth_format,
        "confidence_threshold": arguments.confidence,
        "minimum_confidence_in_saved_predictions": (
            min(all_saved_confidences) if all_saved_confidences else None
        ),
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
        "AP50_from_saved_predictions": ap50,
        "mAP50_95_from_saved_predictions": map50_95,
        **average_precisions,
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(metrics_dir / "metrics.csv", [{"metric": key, "value": value} for key, value in metrics.items()], ["metric", "value"])
    write_csv(metrics_dir / "per_image_metrics.csv", per_image_rows, list(per_image_rows[0].keys()))
    write_csv(
        metrics_dir / "prediction_matches.csv",
        match_rows,
        ["image_name", "result", "prediction_index", "ground_truth_index", "confidence", "iou"],
    )
    pr_curve_path = metrics_dir / "pr_curve.png"
    summary_path = metrics_dir / "evaluation_summary.png"
    save_pr_curve(ap50_recalls, ap50_precisions, ap50, pr_curve_path)
    save_evaluation_summary(metrics, ap50_recalls, ap50_precisions, summary_path)
    logger.info(f"[RESULT] TP={totals['tp']}, FP={totals['fp']}, FN={totals['fn']}")
    logger.info(f"[RESULT] Precision={precision:.6f}, Recall={recall:.6f}, F1={f1:.6f}, mean_IoU={mean_iou:.6f}")
    logger.info(
        f"[RESULT] AP50={ap50:.6f}, mAP50-95={map50_95:.6f} "
        "(all saved predictions)"
    )
    logger.info(f"[RESULT] metrics={metrics_dir}")
    logger.info(f"[RESULT] summary_image={summary_path}")
    logger.info(f"[RESULT] pr_curve={pr_curve_path}")
    logger.info(f"[RESULT] false_positive_images={fp_dir}")
    logger.info(f"[RESULT] false_negative_images={fn_dir}")
    logger.info(
        "[INFO] Precision/Recall/F1은 --confidence 기준, AP/mAP는 저장된 전체 예측을 사용합니다. "
        "inference confidence 미만의 예측은 detections.csv에 없으므로 포함되지 않습니다."
    )


if __name__ == "__main__":
    main()
