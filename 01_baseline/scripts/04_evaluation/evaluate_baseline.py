"""
실행방법:
python 01_baseline/scripts/04_evaluation/evaluate_baseline.py \
  --predictions 01_baseline/results/predictions/baseline_inference_20260819-150000 \
  --conf 0.25 (또는 원하는 값)
"""

import argparse
import csv
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
import numpy as np


# 화면이 없는 서버에서도 그래프를 저장한다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. 기본 경로와 평가 기준
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PREDICTION_RUN = (
    PROJECT_ROOT
    / "01_baseline"
    / "results"
    / "predictions"
    / "baseline_inference_20260812-161125"
)
TEST_IMAGE_DIR = PROJECT_ROOT / "01_baseline" / "data" / "preprocessed" / "images" / "test"
GROUND_TRUTH_DIR = PROJECT_ROOT / "01_baseline" / "data" / "preprocessed" / "labels" / "test"
METRICS_ROOT = PROJECT_ROOT / "01_baseline" / "results" / "metrics"
ERRORS_ROOT = PROJECT_ROOT / "01_baseline" / "results" / "errors"

CONFIDENCE_THRESHOLD = 0.25
MATCH_IOU_THRESHOLD = 0.50
INFERENCE_IOU_ARGUMENT = 0.70

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_arguments():
    """예측 폴더와 평가 threshold를 입력받는다."""
    parser = argparse.ArgumentParser(description="YOLOv26n baseline test 평가")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTION_RUN,
        help="inference 결과 폴더 (labels 폴더를 포함해야 함)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="평가할 최소 confidence (기본값: 0.25)",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=MATCH_IOU_THRESHOLD,
        help="TP로 인정할 IoU 기준 (기본값: 0.50)",
    )
    parser.add_argument(
        "--inference-iou",
        type=float,
        default=INFERENCE_IOU_ARGUMENT,
        help="inference 당시 기록할 IoU 인자 (기본값: 0.70)",
    )
    return parser.parse_args()


def make_logger(log_path):
    """터미널과 파일에 같은 메시지를 기록한다."""
    logger = logging.getLogger(f"baseline_evaluation_{log_path.stem}")
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


def find_test_images():
    """test 이미지를 파일명 순서로 읽는다."""
    images = sorted(
        path.resolve()
        for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"test 이미지가 없습니다: {TEST_IMAGE_DIR}")
    return images


def yolo_to_xyxy(center_x, center_y, width, height):
    """정규화된 YOLO bbox를 정규화된 xyxy 좌표로 변환한다."""
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def read_ground_truth(label_path):
    """정답 YOLO TXT를 읽는다: class cx cy width height."""
    boxes = []
    if not label_path.is_file():
        raise FileNotFoundError(f"정답 라벨이 없습니다: {label_path}")

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"잘못된 정답 형식: {label_path}:{line_number}")

        class_id = int(parts[0])
        center_x, center_y, width, height = (float(value) for value in parts[1:])
        boxes.append(
            {
                "class_id": class_id,
                "box": yolo_to_xyxy(center_x, center_y, width, height),
            }
        )
    return boxes


def read_predictions(label_path):
    """예측 YOLO TXT 전체를 읽는다: class cx cy width height confidence."""
    boxes = []
    if not label_path.is_file():
        raise FileNotFoundError(f"예측 라벨이 없습니다: {label_path}")

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"잘못된 예측 형식: {label_path}:{line_number}")

        class_id = int(parts[0])
        center_x, center_y, width, height, confidence = (
            float(value) for value in parts[1:]
        )
        boxes.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "box": yolo_to_xyxy(center_x, center_y, width, height),
            }
        )

    return sorted(boxes, key=lambda item: item["confidence"], reverse=True)


def make_prediction_map(prediction_label_dir, test_images):
    """예측 TXT를 실제 test 이미지 stem과 연결한다."""
    prediction_paths = sorted(prediction_label_dir.glob("*.txt"))
    if not prediction_paths:
        raise FileNotFoundError(f"예측 TXT가 없습니다: {prediction_label_dir}")

    test_stems = {image_path.stem for image_path in test_images}
    prediction_map = {}

    for prediction_path in prediction_paths:
        stem = prediction_path.stem

        # 새 inference 결과: 원래 이미지 stem이 보존된 경우
        if stem in test_stems:
            prediction_map[stem] = prediction_path
            continue

        # 기존 inference 결과: image0, image1처럼 바뀐 이름을 입력 순서로 복원
        match = re.fullmatch(r"image(\d+)", stem)
        if match:
            image_index = int(match.group(1))
            if image_index >= len(test_images):
                raise ValueError(f"test 범위를 벗어난 예측 이름입니다: {prediction_path.name}")
            prediction_map[test_images[image_index].stem] = prediction_path
            continue

        raise ValueError(f"test 이미지와 연결할 수 없는 예측 파일입니다: {prediction_path.name}")

    if len(prediction_map) != len(test_images):
        missing = sorted(test_stems - set(prediction_map))
        raise ValueError(
            f"test 이미지 {len(test_images)}장과 예측 {len(prediction_map)}장이 일치하지 않습니다. "
            f"누락 예시: {missing[:3]}"
        )

    return prediction_map


def calculate_iou(box_a, box_b):
    """두 xyxy bbox의 IoU를 계산한다."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def match_one_image(predictions, ground_truths, iou_threshold):
    """confidence가 높은 예측부터 아직 사용하지 않은 정답과 1:1 매칭한다."""
    used_ground_truths = set()
    matched_predictions = []

    for prediction_index, prediction in enumerate(predictions):
        best_ground_truth_index = None
        best_iou = 0.0

        for ground_truth_index, ground_truth in enumerate(ground_truths):
            if ground_truth_index in used_ground_truths:
                continue
            if prediction["class_id"] != ground_truth["class_id"]:
                continue

            iou = calculate_iou(prediction["box"], ground_truth["box"])
            if iou > best_iou:
                best_iou = iou
                best_ground_truth_index = ground_truth_index

        is_true_positive = (
            best_ground_truth_index is not None and best_iou >= iou_threshold
        )
        if is_true_positive:
            used_ground_truths.add(best_ground_truth_index)

        matched_predictions.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": best_ground_truth_index if is_true_positive else None,
                "is_true_positive": is_true_positive,
                "iou": best_iou if is_true_positive else 0.0,
            }
        )

    false_negative_indices = [
        index for index in range(len(ground_truths)) if index not in used_ground_truths
    ]
    return matched_predictions, false_negative_indices


def safe_divide(numerator, denominator):
    """0으로 나누는 경우 0을 반환한다."""
    return numerator / denominator if denominator else 0.0


def calculate_ap(all_image_data, iou_threshold, total_ground_truths):
    """한 IoU 기준에서 confidence 순위 기반 101-point interpolated AP를 계산한다."""
    ranked_predictions = []

    for image_name, image_data in all_image_data.items():
        for prediction_index, prediction in enumerate(image_data["predictions"]):
            ranked_predictions.append(
                {
                    "image_name": image_name,
                    "prediction_index": prediction_index,
                    "prediction": prediction,
                }
            )

    ranked_predictions.sort(
        key=lambda item: item["prediction"]["confidence"], reverse=True
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
    interpolated_precisions = []
    for recall_point in recall_points:
        valid = precisions[recalls >= recall_point]
        interpolated_precisions.append(float(valid.max()) if len(valid) else 0.0)

    return float(np.mean(interpolated_precisions)), recalls, precisions


def normalized_box_to_pixels(box, image_width, image_height):
    """정규화된 xyxy 좌표를 이미지 픽셀 좌표로 바꾼다."""
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * image_width)),
        int(round(y1 * image_height)),
        int(round(x2 * image_width)),
        int(round(y2 * image_height)),
    )


def draw_box(image, box, color, text):
    """이미지 위에 bbox와 설명을 그린다."""
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = normalized_box_to_pixels(box, image_width, image_height)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        text,
        (x1, max(y1 - 5, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def save_error_visualization(
    image_path,
    ground_truths,
    predictions,
    matched_predictions,
    false_negative_indices,
    false_positive_dir,
    false_negative_dir,
):
    """FP는 빨강, FN은 주황, TP는 파랑, 정답은 초록으로 표시한다."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")

    for ground_truth_index, ground_truth in enumerate(ground_truths):
        label = "GT"
        color = (0, 255, 0)
        if ground_truth_index in false_negative_indices:
            label = "FN"
            color = (0, 165, 255)
        draw_box(image, ground_truth["box"], color, label)

    has_false_positive = False
    for match in matched_predictions:
        prediction = predictions[match["prediction_index"]]
        if match["is_true_positive"]:
            text = f"TP {prediction['confidence']:.2f} IoU {match['iou']:.2f}"
            color = (255, 0, 0)
        else:
            text = f"FP {prediction['confidence']:.2f}"
            color = (0, 0, 255)
            has_false_positive = True
        draw_box(image, prediction["box"], color, text)

    if has_false_positive:
        cv2.imwrite(str(false_positive_dir / image_path.name), image)
    if false_negative_indices:
        cv2.imwrite(str(false_negative_dir / image_path.name), image)


def save_pr_curve(recalls, precisions, ap50, output_path):
    """저장된 예측의 Precision-Recall curve를 저장한다."""
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(recalls, precisions, linewidth=2, label=f"AP@0.50 = {ap50:.4f}")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Test Precision-Recall Curve")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_evaluation_summary(metrics, recalls, precisions, output_path):
    """최종 test 성능을 한 장에서 확인할 수 있는 요약 그림을 저장한다."""
    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.15])

    # 왼쪽 위: 0~1 범위의 핵심 성능 지표
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
    bars = metric_axis.bar(metric_names, metric_values, color="#2878B5")
    metric_axis.set_ylim(0.0, 1.05)
    metric_axis.set_ylabel("score")
    metric_axis.set_title("Main test metrics")
    metric_axis.grid(axis="y", alpha=0.3)
    metric_axis.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, metric_values):
        metric_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 오른쪽 위: 맞춘 것, 잘못 찾은 것, 놓친 것의 개수
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
    for bar, value in zip(count_bars, count_values):
        count_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    # 아래: confidence 순위에 따른 Precision-Recall 관계
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
        "YOLOv26n Baseline - Final Test Evaluation\n"
        f"images={metrics['test_images']:,}, GT={metrics['ground_truth_count']:,}, "
        f"confidence>={metrics['confidence_threshold']:.2f}, "
        f"match IoU>={metrics['match_iou_threshold']:.2f}",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.01,
        "AP/mAP uses all predictions available in the saved inference labels.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    figure.tight_layout(rect=[0.0, 0.035, 1.0, 0.93])
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_metrics_csv(metrics, output_path):
    """요약 지표를 이름과 값 두 열의 CSV로 저장한다."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for metric_name, value in metrics.items():
            writer.writerow([metric_name, value])


def main():
    arguments = parse_arguments()
    prediction_run = arguments.predictions.resolve()
    prediction_label_dir = prediction_run / "labels"

    for name, value in (
        ("--conf", arguments.conf),
        ("--match-iou", arguments.match_iou),
        ("--inference-iou", arguments.inference_iou),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}는 0~1 사이여야 합니다.")

    test_images = find_test_images()
    prediction_map = make_prediction_map(prediction_label_dir, test_images)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    metrics_dir = METRICS_ROOT / f"baseline_evaluation_{timestamp}"
    false_positive_dir = ERRORS_ROOT / "false_positive" / f"baseline_evaluation_{timestamp}"
    false_negative_dir = ERRORS_ROOT / "false_negative" / f"baseline_evaluation_{timestamp}"
    metrics_dir.mkdir(parents=True, exist_ok=False)
    false_positive_dir.mkdir(parents=True, exist_ok=False)
    false_negative_dir.mkdir(parents=True, exist_ok=False)

    logger = make_logger(metrics_dir / "evaluation.log")
    logger.info(f"[INFO] time_local: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    logger.info(f"[INFO] time_utc:   {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    logger.info(f"[INFO] prediction_run: {prediction_run}")
    logger.info(f"[INFO] test_images: {len(test_images):,}")
    logger.info(f"[INFO] confidence_threshold: {arguments.conf}")
    logger.info(f"[INFO] TP_match_IoU_threshold: {arguments.match_iou}")
    logger.info(f"[INFO] inference_IoU_argument: {arguments.inference_iou}")

    all_image_data = {}
    per_image_rows = []
    match_rows = []
    total_ground_truths = 0
    total_predictions = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    true_positive_ious = []
    all_saved_confidences = []

    for image_number, image_path in enumerate(test_images, start=1):
        ground_truths = read_ground_truth(GROUND_TRUTH_DIR / f"{image_path.stem}.txt")
        all_predictions = read_predictions(prediction_map[image_path.stem])
        predictions = [
            prediction
            for prediction in all_predictions
            if prediction["confidence"] >= arguments.conf
        ]
        all_saved_confidences.extend(
            prediction["confidence"] for prediction in all_predictions
        )
        matched_predictions, false_negative_indices = match_one_image(
            predictions,
            ground_truths,
            arguments.match_iou,
        )

        true_positives = sum(
            1 for match in matched_predictions if match["is_true_positive"]
        )
        false_positives = len(predictions) - true_positives
        false_negatives = len(false_negative_indices)

        total_ground_truths += len(ground_truths)
        total_predictions += len(predictions)
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        for match in matched_predictions:
            prediction = predictions[match["prediction_index"]]
            if match["is_true_positive"]:
                true_positive_ious.append(match["iou"])
            match_rows.append(
                {
                    "image_name": image_path.name,
                    "prediction_index": match["prediction_index"] + 1,
                    "confidence": prediction["confidence"],
                    "result": "TP" if match["is_true_positive"] else "FP",
                    "iou": match["iou"],
                    "ground_truth_index": (
                        match["ground_truth_index"] + 1
                        if match["ground_truth_index"] is not None
                        else ""
                    ),
                }
            )

        per_image_rows.append(
            {
                "image_name": image_path.name,
                "ground_truths": len(ground_truths),
                "predictions": len(predictions),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "mean_matched_iou": (
                    np.mean(
                        [
                            match["iou"]
                            for match in matched_predictions
                            if match["is_true_positive"]
                        ]
                    )
                    if true_positives
                    else 0.0
                ),
            }
        )

        all_image_data[image_path.name] = {
            "ground_truths": ground_truths,
            # AP/mAP는 평가 conf가 아니라 저장된 전체 confidence 순위를 사용한다.
            "predictions": all_predictions,
        }

        if false_positives or false_negatives:
            save_error_visualization(
                image_path,
                ground_truths,
                predictions,
                matched_predictions,
                false_negative_indices,
                false_positive_dir,
                false_negative_dir,
            )

        if image_number == 1 or image_number % 50 == 0:
            logger.info(f"[LOG] evaluated {image_number:04d}/{len(test_images):04d}")

    precision = safe_divide(
        total_true_positives, total_true_positives + total_false_positives
    )
    recall = safe_divide(
        total_true_positives, total_true_positives + total_false_negatives
    )
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    detection_rate = safe_divide(total_true_positives, total_ground_truths)
    mean_iou = float(np.mean(true_positive_ious)) if true_positive_ious else 0.0

    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    average_precisions = {}
    ap50_recalls = np.array([0.0])
    ap50_precisions = np.array([0.0])

    for iou_threshold in iou_thresholds:
        average_precision, recalls, precisions = calculate_ap(
            all_image_data,
            float(iou_threshold),
            total_ground_truths,
        )
        average_precisions[f"AP@{iou_threshold:.2f}"] = average_precision
        if math.isclose(iou_threshold, 0.50):
            ap50_recalls = recalls
            ap50_precisions = precisions

    ap50 = average_precisions["AP@0.50"]
    map50_95 = float(np.mean(list(average_precisions.values())))

    metrics = {
        "confidence_threshold": arguments.conf,
        "minimum_confidence_in_saved_predictions": (
            min(all_saved_confidences) if all_saved_confidences else None
        ),
        "match_iou_threshold": arguments.match_iou,
        "inference_iou_argument": arguments.inference_iou,
        "test_images": len(test_images),
        "ground_truth_count": total_ground_truths,
        "prediction_count": total_predictions,
        "true_positives": total_true_positives,
        "false_positives": total_false_positives,
        "false_negatives": total_false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "detection_rate_percent": detection_rate * 100.0,
        "mean_iou_of_true_positives": mean_iou,
        "AP50_from_saved_predictions": ap50,
        "mAP50_95_from_saved_predictions": map50_95,
        **average_precisions,
    }

    with (metrics_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    save_metrics_csv(metrics, metrics_dir / "metrics.csv")

    with (metrics_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=per_image_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_image_rows)

    with (metrics_dir / "prediction_matches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [
            "image_name",
            "prediction_index",
            "confidence",
            "result",
            "iou",
            "ground_truth_index",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(match_rows)

    save_pr_curve(ap50_recalls, ap50_precisions, ap50, metrics_dir / "pr_curve.png")
    save_evaluation_summary(
        metrics,
        ap50_recalls,
        ap50_precisions,
        metrics_dir / "evaluation_summary.png",
    )

    logger.info("[RESULT] Evaluation finished")
    logger.info(
        f"[RESULT] TP={total_true_positives}, FP={total_false_positives}, "
        f"FN={total_false_negatives}"
    )
    logger.info(
        f"[RESULT] Precision={precision:.6f}, Recall={recall:.6f}, "
        f"F1={f1_score:.6f}"
    )
    logger.info(
        f"[RESULT] Detection rate={detection_rate * 100:.2f}%, "
        f"mIoU(TP)={mean_iou:.6f}"
    )
    logger.info(
        f"[RESULT] AP50={ap50:.6f}, mAP50-95={map50_95:.6f} "
        "(all saved predictions)"
    )
    logger.info(f"[INFO] metrics: {metrics_dir}")
    logger.info(f"[INFO] false_positive_images: {false_positive_dir}")
    logger.info(f"[INFO] false_negative_images: {false_negative_dir}")
    logger.info(
        "[INFO] Precision/Recall/F1은 --conf 기준, AP/mAP는 저장된 전체 예측을 사용합니다. "
        "현재 inference가 conf=0.25였다면 그보다 낮은 예측은 저장되지 않았으므로, "
        "공식 전체 PR mAP가 필요하면 conf=0.001로 inference 후 다시 평가하세요."
    )


if __name__ == "__main__":
    main()
