"""저장된 test 예측을 class-aware IoU 1:1 매칭으로 평가한다."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "03_two_stage"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from two_stage_utils import make_stage_logger  # noqa: E402


DEFAULT_DATA = PROJECT_DIR / "configs" / "license_plate_vehicle.yaml"
METRICS_ROOT = PROJECT_DIR / "results" / "metrics"
ERRORS_ROOT = PROJECT_DIR / "results" / "errors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vehicle/license_plate test 평가")
    parser.add_argument("--predictions", type=Path, required=True, help="inference 결과 폴더")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--confidence", "--conf", dest="confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def read_dataset(data_yaml: Path) -> tuple[Path, Path, dict[int, str]]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"데이터 YAML이 없습니다: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    image_dir = root / config["test"]
    label_dir = root / "labels" / "test"
    names_value = config["names"]
    names = ({int(key): str(value) for key, value in names_value.items()}
             if isinstance(names_value, dict) else dict(enumerate(names_value)))
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"test images/labels가 없습니다: {image_dir}, {label_dir}")
    return image_dir, label_dir, names


def read_summary(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"image_summary.csv가 없습니다: {path}")
    with path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"image_name", "image_width", "image_height"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"image_summary.csv 필수 열이 없습니다: {sorted(required)}")
    return rows


def read_predictions(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"detections.csv가 없습니다: {path}")
    predictions: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_name", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"detections.csv 필수 열이 없습니다: {sorted(required)}")
        for row in reader:
            predictions[row["image_name"]].append({
                "class_id": int(row["class_id"]), "confidence": float(row["confidence"]),
                "box": tuple(float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax")),
            })
    for boxes in predictions.values():
        boxes.sort(key=lambda item: item["confidence"], reverse=True)
    return predictions


def read_ground_truth(path: Path, width: int, height: int, valid_classes: set[int]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"정답 라벨이 없습니다: {path}")
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"정답 라벨 형식 오류: {path}:{line_number}")
        class_id = int(parts[0])
        if class_id not in valid_classes:
            raise ValueError(f"알 수 없는 class ID: {path}:{line_number}: {class_id}")
        cx, cy, box_width, box_height = map(float, parts[1:])
        boxes.append({"class_id": class_id, "box": (
            (cx - box_width / 2) * width, (cy - box_height / 2) * height,
            (cx + box_width / 2) * width, (cy + box_height / 2) * height,
        )})
    return boxes


def match_boxes(predictions: list[dict], ground_truths: list[dict], threshold: float) -> tuple[list[dict], list[int]]:
    used = set()
    matches = []
    for prediction_index, prediction in enumerate(predictions):
        best_index, best_iou = None, 0.0
        for gt_index, ground_truth in enumerate(ground_truths):
            if gt_index in used or prediction["class_id"] != ground_truth["class_id"]:
                continue
            score = calculate_iou(prediction["box"], ground_truth["box"])
            if score > best_iou:
                best_index, best_iou = gt_index, score
        is_tp = best_index is not None and best_iou >= threshold
        if is_tp:
            used.add(best_index)
        matches.append({"prediction_index": prediction_index, "ground_truth_index": best_index if is_tp else None,
                        "is_true_positive": is_tp, "iou": best_iou if is_tp else 0.0})
    return matches, [index for index in range(len(ground_truths)) if index not in used]


def average_precision(records: dict[str, dict], class_id: int, threshold: float) -> tuple[float, np.ndarray, np.ndarray]:
    total_ground_truths = sum(
        1 for record in records.values() for box in record["ground_truths"] if box["class_id"] == class_id
    )
    ranked = []
    for image_name, record in records.items():
        for prediction in record["predictions"]:
            if prediction["class_id"] == class_id:
                ranked.append((prediction["confidence"], image_name, prediction))
    ranked.sort(key=lambda item: item[0], reverse=True)
    used = {image_name: set() for image_name in records}
    tp, fp = [], []
    for _, image_name, prediction in ranked:
        ground_truths = records[image_name]["ground_truths"]
        best_index, best_iou = None, 0.0
        for index, ground_truth in enumerate(ground_truths):
            if index in used[image_name] or ground_truth["class_id"] != class_id:
                continue
            score = calculate_iou(prediction["box"], ground_truth["box"])
            if score > best_iou:
                best_index, best_iou = index, score
        is_tp = best_index is not None and best_iou >= threshold
        if is_tp:
            used[image_name].add(best_index)
        tp.append(int(is_tp))
        fp.append(int(not is_tp))
    if total_ground_truths == 0 or not ranked:
        return 0.0, np.array([0.0]), np.array([0.0])
    cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
    recalls = cumulative_tp / total_ground_truths
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall_points = np.linspace(0.0, 1.0, 101)
    ap = np.mean([precisions[recalls >= point].max() if np.any(recalls >= point) else 0.0 for point in recall_points])
    return float(ap), recalls, precisions


def draw_box(image: np.ndarray, box: tuple[float, ...], color: tuple[int, ...], text: str) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


def save_error_image(image_path: Path, predictions: list[dict], ground_truths: list[dict],
                     matches: list[dict], false_negatives: list[int], names: dict[int, str],
                     fp_dir: Path, fn_dir: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")
    for index, ground_truth in enumerate(ground_truths):
        is_fn = index in false_negatives
        draw_box(image, ground_truth["box"], (0, 165, 255) if is_fn else (0, 255, 0),
                 f"{'FN' if is_fn else 'GT'} {names[ground_truth['class_id']]}")
    has_fp = False
    for match in matches:
        prediction = predictions[match["prediction_index"]]
        if match["is_true_positive"]:
            color, prefix = (255, 0, 0), "TP"
        else:
            color, prefix, has_fp = (0, 0, 255), "FP", True
        draw_box(image, prediction["box"], color,
                 f"{prefix} {names[prediction['class_id']]} {prediction['confidence']:.2f}")
    if has_fp:
        cv2.imwrite(str(fp_dir / image_path.name), image)
    if false_negatives:
        cv2.imwrite(str(fn_dir / image_path.name), image)


def save_summary(metrics: dict, output_path: Path) -> None:
    """전체 및 클래스별 핵심 지표를 한 장의 그림으로 저장한다."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    overall_names = ["Precision", "Recall", "F1", "mIoU(TP)", "mAP50", "mAP50-95"]
    overall_values = [metrics[key] for key in (
        "precision", "recall", "f1_score", "mean_iou_of_true_positives", "mAP50", "mAP50_95"
    )]
    bars = axes[0].bar(overall_names, overall_values, color="#2878B5")
    axes[0].set(title="Overall test metrics", ylabel="score", ylim=(0, 1.05))
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, overall_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")

    class_names = list(metrics["per_class"])
    positions = np.arange(len(class_names))
    width = 0.25
    for offset, key, label in ((-width, "precision", "Precision"), (0, "recall", "Recall"),
                               (width, "f1_score", "F1")):
        axes[1].bar(positions + offset, [metrics["per_class"][name][key] for name in class_names],
                    width=width, label=label)
    axes[1].set(title="Per-class metrics", ylabel="score", ylim=(0, 1.05), xticks=positions, xticklabels=class_names)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    figure.suptitle(
        f"Two-stage YOLOv26n test | conf>={metrics['confidence_threshold']:.3f}, "
        f"match IoU>={metrics['match_iou_threshold']:.2f}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    logger, _ = make_stage_logger(PROJECT_DIR, "evaluation", timestamp)
    if not 0 <= args.confidence <= 1 or not 0 <= args.match_iou <= 1:
        raise ValueError("confidence/match-iou는 0~1 사이여야 합니다.")
    prediction_dir = args.predictions.resolve()
    image_dir, label_dir, names = read_dataset(args.data.resolve())
    summaries = read_summary(prediction_dir / "image_summary.csv")
    all_predictions = read_predictions(prediction_dir / "detections.csv")
    output = (args.output_dir or METRICS_ROOT / f"two_stage_evaluation_{timestamp}").resolve()
    if output.exists():
        raise FileExistsError(f"평가 출력 폴더가 이미 있습니다: {output}")
    fp_dir = ERRORS_ROOT / "false_positive" / output.name
    fn_dir = ERRORS_ROOT / "false_negative" / output.name
    output.mkdir(parents=True)
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[INFO] prediction_run={prediction_dir}")
    logger.info(f"[INFO] data={args.data.resolve()}")
    logger.info(f"[INFO] confidence={args.confidence}, match_iou={args.match_iou}")
    logger.info(f"[INFO] output={output}")

    records = {}
    per_image = []
    match_rows = []
    totals = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    class_totals = {class_id: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []}
                    for class_id in names}
    all_ious = []
    for number, summary in enumerate(summaries, 1):
        image_name = summary["image_name"]
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"summary의 test 이미지가 없습니다: {image_path}")
        width, height = int(summary["image_width"]), int(summary["image_height"])
        ground_truths = read_ground_truth(label_dir / f"{image_path.stem}.txt", width, height, set(names))
        saved_predictions = all_predictions.get(image_name, [])
        unknown = {item["class_id"] for item in saved_predictions} - set(names)
        if unknown:
            raise ValueError(f"예측에 알 수 없는 class ID가 있습니다: {image_name}: {unknown}")
        predictions = [item for item in saved_predictions if item["confidence"] >= args.confidence]
        matches, false_negatives = match_boxes(predictions, ground_truths, args.match_iou)
        true_positives = sum(item["is_true_positive"] for item in matches)
        false_positives = len(predictions) - true_positives
        totals["gt"] += len(ground_truths)
        totals["pred"] += len(predictions)
        totals["tp"] += true_positives
        totals["fp"] += false_positives
        totals["fn"] += len(false_negatives)
        for class_id in names:
            class_ground_truths = sum(item["class_id"] == class_id for item in ground_truths)
            class_predictions = sum(item["class_id"] == class_id for item in predictions)
            class_tp = sum(item["is_true_positive"] and predictions[item["prediction_index"]]["class_id"] == class_id for item in matches)
            class_totals[class_id]["gt"] += class_ground_truths
            class_totals[class_id]["pred"] += class_predictions
            class_totals[class_id]["tp"] += class_tp
            class_totals[class_id]["fp"] += class_predictions - class_tp
            class_totals[class_id]["fn"] += class_ground_truths - class_tp
        for match in matches:
            prediction = predictions[match["prediction_index"]]
            if match["is_true_positive"]:
                all_ious.append(match["iou"])
                class_totals[prediction["class_id"]]["ious"].append(match["iou"])
            match_rows.append({"image_name": image_name, "class_id": prediction["class_id"],
                               "confidence": prediction["confidence"],
                               "result": "TP" if match["is_true_positive"] else "FP", "iou": match["iou"]})
        per_image.append({"image_name": image_name, "ground_truths": len(ground_truths),
                          "predictions": len(predictions), "true_positives": true_positives,
                          "false_positives": false_positives, "false_negatives": len(false_negatives),
                          "mean_matched_iou": float(np.mean([item["iou"] for item in matches if item["is_true_positive"]])) if true_positives else 0.0})
        records[image_name] = {"ground_truths": ground_truths, "predictions": saved_predictions}
        if false_positives or false_negatives:
            save_error_image(image_path, predictions, ground_truths, matches, false_negatives,
                             names, fp_dir, fn_dir)
        if number == 1 or number % 100 == 0:
            logger.info(f"[LOG] evaluation {number:,}/{len(summaries):,}")

    iou_thresholds = [round(value, 2) for value in np.arange(0.50, 0.96, 0.05)]
    per_class = {}
    figure, axis = plt.subplots(figsize=(8, 6))
    for class_id, class_name in names.items():
        values = class_totals[class_id]
        precision = safe_divide(values["tp"], values["tp"] + values["fp"])
        recall = safe_divide(values["tp"], values["tp"] + values["fn"])
        aps = {}
        for threshold in iou_thresholds:
            ap, recalls, precisions = average_precision(records, class_id, threshold)
            aps[f"AP@{threshold:.2f}"] = ap
            if math.isclose(threshold, 0.50):
                axis.plot(recalls, precisions, label=f"{class_name} AP50={ap:.3f}")
        per_class[class_name] = {
            "class_id": class_id, "ground_truth_count": values["gt"], "prediction_count": values["pred"],
            "true_positives": values["tp"], "false_positives": values["fp"], "false_negatives": values["fn"],
            "precision": precision, "recall": recall,
            "f1_score": safe_divide(2 * precision * recall, precision + recall),
            "detection_rate_percent": recall * 100,
            "mean_iou_of_true_positives": float(np.mean(values["ious"])) if values["ious"] else 0.0,
            "AP50": aps["AP@0.50"], "mAP50_95": float(np.mean(list(aps.values()))), **aps,
        }
    axis.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision", title="Test Precision-Recall Curves")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "pr_curve.png", dpi=150)
    plt.close(figure)

    precision = safe_divide(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_divide(totals["tp"], totals["tp"] + totals["fn"])
    valid_class_results = [result for result in per_class.values() if result["ground_truth_count"] > 0]
    inference_config_path = prediction_dir / "inference_config.json"
    inference_config = json.loads(inference_config_path.read_text(encoding="utf-8")) if inference_config_path.is_file() else {}
    metrics = {
        "confidence_threshold": args.confidence, "match_iou_threshold": args.match_iou,
        "inference_confidence_threshold": inference_config.get("confidence"),
        "vehicle_crop_confidence_threshold": inference_config.get("vehicle_crop_confidence"),
        "inference_iou_argument": inference_config.get("iou_argument"),
        "global_plate_nms_iou_threshold": inference_config.get("global_plate_nms_iou"),
        "limited_inference_run": inference_config.get("limited_run"),
        "test_images": len(summaries), "ground_truth_count": totals["gt"], "prediction_count": totals["pred"],
        "true_positives": totals["tp"], "false_positives": totals["fp"], "false_negatives": totals["fn"],
        "precision": precision, "recall": recall,
        "f1_score": safe_divide(2 * precision * recall, precision + recall),
        "detection_rate_percent": recall * 100,
        "mean_iou_of_true_positives": float(np.mean(all_ious)) if all_ious else 0.0,
        "mAP50": float(np.mean([result["AP50"] for result in valid_class_results])) if valid_class_results else 0.0,
        "mAP50_95": float(np.mean([result["mAP50_95"] for result in valid_class_results])) if valid_class_results else 0.0,
        "per_class": per_class,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_summary(metrics, output / "evaluation_summary.png")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if key != "per_class":
                writer.writerow([key, value])
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        rows = [{"class_name": name, **values} for name, values in per_class.items()]
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=per_image[0].keys())
        writer.writeheader()
        writer.writerows(per_image)
    with (output / "prediction_matches.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_name", "class_id", "confidence", "result", "iou"])
        writer.writeheader()
        writer.writerows(match_rows)
    log_lines = [
        f"[INFO] time_local={datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"[INFO] prediction_run={prediction_dir}",
        f"[INFO] confidence_threshold={args.confidence}",
        f"[INFO] match_iou_threshold={args.match_iou}",
        f"[INFO] inference_config={inference_config}",
        f"[RESULT] TP={totals['tp']}, FP={totals['fp']}, FN={totals['fn']}",
        f"[RESULT] Precision={precision:.6f}, Recall={recall:.6f}, F1={metrics['f1_score']:.6f}",
        f"[RESULT] mIoU(TP)={metrics['mean_iou_of_true_positives']:.6f}, "
        f"mAP50={metrics['mAP50']:.6f}, mAP50-95={metrics['mAP50_95']:.6f}",
    ]
    (output / "evaluation.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    logger.info(f"[RESULT] TP={totals['tp']}, FP={totals['fp']}, FN={totals['fn']}")
    logger.info(f"[RESULT] Precision={precision:.6f}, Recall={recall:.6f}, F1={metrics['f1_score']:.6f}")
    logger.info(f"[RESULT] mIoU(TP)={metrics['mean_iou_of_true_positives']:.6f}, "
                f"mAP50={metrics['mAP50']:.6f}, mAP50-95={metrics['mAP50_95']:.6f}")
    logger.info(f"[RESULT] output={output}")


if __name__ == "__main__":
    main()
