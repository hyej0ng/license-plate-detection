"""01_baseline/02_quarter 모델의 외부 test 번호판 예측을 평가한다.

지원하는 예측 형식:
- Quarter inference 폴더: detections.csv
- Ultralytics predict 폴더: labels/*.txt (save_conf=True 필요)

정답은 03_two_stage의 test YOLO 라벨 중 plate class(기본값 0)만 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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

from two_stage_utils import IMAGE_EXTENSIONS, image_index, iou, make_stage_logger  # noqa: E402


DEFAULT_DATA = PROJECT_DIR / "configs" / "license_plate_vehicle.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="baseline/quarter 단일 번호판 모델의 03 외부 test 결과 평가"
    )
    parser.add_argument("--predictions", type=Path, required=True, help="inference 결과 폴더")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="03 test 데이터 YAML")
    parser.add_argument("--confidence", "--conf", dest="confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--plate-class-id", type=int, default=0)
    parser.add_argument("--model-name", help="그래프에 표시할 모델 이름(기본: 결과 폴더명)")
    parser.add_argument("--output-dir", type=Path, help="기본: <predictions>/evaluation_<timestamp>")
    return parser.parse_args()


def validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}은 0~1 사이여야 합니다: {value}")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def normalized_to_xyxy(values: tuple[float, float, float, float], width: int, height: int):
    cx, cy, box_width, box_height = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"bbox에 유한하지 않은 값이 있습니다: {values}")
    if not all(0.0 <= value <= 1.0 for value in values) or box_width <= 0 or box_height <= 0:
        raise ValueError(f"normalized bbox 범위가 잘못되었습니다: {values}")
    return (
        (cx - box_width / 2) * width,
        (cy - box_height / 2) * height,
        (cx + box_width / 2) * width,
        (cy + box_height / 2) * height,
    )


def read_dataset(data_yaml: Path) -> tuple[dict[str, Path], Path, dict[int, str]]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"데이터 YAML이 없습니다: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    test_value = Path(config["test"])
    image_dir = test_value if test_value.is_absolute() else root / test_value
    label_dir = root / "labels" / test_value.name
    names_value = config.get("names", {})
    names = (
        {int(key): str(value) for key, value in names_value.items()}
        if isinstance(names_value, dict)
        else dict(enumerate(names_value))
    )
    images = image_index(image_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"test label 폴더가 없습니다: {label_dir}")
    return images, label_dir, names


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {path}")
    height, width = image.shape[:2]
    return width, height


def read_ground_truth(path: Path, width: int, height: int, plate_class_id: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"정답 라벨이 없습니다: {path}")
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"정답 YOLO 라벨 형식 오류: {path}:{line_number}")
        class_id = int(parts[0])
        if class_id != plate_class_id:
            continue
        values = tuple(map(float, parts[1:5]))
        boxes.append({"class_id": plate_class_id, "box": normalized_to_xyxy(values, width, height)})
    return boxes


def resolve_prediction_format(prediction_dir: Path) -> str:
    if (prediction_dir / "detections.csv").is_file():
        return "detections_csv"
    if (prediction_dir / "labels").is_dir():
        return "yolo_txt"
    raise FileNotFoundError(
        "예측 형식을 찾지 못했습니다. Quarter는 detections.csv, Baseline은 "
        f"labels/*.txt(save_conf=True)가 필요합니다: {prediction_dir}"
    )


def read_csv_predictions(path: Path, plate_class_id: int) -> tuple[dict[str, list[dict]], int]:
    predictions: dict[str, list[dict]] = defaultdict(list)
    ignored_classes = 0
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_name", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"detections.csv 필수 열이 없습니다: {sorted(required)}")
        for line_number, row in enumerate(reader, 2):
            class_id = int(row["class_id"])
            if class_id != plate_class_id:
                ignored_classes += 1
                continue
            confidence = float(row["confidence"])
            box = tuple(float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
            if not 0.0 <= confidence <= 1.0 or not all(math.isfinite(value) for value in box):
                raise ValueError(f"예측 값 오류: {path}:{line_number}")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"예측 bbox 크기 오류: {path}:{line_number}: {box}")
            predictions[row["image_name"]].append(
                {"class_id": plate_class_id, "confidence": confidence, "box": box}
            )
    return predictions, ignored_classes


def read_yolo_predictions(
    prediction_dir: Path,
    images: dict[str, Path],
    sizes: dict[str, tuple[int, int]],
    plate_class_id: int,
) -> tuple[dict[str, list[dict]], int]:
    predictions: dict[str, list[dict]] = defaultdict(list)
    ignored_classes = 0
    labels_dir = prediction_dir / "labels"
    unknown_labels = sorted(
        path.name for path in labels_dir.glob("*.txt") if path.stem not in images
    )
    if unknown_labels:
        raise ValueError(f"test 이미지와 대응하지 않는 예측 TXT가 있습니다: {unknown_labels[:3]}")
    for stem, path in images.items():
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.is_file():
            continue  # Ultralytics는 detection이 없는 이미지의 TXT를 만들지 않는다.
        width, height = sizes[stem]
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"예측 TXT는 class cx cy w h confidence 6열이어야 합니다: "
                    f"{label_path}:{line_number}. inference에서 save_conf=True를 사용하세요."
                )
            class_id = int(parts[0])
            if class_id != plate_class_id:
                ignored_classes += 1
                continue
            confidence = float(parts[5])
            validate_ratio("prediction confidence", confidence)
            values = tuple(map(float, parts[1:5]))
            predictions[path.name].append(
                {
                    "class_id": plate_class_id,
                    "confidence": confidence,
                    "box": normalized_to_xyxy(values, width, height),
                }
            )
    return predictions, ignored_classes


def read_inference_metadata(prediction_dir: Path) -> dict:
    json_path = prediction_dir / "inference_config.json"
    if json_path.is_file():
        return json.loads(json_path.read_text(encoding="utf-8"))
    yaml_path = prediction_dir / "args.yaml"
    if yaml_path.is_file():
        value = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    return {}


def inference_confidence(metadata: dict) -> float | None:
    for key in ("confidence_threshold", "confidence", "conf"):
        value = metadata.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def match_boxes(predictions: list[dict], ground_truths: list[dict], threshold: float):
    used_ground_truths: set[int] = set()
    matches = []
    for prediction_index, prediction in enumerate(predictions):
        best_index, best_iou = None, 0.0
        for ground_truth_index, ground_truth in enumerate(ground_truths):
            if ground_truth_index in used_ground_truths:
                continue
            score = iou(prediction["box"], ground_truth["box"])
            if score > best_iou:
                best_index, best_iou = ground_truth_index, score
        is_true_positive = best_index is not None and best_iou >= threshold
        if is_true_positive:
            used_ground_truths.add(best_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": best_index if is_true_positive else None,
                "is_true_positive": is_true_positive,
                "iou": best_iou if is_true_positive else 0.0,
            }
        )
    false_negatives = [
        index for index in range(len(ground_truths)) if index not in used_ground_truths
    ]
    return matches, false_negatives


def calculate_ap(records: dict[str, dict], threshold: float, total_ground_truths: int):
    ranked = []
    for image_name, record in records.items():
        for prediction in record["predictions"]:
            ranked.append((prediction["confidence"], image_name, prediction))
    ranked.sort(key=lambda item: item[0], reverse=True)
    used = {image_name: set() for image_name in records}
    true_positives, false_positives = [], []
    for _, image_name, prediction in ranked:
        best_index, best_iou = None, 0.0
        for ground_truth_index, ground_truth in enumerate(records[image_name]["ground_truths"]):
            if ground_truth_index in used[image_name]:
                continue
            score = iou(prediction["box"], ground_truth["box"])
            if score > best_iou:
                best_index, best_iou = ground_truth_index, score
        is_true_positive = best_index is not None and best_iou >= threshold
        if is_true_positive:
            used[image_name].add(best_index)
        true_positives.append(int(is_true_positive))
        false_positives.append(int(not is_true_positive))
    if total_ground_truths == 0 or not ranked:
        return 0.0, np.array([0.0]), np.array([0.0])
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recalls = cumulative_tp / total_ground_truths
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall_points = np.linspace(0.0, 1.0, 101)
    ap = np.mean(
        [precisions[recalls >= point].max() if np.any(recalls >= point) else 0.0 for point in recall_points]
    )
    return float(ap), recalls, precisions


def draw_box(image: np.ndarray, box: tuple[float, ...], color: tuple[int, ...], text: str) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, color, 1, cv2.LINE_AA,
    )


def save_error_image(
    image_path: Path,
    predictions: list[dict],
    ground_truths: list[dict],
    matches: list[dict],
    false_negatives: list[int],
    fp_dir: Path,
    fn_dir: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"시각화 이미지를 읽지 못했습니다: {image_path}")
    for index, ground_truth in enumerate(ground_truths):
        is_false_negative = index in false_negatives
        draw_box(
            image,
            ground_truth["box"],
            (0, 165, 255) if is_false_negative else (0, 255, 0),
            "FN plate" if is_false_negative else "GT plate",
        )
    has_false_positive = False
    for match in matches:
        prediction = predictions[match["prediction_index"]]
        if match["is_true_positive"]:
            draw_box(
                image, prediction["box"], (255, 0, 0),
                f"TP {prediction['confidence']:.2f} IoU {match['iou']:.2f}",
            )
        else:
            has_false_positive = True
            draw_box(image, prediction["box"], (0, 0, 255), f"FP {prediction['confidence']:.2f}")
    if has_false_positive and not cv2.imwrite(str(fp_dir / image_path.name), image):
        raise IOError(f"FP 이미지 저장 실패: {image_path.name}")
    if false_negatives and not cv2.imwrite(str(fn_dir / image_path.name), image):
        raise IOError(f"FN 이미지 저장 실패: {image_path.name}")


def save_pr_curve(recalls: np.ndarray, precisions: np.ndarray, metrics: dict, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(recalls, precisions, linewidth=2, label=f"plate AP50={metrics['AP50_from_saved_predictions']:.3f}")
    axis.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision")
    axis.set_title("Plate Precision-Recall curve (saved predictions)")
    axis.grid(alpha=0.3)
    axis.legend()
    if metrics["ap_is_truncated"]:
        axis.text(
            0.01, 0.02, "Caution: predictions were saved after confidence filtering; AP is truncated.",
            transform=axis.transAxes, fontsize=9, color="#B22222",
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_confusion_matrix(metrics: dict, output_path: Path) -> None:
    matrix = np.array(
        [[metrics["true_positives"], metrics["false_negatives"]],
         [metrics["false_positives"], 0]],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["plate", "background"], yticklabels=["plate", "background"],
        xlabel="Predicted", ylabel="Actual",
        title="Object-detection confusion matrix",
    )
    annotations = [
        [f"TP\n{metrics['true_positives']}", f"FN\n{metrics['false_negatives']}"],
        [f"FP\n{metrics['false_positives']}", "N/A\n(not counted)"],
    ]
    maximum = max(float(matrix.max()), 1.0)
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > maximum / 2 else "black"
            axis.text(column, row, annotations[row][column], ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_summary(metrics: dict, recalls: np.ndarray, precisions: np.ndarray, output_path: Path) -> None:
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    metric_axis = figure.add_subplot(grid[0, 0])
    metric_names = ["Precision", "Recall", "F1", "mIoU(TP)", "AP50", "mAP50-95"]
    metric_values = [
        metrics["precision"], metrics["recall"], metrics["f1_score"],
        metrics["mean_iou_of_true_positives"], metrics["AP50_from_saved_predictions"],
        metrics["mAP50_95_from_saved_predictions"],
    ]
    bars = metric_axis.bar(metric_names, metric_values, color="#2878B5")
    metric_axis.set(title="Plate detection metrics", ylabel="score", ylim=(0, 1.05))
    metric_axis.tick_params(axis="x", rotation=25)
    metric_axis.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, metric_values):
        metric_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")

    count_axis = figure.add_subplot(grid[0, 1])
    count_names = ["GT", "Pred", "TP", "FP", "FN"]
    count_values = [
        metrics["ground_truth_count"], metrics["prediction_count"], metrics["true_positives"],
        metrics["false_positives"], metrics["false_negatives"],
    ]
    count_bars = count_axis.bar(count_names, count_values, color=["#666666", "#2878B5", "#2CA02C", "#D62728", "#FF8C00"])
    count_axis.set(title="Detection counts", ylabel="boxes")
    count_axis.grid(axis="y", alpha=0.3)
    for bar, value in zip(count_bars, count_values):
        count_axis.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")

    pr_axis = figure.add_subplot(grid[1, :])
    pr_axis.plot(recalls, precisions, linewidth=2, color="#2878B5")
    pr_axis.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision")
    pr_axis.set_title(f"PR curve | AP50={metrics['AP50_from_saved_predictions']:.3f}")
    pr_axis.grid(alpha=0.3)
    caveat = "AP/mAP uses every prediction available in the saved result."
    if metrics["ap_is_truncated"]:
        caveat += " Saved predictions were confidence-filtered, so AP/mAP is truncated."
    figure.suptitle(
        f"{metrics['model_name']} on 03 external test (plate only)\n"
        f"images={metrics['test_images']}, conf>={metrics['confidence_threshold']:.3f}, "
        f"match IoU>={metrics['match_iou_threshold']:.2f}",
        fontsize=15,
    )
    figure.text(0.5, 0.01, caveat, ha="center", fontsize=9, color="dimgray")
    figure.tight_layout(rect=(0, 0.03, 1, 0.93))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_dict_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"CSV에 저장할 행이 없습니다: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_ratio("confidence", args.confidence)
    validate_ratio("match_iou", args.match_iou)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    logger, central_log_path = make_stage_logger(PROJECT_DIR, "external_evaluation", timestamp)

    prediction_dir = args.predictions.resolve()
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"inference 결과 폴더가 없습니다: {prediction_dir}")
    output_dir = (args.output_dir or prediction_dir / f"evaluation_{timestamp}").resolve()
    if output_dir.exists():
        raise FileExistsError(f"평가 출력 폴더가 이미 있습니다: {output_dir}")
    output_dir.mkdir(parents=True)
    fp_dir = output_dir / "false_positive"
    fn_dir = output_dir / "false_negative"
    fp_dir.mkdir()
    fn_dir.mkdir()
    output_handler = logging.FileHandler(output_dir / "evaluation.log", encoding="utf-8")
    output_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(output_handler)

    images, label_dir, names = read_dataset(args.data.resolve())
    if args.plate_class_id not in names:
        raise ValueError(f"YAML names에 plate class ID {args.plate_class_id}가 없습니다: {names}")
    sizes = {stem: image_size(path) for stem, path in images.items()}
    prediction_format = resolve_prediction_format(prediction_dir)
    if prediction_format == "detections_csv":
        all_predictions, ignored_predictions = read_csv_predictions(
            prediction_dir / "detections.csv", args.plate_class_id
        )
    else:
        all_predictions, ignored_predictions = read_yolo_predictions(
            prediction_dir, images, sizes, args.plate_class_id
        )
    unknown_images = sorted(set(all_predictions) - {path.name for path in images.values()})
    if unknown_images:
        raise ValueError(f"03 test에 없는 이미지 예측이 있습니다: {unknown_images[:3]}")

    metadata = read_inference_metadata(prediction_dir)
    saved_inference_confidence = inference_confidence(metadata)
    model_name = args.model_name or prediction_dir.name
    model_value = metadata.get("model")
    if args.model_name is None and model_value:
        model_name = f"{prediction_dir.name} ({Path(str(model_value)).stem})"

    logger.info(f"[INFO] prediction_run={prediction_dir}")
    logger.info(f"[INFO] prediction_format={prediction_format}")
    logger.info(f"[INFO] data={args.data.resolve()}")
    logger.info(f"[INFO] plate_class={args.plate_class_id}:{names[args.plate_class_id]}")
    logger.info(f"[INFO] confidence={args.confidence}, match_iou={args.match_iou}")
    logger.info(f"[INFO] output={output_dir}")
    if ignored_predictions:
        logger.info(f"[INFO] ignored_non_plate_predictions={ignored_predictions}")

    totals = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    matched_ious: list[float] = []
    saved_confidences: list[float] = []
    per_image_rows = []
    match_rows = []
    records = {}
    for number, stem in enumerate(sorted(images), 1):
        image_path = images[stem]
        width, height = sizes[stem]
        ground_truths = read_ground_truth(
            label_dir / f"{stem}.txt", width, height, args.plate_class_id
        )
        saved_predictions = sorted(
            all_predictions.get(image_path.name, []),
            key=lambda item: item["confidence"],
            reverse=True,
        )
        saved_confidences.extend(item["confidence"] for item in saved_predictions)
        predictions = [item for item in saved_predictions if item["confidence"] >= args.confidence]
        matches, false_negatives = match_boxes(predictions, ground_truths, args.match_iou)
        true_positives = sum(item["is_true_positive"] for item in matches)
        false_positives = len(predictions) - true_positives
        false_negative_count = len(false_negatives)
        image_ious = [item["iou"] for item in matches if item["is_true_positive"]]
        matched_ious.extend(image_ious)
        totals["gt"] += len(ground_truths)
        totals["pred"] += len(predictions)
        totals["tp"] += true_positives
        totals["fp"] += false_positives
        totals["fn"] += false_negative_count
        per_image_rows.append(
            {
                "image_name": image_path.name,
                "ground_truths": len(ground_truths),
                "predictions": len(predictions),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negative_count,
                "precision": safe_divide(true_positives, true_positives + false_positives),
                "recall": safe_divide(true_positives, true_positives + false_negative_count),
                "f1_score": safe_divide(2 * true_positives, 2 * true_positives + false_positives + false_negative_count),
                "mean_matched_iou": float(np.mean(image_ious)) if image_ious else 0.0,
            }
        )
        for match in matches:
            prediction = predictions[match["prediction_index"]]
            match_rows.append(
                {
                    "image_name": image_path.name,
                    "result": "TP" if match["is_true_positive"] else "FP",
                    "prediction_index": match["prediction_index"] + 1,
                    "ground_truth_index": (
                        match["ground_truth_index"] + 1
                        if match["ground_truth_index"] is not None else ""
                    ),
                    "confidence": prediction["confidence"],
                    "iou": match["iou"],
                }
            )
        for ground_truth_index in false_negatives:
            match_rows.append(
                {
                    "image_name": image_path.name,
                    "result": "FN",
                    "prediction_index": "",
                    "ground_truth_index": ground_truth_index + 1,
                    "confidence": "",
                    "iou": 0.0,
                }
            )
        records[image_path.name] = {
            "ground_truths": ground_truths,
            "predictions": saved_predictions,
        }
        if false_positives or false_negative_count:
            save_error_image(
                image_path, predictions, ground_truths, matches, false_negatives, fp_dir, fn_dir
            )
        if number == 1 or number % 50 == 0 or number == len(images):
            logger.info(f"[LOG] evaluated {number:04d}/{len(images):04d}")

    if totals["gt"] != totals["tp"] + totals["fn"]:
        raise AssertionError("GT != TP + FN")
    if totals["pred"] != totals["tp"] + totals["fp"]:
        raise AssertionError("Pred != TP + FP")

    average_precisions = {}
    ap50_recalls, ap50_precisions = np.array([0.0]), np.array([0.0])
    for threshold in np.arange(0.50, 0.96, 0.05):
        ap, recalls, precisions = calculate_ap(records, float(threshold), totals["gt"])
        average_precisions[f"AP@{threshold:.2f}"] = ap
        if np.isclose(threshold, 0.50):
            ap50_recalls, ap50_precisions = recalls, precisions

    precision = safe_divide(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_divide(totals["tp"], totals["tp"] + totals["fn"])
    minimum_saved_confidence = min(saved_confidences) if saved_confidences else None
    effective_save_confidence = saved_inference_confidence
    if effective_save_confidence is None:
        effective_save_confidence = minimum_saved_confidence
    ap_is_truncated = bool(
        effective_save_confidence is not None and effective_save_confidence > (0.01 if saved_inference_confidence is None else 0.001 + 1e-9)
    )
    metrics = {
        "model_name": model_name,
        "prediction_run": str(prediction_dir),
        "prediction_format": prediction_format,
        "data_yaml": str(args.data.resolve()),
        "plate_class_id": args.plate_class_id,
        "plate_class_name": names[args.plate_class_id],
        "confidence_threshold": args.confidence,
        "match_iou_threshold": args.match_iou,
        "inference_confidence_threshold": saved_inference_confidence,
        "minimum_confidence_in_saved_predictions": minimum_saved_confidence,
        "ap_is_truncated": ap_is_truncated,
        "test_images": len(images),
        "ground_truth_count": totals["gt"],
        "prediction_count": totals["pred"],
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "precision": precision,
        "recall": recall,
        "f1_score": safe_divide(2 * precision * recall, precision + recall),
        "detection_rate_percent": recall * 100,
        "mean_iou_of_true_positives": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "AP50_from_saved_predictions": average_precisions["AP@0.50"],
        "mAP50_95_from_saved_predictions": float(np.mean(list(average_precisions.values()))),
        **average_precisions,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scalar_rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    write_dict_rows(output_dir / "metrics.csv", scalar_rows)
    write_dict_rows(output_dir / "per_image_metrics.csv", per_image_rows)
    write_dict_rows(output_dir / "prediction_matches.csv", match_rows)
    summary_row = {
        "model_name": model_name,
        "test_images": metrics["test_images"],
        "confidence_threshold": metrics["confidence_threshold"],
        "match_iou_threshold": metrics["match_iou_threshold"],
        "ground_truth_count": metrics["ground_truth_count"],
        "prediction_count": metrics["prediction_count"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "detection_rate_percent": metrics["detection_rate_percent"],
        "mean_iou_of_true_positives": metrics["mean_iou_of_true_positives"],
        "AP50_from_saved_predictions": metrics["AP50_from_saved_predictions"],
        "mAP50_95_from_saved_predictions": metrics["mAP50_95_from_saved_predictions"],
        "ap_is_truncated": metrics["ap_is_truncated"],
    }
    write_dict_rows(output_dir / "evaluation_summary_table.csv", [summary_row])
    save_pr_curve(ap50_recalls, ap50_precisions, metrics, output_dir / "pr_curve.png")
    save_confusion_matrix(metrics, output_dir / "confusion_matrix.png")
    save_summary(metrics, ap50_recalls, ap50_precisions, output_dir / "evaluation_summary.png")

    logger.info(f"[RESULT] TP={totals['tp']}, FP={totals['fp']}, FN={totals['fn']}")
    logger.info(
        f"[RESULT] Precision={metrics['precision']:.6f}, Recall={metrics['recall']:.6f}, "
        f"F1={metrics['f1_score']:.6f}, mIoU(TP)={metrics['mean_iou_of_true_positives']:.6f}"
    )
    logger.info(
        f"[RESULT] AP50(saved)={metrics['AP50_from_saved_predictions']:.6f}, "
        f"mAP50-95(saved)={metrics['mAP50_95_from_saved_predictions']:.6f}"
    )
    if ap_is_truncated:
        logger.warning(
            "[WARNING] inference에서 낮은 confidence 예측이 이미 제거되었습니다. "
            "AP/mAP는 저장된 예측 범위에서만 계산된 truncated 값입니다. "
            "정확한 PR/AP 비교는 inference를 conf=0.001로 다시 저장하세요."
        )
    logger.info(f"[RESULT] false_positive_images={fp_dir} ({len(list(fp_dir.iterdir()))}장)")
    logger.info(f"[RESULT] false_negative_images={fn_dir} ({len(list(fn_dir.iterdir()))}장)")
    logger.info(f"[RESULT] output={output_dir}")
    logger.info(f"[INFO] central_log={central_log_path}")


if __name__ == "__main__":
    main()
