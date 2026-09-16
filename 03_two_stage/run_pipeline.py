"""03_two_stage 전체 파이프라인을 순서대로 실행하고 통합 로그를 저장한다."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from two_stage_utils import make_stage_logger  # noqa: E402


RAW_ROOT = PROJECT_DIR / "data" / "raw" / "roboflow_v2"
PROCESSED_ROOT = PROJECT_DIR / "data" / "preprocessed"
DATA_YAML = PROJECT_DIR / "configs" / "license_plate_vehicle.yaml"
STAGES = ("download", "preprocess", "check", "train", "inference", "evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roboflow → YOLOv26n 전체 파이프라인")
    parser.add_argument("--start-stage", choices=STAGES, default="download")
    parser.add_argument("--end-stage", choices=STAGES, default="evaluation")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-from-val-ratio",
        type=float,
        default=0.5,
        help="원본 test가 없을 때 valid 중 test로 사용할 비율",
    )
    parser.add_argument("--model", type=Path, help="train을 건너뛸 때 inference에 사용할 best.pt")
    parser.add_argument("--predictions", type=Path, help="inference를 건너뛸 때 평가할 결과 폴더")
    parser.add_argument("--inference-conf", type=float, default=0.001)
    parser.add_argument("--vehicle-crop-conf", type=float, default=0.25)
    parser.add_argument("--global-nms-iou", type=float, default=0.50)
    parser.add_argument("--evaluation-conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    return parser.parse_args()


def run(command: list[str], logger) -> None:
    logger.info("[PIPELINE] " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info(line.rstrip("\n"))
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def processed_is_ready() -> bool:
    return DATA_YAML.is_file() and all(
        (PROCESSED_ROOT / kind / split).is_dir()
        and any((PROCESSED_ROOT / kind / split).iterdir())
        for kind in ("images", "labels") for split in ("train", "val", "test")
    )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    logger, log_path = make_stage_logger(PROJECT_DIR, "pipeline", timestamp)
    start, end = STAGES.index(args.start_stage), STAGES.index(args.end_stage)
    if start > end:
        raise ValueError("start-stage는 end-stage보다 앞 단계여야 합니다.")
    selected = STAGES[start : end + 1]
    python = sys.executable
    trained_model = args.model.resolve() if args.model else None
    prediction_dir = args.predictions.resolve() if args.predictions else None
    logger.info(f"[INFO] selected_stages={list(selected)}")
    logger.info(f"[INFO] device={args.device}, epochs={args.epochs}, batch={args.batch}, workers={args.workers}")

    if "download" in selected:
        if (RAW_ROOT / "data.yaml").is_file():
            logger.info(f"[PIPELINE] download 재사용: {RAW_ROOT}")
        else:
            run([python, "03_two_stage/scripts/00_download/download_roboflow.py"], logger)

    if "preprocess" in selected:
        if processed_is_ready():
            logger.info(f"[PIPELINE] preprocess 재사용: {PROCESSED_ROOT}")
        else:
            run([
                python,
                "03_two_stage/scripts/01_preprocessing/preprocess_roboflow.py",
                "--resume",
                "--test-from-val-ratio",
                str(args.test_from_val_ratio),
                "--split-seed",
                str(args.seed),
            ], logger)

    if "check" in selected:
        run([python, "03_two_stage/scripts/02_training/train_two_stage.py", "--check-only",
             "--device", args.device, "--epochs", str(args.epochs), "--batch", str(args.batch),
             "--workers", str(args.workers), "--seed", str(args.seed)], logger)

    if "train" in selected:
        run_name = f"two_stage_yolo26n_{timestamp}"
        run([python, "03_two_stage/scripts/02_training/train_two_stage.py",
             "--device", args.device, "--epochs", str(args.epochs), "--batch", str(args.batch),
             "--workers", str(args.workers), "--seed", str(args.seed), "--run-name", run_name], logger)
        trained_model = PROJECT_DIR / "runs" / run_name / "weights" / "best.pt"

    if "inference" in selected:
        if trained_model is None:
            raise ValueError("train을 건너뛸 때는 --model /path/to/best.pt가 필요합니다.")
        prediction_dir = PROJECT_DIR / "results" / "predictions" / f"two_stage_inference_{timestamp}"
        run([python, "03_two_stage/scripts/03_inference/inference_two_stage.py",
             "--model", str(trained_model), "--device", args.device,
             "--conf", str(args.inference_conf),
             "--vehicle-crop-conf", str(args.vehicle_crop_conf),
             "--global-nms-iou", str(args.global_nms_iou),
             "--output-dir", str(prediction_dir)], logger)

    if "evaluation" in selected:
        if prediction_dir is None:
            raise ValueError("inference를 건너뛸 때는 --predictions /path/to/inference-run이 필요합니다.")
        run([python, "03_two_stage/scripts/04_evaluation/evaluate_two_stage.py",
             "--predictions", str(prediction_dir), "--confidence", str(args.evaluation_conf),
             "--match-iou", str(args.match_iou)], logger)

    logger.info("[PIPELINE] 완료")
    logger.info(f"[PIPELINE] combined_log={log_path}")
    if trained_model:
        logger.info(f"[PIPELINE] model={trained_model}")
    if prediction_dir:
        logger.info(f"[PIPELINE] predictions={prediction_dir}")


if __name__ == "__main__":
    main()
