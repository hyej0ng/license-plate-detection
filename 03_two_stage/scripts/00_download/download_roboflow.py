"""
Roboflow Universe의 license_plate_vehicle v2를 YOLOv8 형식으로 받는다.

실행방법:
cd /home/hyejong/landing_pjt
conda activate yolo

# 최초 한 번 Roboflow SDK를 설치
python -m pip install \
  -r 03_two_stage/requirements-download.txt

# Roboflow API key를 현재 터미널에 설정
export ROBOFLOW_API_KEY='본인의_private_API_key'

# 다운로드
python 03_two_stage/scripts/00_download/download_roboflow.py

# 이 스크립트는 다음 데이터를 요청함
workspace: parv217
project: license_plate_vehicle
version: 2
export format: yolov8

"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from two_stage_utils import make_stage_logger  # noqa: E402


DEFAULT_OUTPUT = PROJECT_DIR / "data" / "raw" / "roboflow_v2"
WORKSPACE = "parv217"
PROJECT = "license_plate_vehicle"
VERSION = 2
FORMAT = "yolov8" # YOLOv8 전용 모델 파일이 아니라, Ultralytics YOLO 계열에서 공통으로 사용하는 탐지 데이터 형식임. 라벨을 어떤 파일 형식으로 받을지 지정하는거


def redact_secret(message: str, secret: str) -> str:
    """SDK 예외 URL에 포함될 수 있는 API key를 traceback에서 제거한다."""
    return message.replace(secret, "***REDACTED***")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roboflow 공개 데이터셋 다운로드")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    logger, _ = make_stage_logger(PROJECT_DIR, "download", timestamp)
    output = args.output.resolve()
    if (output / "data.yaml").is_file():
        logger.info(f"[SKIP] 이미 다운로드되어 있습니다: {output}")
        return
    if output.exists():
        raise FileExistsError(
            f"출력 폴더가 이미 있습니다: {output}\n"
            "Roboflow SDK는 빈 폴더도 완료된 다운로드로 취급할 수 있습니다. "
            "기존 폴더를 보존하고 다른 --output을 사용하세요."
        )

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY가 없습니다. Roboflow 계정에서 private API key를 확인한 뒤 "
            "`export ROBOFLOW_API_KEY=...`로 현재 셸에만 설정하세요."
        )
    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise RuntimeError(
            "Roboflow SDK가 없습니다. 먼저 `python -m pip install -r "
            "03_two_stage/requirements-download.txt`를 실행하세요."
        ) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[INFO] dataset={WORKSPACE}/{PROJECT}/{VERSION}, format={FORMAT}")
    logger.info(f"[INFO] output={output}")
    try:
        client = Roboflow(api_key=api_key)
        project = client.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(VERSION).download(
            model_format=FORMAT,
            location=str(output),
            overwrite=False,
        )
    except Exception as error:
        safe_message = redact_secret(str(error), api_key)
        network_markers = (
            "Network is unreachable",
            "Failed to establish a new connection",
            "Name or service not known",
            "Temporary failure in name resolution",
            "Could not resolve host",
        )
        if any(marker in safe_message for marker in network_markers):
            raise RuntimeError(
                "Roboflow API에 연결하지 못했습니다. 이 서버의 외부 인터넷/DNS/방화벽을 "
                "확인하세요. 인터넷 연결이 허용되지 않는 서버라면 다른 PC에서 YOLOv8 "
                "형식 ZIP을 내려받아 03_two_stage/data/raw/roboflow_v2에 옮기세요."
            ) from None
        raise RuntimeError(f"Roboflow 다운로드 실패: {safe_message}") from None
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"다운로드는 끝났지만 data.yaml이 없습니다: {data_yaml}")
    logger.info(f"[RESULT] downloaded: {dataset.location}")
    logger.info(f"[RESULT] source_yaml: {data_yaml}")


if __name__ == "__main__":
    main()
