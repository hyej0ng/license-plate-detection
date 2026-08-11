# Small-Object License Plate Detection

YOLOv26n을 기반으로 차량 번호판을 탐지하고, 원거리의 작은 객체 탐지 성능을 개선하는 방법을 비교하는 프로젝트입니다. 단순히 모델을 학습하는 데 그치지 않고 데이터 전처리, 좌표 변환 검증, 실험 재현성, 오류 사례 분석까지 하나의 탐지 파이프라인으로 정리하는 것을 목표로 합니다.

> 현재 상태: baseline 데이터 전처리 및 학습 코드 구현 완료, baseline 학습 진행 중. 4분할 탐색과 차량 중심 2단계 탐지는 후속 구현 단계입니다. 최종 test set 평가는 학습 완료 후 업데이트합니다.

## Research Questions

이 프로젝트는 다음 질문을 중심으로 진행합니다.

1. 기본 YOLOv26n은 번호판처럼 작은 객체를 어느 정도 탐지할 수 있는가?
2. 이미지를 4개 영역으로 나누어 탐색하면 원거리 번호판의 recall을 높일 수 있는가?
3. 차량을 먼저 찾고 차량 영역에서 번호판을 다시 탐지하는 2단계 방식은 전체 이미지 기반 탐지보다 효과적인가?
4. 각 방법의 성능 차이는 해상도, 객체 크기, 데이터 분포, 오탐지와 미탐지 사례로 어떻게 설명할 수 있는가?

## Methods

| Method | Description | Status |
| --- | --- | --- |
| Baseline | 640×640 letterbox 전처리 후 YOLOv26n으로 번호판 탐지 | Training |
| Quarter search | 이미지를 마진이 있는 4개 영역으로 분할하고, 원본 좌표 복원 후 NMS 적용 | Planned |
| Two-stage detection | 차량 탐지 → 차량 crop → 번호판 재탐지 → 원본 좌표 복원 | Planned |

모든 방법은 동일한 test set에서 mIoU, Precision, Recall, F1 Score, 탐지율, mAP@0.5, mAP@0.5:0.95를 기준으로 비교할 예정입니다. 평가 결과에는 confidence threshold와 NMS IoU threshold를 함께 기록합니다.

## Current Implementation

### Data preprocessing

- 이미지와 JSON 라벨을 파일 stem으로 매칭
- JSON의 번호판 bounding box를 `(xmin, ymin, xmax, ymax)`로 변환
- 중복되거나 유효하지 않은 bounding box 제거
- 원본 비율을 유지하는 640×640 letterbox resize
- padding과 scale을 반영해 bounding box 좌표 변환
- YOLO 형식 `(class, cx, cy, width, height)`으로 정규화
- 변환 전후 bounding box를 이미지에 다시 그려 시각적으로 검증

현재 전처리된 데이터 구성은 다음과 같습니다.

| Split | Images | Label files |
| --- | ---: | ---: |
| Train | 193,975 | 193,975 |
| Validation | 13,300 | 13,300 |
| Test | 485 | 485 |

### Baseline training

| Item | Setting |
| --- | --- |
| Model | YOLOv26n |
| Input size | 640×640 |
| Epochs | 50 |
| Optimizer | SGD |
| Initial learning rate | 0.003 |
| Batch size | Auto batch |
| Seed | 42 |
| Validation confidence | 0.001 |
| Validation NMS IoU | 0.70 |

학습 스크립트는 새 학습, 지정 가중치에서 새 optimizer로 시작하는 학습, optimizer와 epoch를 복구하는 정확한 resume를 구분합니다. 매 epoch마다 loss와 detection metrics를 기록하고 loss curve를 저장합니다.

## Repository Structure

```text
.
├── 01_baseline/
│   ├── configs/                   # 데이터셋 YAML 예시
│   └── scripts/
│       ├── 01_preprocessing/      # JSON → YOLO 전처리 및 시각 검증
│       ├── 02_training/           # YOLOv26n 학습 및 resume
│       ├── 03_inference/          # 구현 예정
│       └── 04_evaluation/         # 구현 예정
├── 02_quarter/                    # 4분할 탐색 실험
├── 03_two_stage/                  # 차량 중심 2단계 탐지 실험
├── common/                        # 공통 유틸리티와 로컬 모델 경로
├── comparison/                    # 공개 가능한 최종 표와 그래프
└── requirements.txt
```

데이터, Jupyter notebook, 학습 로그, 실험 run, 예측 이미지, 모델 가중치는 저장소에 포함하지 않습니다. 최종 결과 중 공개 가능한 표와 비식별 그래프만 `comparison/`에 선별하여 추가합니다.

## Setup

Python 3.11과 CUDA를 사용할 수 있는 환경을 권장합니다.

```bash
conda create -n license-plate-detection python=3.11 -y
conda activate license-plate-detection
python -m pip install -r requirements.txt
```

로컬 데이터 설정 파일을 만든 뒤 `path`를 전처리 데이터의 절대 경로로 변경합니다.

```bash
cp 01_baseline/configs/license_plate.example.yaml \
  01_baseline/configs/license_plate.yaml
```

사전학습 모델은 배포처의 이용 조건을 확인한 뒤 아래 로컬 경로에 준비합니다. `.pt` 파일은 Git에서 제외됩니다.

```text
common/weights/pretrained/yolo26n.pt
```

## Usage

원본 데이터 루트를 환경변수로 설정한 뒤, 먼저 한 장의 좌표 변환 결과를 검증합니다. 각 명령에 `--data-root`를 직접 전달해도 됩니다.

```bash
export LICENSE_PLATE_DATA_ROOT=/absolute/path/to/raw-dataset

python 01_baseline/scripts/01_preprocessing/preprocess_baseline.py \
  --mode visualize_raw --split train

python 01_baseline/scripts/01_preprocessing/preprocess_baseline.py \
  --mode preprocess_one --split train
```

시각 검증을 마친 후 전체 데이터를 전처리하고 baseline을 학습합니다.

```bash
python 01_baseline/scripts/01_preprocessing/preprocess_baseline.py \
  --mode preprocess_all

python 01_baseline/scripts/02_training/train_baseline.py
```

중단된 학습을 정확히 재개하거나 특정 가중치에서 새 optimizer로 학습할 수 있습니다.

```bash
python 01_baseline/scripts/02_training/train_baseline.py \
  --resume /absolute/path/to/last.pt

python 01_baseline/scripts/02_training/train_baseline.py \
  --weights /absolute/path/to/best.pt
```

## Results

최종 test set 결과가 준비되면 아래 표와 함께 PR curve, confusion matrix, FP/FN 사례 분석을 추가합니다. 학습 도중의 validation 수치를 최종 성능으로 보고하지 않습니다.

| Method | mIoU | Precision | Recall | F1 | Detection rate | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | — | — | — | — | — | — |
| Quarter search | — | — | — | — | — | — | — |
| Two-stage detection | — | — | — | — | — | — | — |

## Data and Model Policy

- 원본 데이터와 전처리 데이터는 용량, 라이선스, 개인정보 가능성을 고려해 공개하지 않습니다.
- 차량 및 번호판이 포함된 예측·오류 이미지는 공개 권한과 비식별 여부를 확인한 자료만 사용합니다.
- 학습 가중치와 YOLO 모델 파일은 저장소에 포함하지 않습니다.
- 데이터셋의 정확한 명칭, 출처, 라이선스는 공개 권한을 확인한 후 문서에 추가합니다.

## Project Context

AI 대학원 진학을 준비하며 수행한 학부연구생 프로젝트입니다. 작은 객체 탐지 문제를 중심으로 데이터 처리의 정확성, 실험 설계, 모델 간 공정한 비교, 정량·정성 평가 과정을 연구 기록으로 남기고 있습니다.
