# Roboflow Vehicle + License Plate / YOLOv26n

Roboflow Universe의 [`parv217/license_plate_vehicle` version 2](https://universe.roboflow.com/parv217/license_plate_vehicle/dataset/2)를 내려받아 `01_baseline`과 같은 순서로 전처리, YOLOv26n 학습+validation, test inference, 평가하는 프로젝트입니다.

데이터셋 페이지 기준으로 객체 탐지 이미지 6,272장, 클래스 2개(`vehicle`, `license plate`), 라이선스는 CC BY 4.0입니다. 원본 class 순서와 관계없이 전처리 결과는 다음으로 통일합니다.

```text
0: plate
1: car
```

## 처리 흐름

```text
Roboflow YOLOv8 export
  -> 원본 data.yaml과 split/라벨 검증
  -> class ID 재매핑
  -> 640x640 letterbox + bbox 좌표 재계산
  -> raw/processed bbox 시각 검증
  -> YOLOv26n train + 매 epoch validation
  -> test 전체 이미지에서 car 탐지
  -> confidence 0.25 이상 car crop을 batch로 재입력해 plate 탐지
  -> plate 좌표를 원본으로 복원하고 global NMS
  -> confidence 필터 + class-aware IoU 1:1 매칭 평가
  -> 전체/클래스별 P, R, F1, mIoU, AP50, mAP50-95 및 FP/FN 저장
```

원본 다운로드는 `03_two_stage/data/raw`, 전처리 결과는 `03_two_stage/data/preprocessed`에 저장합니다. 클린업으로 active raw에서 제외한 파일은 `03_two_stage/data/quarantine`에 복구 가능하게 보관합니다.

## 1. 환경 준비와 다운로드

저장소 루트에서 기존 학습 환경을 활성화합니다.

```bash
conda activate yolo
python -m pip install -r 03_two_stage/requirements-download.txt
```

Roboflow에 로그인한 뒤 계정 설정에서 private API key를 확인하고 현재 셸의 환경변수로만 설정합니다. 키를 `.py`, `.env`, README 또는 명령 인자에 저장하지 마세요.

```bash
export ROBOFLOW_API_KEY='본인의_private_API_key'
python 03_two_stage/scripts/00_download/download_roboflow.py
```

`[Errno 101] Network is unreachable`가 나오면 API key나 데이터셋 경로 문제가 아니라
서버의 외부 인터넷 연결이 차단된 상태입니다. 서버 관리자에게
`https://api.roboflow.com`과 Roboflow 다운로드 도메인의 HTTPS(443) 접근을 요청하거나,
인터넷이 되는 PC에서 YOLOv8 형식 ZIP을 내려받아 아래 위치로 옮기세요. 터미널 로그에
API key가 노출됐다면 해당 key는 즉시 폐기하고 새로 발급합니다.

브라우저에서 `YOLOv8` 형식으로 직접 다운로드했다면 압축을 아래처럼 풀어 `data.yaml`이 바로 보이게 하면 다운로드 스크립트를 생략할 수 있습니다.

```text
03_two_stage/data/raw/roboflow_v2/
├── data.yaml
├── train/
├── valid/
└── test/       # export에 있는 경우; version 2에는 없음
```

## 2. 전처리와 좌표 검증

먼저 별도 임시 출력 위치에 split당 5장만 처리해 bbox 시각화를 확인합니다.

```bash
python 03_two_stage/scripts/01_preprocessing/preprocess_roboflow.py \
  --output-root /tmp/license_plate_vehicle_smoke \
  --config-output /tmp/license_plate_vehicle_smoke.yaml \
  --max-samples 5 \
  --visualize-count 5
```

다음 두 폴더의 같은 이미지를 비교합니다.

```text
/tmp/license_plate_vehicle_smoke/visualization/raw_bbox/
/tmp/license_plate_vehicle_smoke/visualization/processed_bbox/
```

좌표가 맞으면 전체 데이터를 처리합니다.

```bash
python 03_two_stage/scripts/01_preprocessing/preprocess_roboflow.py \
  --image-size 640 \
  --visualize-count 5
```

이 단계가 실제 절대 경로를 담은 `03_two_stage/configs/license_plate_vehicle.yaml`을 생성합니다. 기존 결과는 안전을 위해 자동으로 덮어쓰지 않습니다.
원본에 너비 또는 높이가 0인 퇴화 bbox가 있으면 해당 박스만 제외하고 파일·라인·좌표를
`logs/preprocess_<timestamp>.log`과 `preprocessing_stats_<timestamp>.json`에 기록합니다.

중단된 전처리를 이어서 실행할 때는 `--resume`을 붙입니다. 이미 완료된 640×640
이미지와 라벨 쌍을 검증해 건너뛰고, 누락된 쌍부터 다시 처리합니다.

```bash
python 03_two_stage/scripts/01_preprocessing/preprocess_roboflow.py \
  --image-size 640 \
  --visualize-count 5 \
  --resume
```

현재 Roboflow version 2 export에는 별도 `test` 폴더가 없으므로, 전처리 스크립트가
클린업 후 `valid` 205장을 고정 시드 42로 val 103장/test 102장으로 나눕니다.
이 val/test 분할은 active raw 파일을 추가로 수정하지 않으며, 비율과 시드는 `--test-from-val-ratio`, `--split-seed`로
바꿀 수 있습니다.

다운로드 직후에는 train 18,192장과 valid 208장이었습니다. 전체 검사 후
완전 중복, train–valid 원본 source 누수, 퇴화 bbox를 정리해 active raw는
train 15,648장과 valid 205장입니다. 격리 내역과 복구 manifest는
`03_two_stage/data/quarantine/20260916_dataset_audit/` 아래에 있습니다.

## 3. 데이터 연결 검사와 학습+validation

```bash
python 03_two_stage/scripts/02_training/train_two_stage.py \
  --check-only \
  --device cpu
```

GPU 번호와 batch를 환경에 맞춰 학습합니다. `--batch -1`은 Ultralytics 자동 batch입니다.

```bash
python 03_two_stage/scripts/02_training/train_two_stage.py \
  --device 0 \
  --epochs 50 \
  --batch -1 \
  --workers 8 \
  --patience 0 \
  --seed 42
```

매 epoch validation이 실행되며 `results.csv`, `loss_curve.png`, PR curve, confusion matrix와 `weights/best.pt`가 `03_two_stage/runs/<run-name>/`에 저장됩니다. `loss_curve.png`는 위쪽에 train/validation 총 loss(`box + cls + DFL`), 아래쪽에 각 loss 구성요소를 저장하며 epoch마다 갱신됩니다.

중단 시 optimizer와 epoch까지 복구하려면 정상 종료로 optimizer가 제거되기 전의 `last.pt`를 사용합니다.

```bash
python 03_two_stage/scripts/02_training/train_two_stage.py \
  --resume 03_two_stage/runs/<run-name>/weights/last.pt \
  --device 0 \
  --epochs 50
```

## 4. Test inference

AP/PR curve 계산에는 낮은 점수의 예측도 필요하므로 inference는 `--conf 0.001`로 저장하고, 최종 운영 threshold `0.25`는 평가 단계에서 적용합니다.

```bash
python 03_two_stage/scripts/03_inference/inference_two_stage.py \
  --model 03_two_stage/runs/<run-name>/weights/best.pt \
  --source 03_two_stage/data/preprocessed/images/test \
  --conf 0.001 \
  --vehicle-crop-conf 0.25 \
  --iou 0.70 \
  --global-nms-iou 0.50 \
  --device 0
```

1단계에서는 전체 이미지의 `car` 예측을 저장하고, 그중 `--vehicle-crop-conf` 이상인 차량 영역만 crop합니다. 2단계는 crop에서 `plate`만 탐지하고 좌표를 원본 이미지로 복원한 뒤 여러 차량 crop에서 중복된 plate를 global NMS로 통합합니다. 예측 bbox 이미지, YOLO 예측 TXT, stage/부모 차량이 기록된 `detections.csv`, 모든 threshold가 기록된 `inference_config.json`이 `03_two_stage/results/predictions/<inference-run>/`에 저장됩니다.

## 5. Test evaluation

```bash
python 03_two_stage/scripts/04_evaluation/evaluate_two_stage.py \
  --predictions 03_two_stage/results/predictions/<inference-run> \
  --confidence 0.25 \
  --match-iou 0.50
```

전체 및 클래스별 Precision, Recall, F1, 탐지율, TP의 mIoU, AP50, mAP50-95와 PR curve가 `results/metrics`에 저장됩니다. FP/FN 시각화는 `results/errors`에 저장됩니다. 동일한 비교에서는 inference confidence, evaluation confidence, match IoU를 세 모델 모두 같게 유지해야 합니다.
평가를 실행할 때마다 `evaluation_summary.png`의 그래프 아래에 전체·번호판·차량의
GT, 예측, TP, FP, FN, Precision, Recall, F1, 탐지율, mIoU, AP 요약표가 같이 생성됩니다.
동일한 수치는 `evaluation_summary_table.csv`로도 저장되며, confidence/IoU 기준값을 각 행에 함께 기록합니다.

## 한 명령으로 전체 실행

API key와 GPU를 준비했다면 다음 명령이 다운로드부터 평가까지 순서대로 실행합니다. 이미 완료된 다운로드와 전처리는 감지해 재사용합니다.

```bash
python 03_two_stage/run_pipeline.py \
  --device 0 \
  --epochs 50 \
  --batch -1 \
  --workers 8 \
  --inference-conf 0.001 \
  --vehicle-crop-conf 0.25 \
  --global-nms-iou 0.50 \
  --evaluation-conf 0.25 \
  --match-iou 0.50
```

일부 단계부터 재개할 수도 있습니다.

```bash
# 기존 best.pt부터 inference + evaluation
python 03_two_stage/run_pipeline.py \
  --start-stage inference \
  --model 03_two_stage/runs/<run-name>/weights/best.pt \
  --device 0

# 기존 inference 결과만 evaluation
python 03_two_stage/run_pipeline.py \
  --start-stage evaluation \
  --predictions 03_two_stage/results/predictions/<inference-run>
```

## 실행 로그

각 스크립트를 단독 실행해도 터미널 출력과 같은 핵심 내용이 다음 폴더에 저장됩니다.

```text
03_two_stage/logs/
├── download_<timestamp>.log
├── preprocess_<timestamp>.log
├── check_<timestamp>.log
├── train_<timestamp>.log
├── inference_<timestamp>.log
├── evaluation_<timestamp>.log
└── pipeline_<timestamp>.log
```

`run_pipeline.py`로 실행하면 각 단계의 개별 로그와 함께 모든 하위 프로세스 출력을
합친 `pipeline_<timestamp>.log`도 생성됩니다. 학습 run 안의 `results.csv`와
`loss_curve.png`는 로그와 별도로 epoch별 수치 및 train/validation loss 변화를 보존합니다.

## 출처와 라이선스

- Dataset: `license_plate_vehicle`, parv217, Roboflow Universe, version 2
- Dataset license displayed by Roboflow Universe: CC BY 4.0
