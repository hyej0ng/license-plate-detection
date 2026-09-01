# Margin-based Quarter Detection

원본 이미지를 서로 조금 겹치는 네 영역으로 나누어 작은 번호판을 더 크게 보는
YOLOv26n 파이프라인입니다. `01_baseline`은 수정하지 않으며, JSON bbox 해석과
letterbox 방식, 학습 로그 형식을 동일하게 유지합니다.

## 핵심 설정

- `margin_ratio=0.05`: 중앙 경계의 양쪽을 원본 폭/높이의 5%씩 확장합니다.
  따라서 좌우 또는 상하 crop의 전체 겹침 폭은 해당 축의 10%입니다.
- `min_visible_ratio=0.50`: 원본 bbox 면적 중 crop 안에 50% 이상 보일 때만
  해당 crop의 학습 라벨로 사용합니다.
- `image_size=640`: 각 crop을 종횡비를 유지하는 letterbox 방식으로 640×640로 만듭니다.
- inference의 `margin_ratio`는 학습 데이터 생성 때와 같은 값을 사용해야 합니다.

## 처리 순서

```text
Train/Val 원본 + JSON
  -> 겹치는 4개 crop
  -> bbox 교집합/가시 비율 검사
  -> 640 letterbox + YOLO 라벨
  -> YOLOv26n train/validation

Test 원본
  -> 같은 공식으로 메모리 4분할
  -> crop별 예측
  -> letterbox 역변환
  -> 원본 좌표 복원
  -> Global NMS
  -> 원본 JSON과 IoU 1:1 매칭 평가
```

## 실행

저장소 루트에서 `conda activate yolo` 후 실행합니다.

먼저 소수 샘플로 좌표를 확인합니다.

```bash
python 02_quarter/scripts/01_preprocessing/preprocess_quarter.py \
  --splits train val \
  --margin-ratio 0.05 \
  --min-visible-ratio 0.50 \
  --image-size 640 \
  --max-samples 5 \
  --visualize-count 5
```

시각화가 맞으면 `--max-samples`를 빼고 전체 Train/Val을 생성합니다.

```bash
python 02_quarter/scripts/01_preprocessing/preprocess_quarter.py \
  --splits train val \
  --margin-ratio 0.05 \
  --min-visible-ratio 0.50 \
  --image-size 640
```

학습과 validation을 실행합니다.

전체 학습 전에 데이터와 가중치 연결만 검사할 수도 있습니다.

```bash
python 02_quarter/scripts/02_training/train_quarter.py --check-only --device cpu
```

```bash
python 02_quarter/scripts/02_training/train_quarter.py \
  --device 0 \
  --epochs 50 \
  --batch -1 \
  --patience 0 \
  --seed 42
```

중단된 학습을 정확히 이어갈 때는 다음과 같이 실행합니다.

```bash
python 02_quarter/scripts/02_training/train_quarter.py \
  --resume 02_quarter/runs/<run-name>/weights/last.pt \
  --device 0
```

Test inference에서는 전처리와 같은 margin을 사용합니다.

```bash
python 02_quarter/scripts/03_inference/inference_quarter.py \
  --model 02_quarter/runs/<run-name>/weights/best.pt \
  --margin-ratio 0.05 \
  --confidence 0.25 \
  --nms-iou 0.50 \
  --device 0
```

위 명령이 출력한 inference 폴더를 평가합니다.

```bash
python 02_quarter/scripts/04_evaluation/evaluate_quarter.py \
  --predictions 02_quarter/results/predictions/<inference-run> \
  --ground-truth-format json \
  --confidence 0.25 \
  --match-iou 0.50
```

동일한 평가기로 baseline의 YOLO 정답/예측 CSV도 평가할 수 있습니다.

```bash
python 02_quarter/scripts/04_evaluation/evaluate_quarter.py \
  --predictions 01_baseline/results/predictions/<baseline-run> \
  --ground-truth-format yolo \
  --image-root 01_baseline/data/preprocessed/images/test \
  --ground-truth-root 01_baseline/data/preprocessed/labels/test \
  --confidence 0.25 \
  --match-iou 0.50
```

`--max-samples`는 전처리/inference 흐름 확인용입니다. 제한된 샘플의 성능을
최종 실험 성능으로 사용하지 않습니다.
