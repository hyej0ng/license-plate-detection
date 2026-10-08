# Small Object License Plate Detection

This project uses YOLOv26n to detect vehicle license plates and compares methods for improving the detection of small objects at long distances. Its goal is to present data preprocessing, coordinate transformation verification, experiment reproducibility, and error case analysis as one complete detection pipeline instead of focusing only on model training.

## Research Questions

This project focuses on the following questions.

1. How well can the baseline YOLOv26n detect small objects such as license plates?
2. Can searching four separate image regions improve recall for distant license plates?
3. Is a two stage method that detects vehicles first and then searches each vehicle region for license plates more effective than detection on the entire image?
4. How can differences in performance be explained through resolution, object size, data distribution, false positives, and false negatives?

## Methods

| Method | Description | Status |
| --- | --- | --- |
| Baseline | Detect license plates with YOLOv26n after 640×640 letterbox preprocessing | Training |
| Quarter search | Split each image into four regions with margins, restore the source coordinates, and apply NMS | Planned |
| Two stage detection | Detect vehicles → crop vehicle regions → detect license plates again → restore source coordinates | Planned |

All methods will be compared on the same test set using mIoU, Precision, Recall, F1 Score, detection rate, mAP@0.5, and mAP@0.5:0.95. Each evaluation result will record the confidence threshold and NMS IoU threshold.

## Current Implementation

### Data Preprocessing

* Match images and JSON labels by file stem
* Convert license plate bounding boxes from JSON to `(xmin, ymin, xmax, ymax)`
* Remove duplicate or invalid bounding boxes
* Resize images to 640×640 with letterboxing while preserving the source aspect ratio
* Transform bounding box coordinates according to the padding and scale
* Normalize labels to the YOLO format `(class, cx, cy, width, height)`
* Draw bounding boxes on images before and after transformation for visual verification

The current preprocessed dataset has the following composition.

| Split | Images | Label files |
| --- | ---: | ---: |
| Train | 193,975 | 193,975 |
| Validation | 13,300 | 13,300 |
| Test | 485 | 485 |

### Baseline Training

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

The training script distinguishes among a new training run, a run that starts with a new optimizer from specified weights, and an exact resume that restores the optimizer and epoch. It records loss and detection metrics after every epoch and saves loss curves.

## Repository Structure

```text
.
├── 01_baseline/
│   ├── configs/                   # Dataset YAML example
│   └── scripts/
│       ├── 01_preprocessing/      # JSON → YOLO preprocessing and visual verification
│       ├── 02_training/           # YOLOv26n training and resume
│       ├── 03_inference/          # Planned
│       └── 04_evaluation/         # Planned
├── 02_quarter/                    # Quarter search experiments
├── 03_two_stage/                  # Vehicle focused two stage detection experiments
├── common/                        # Shared utilities and local model paths
├── comparison/                    # Final tables and graphs that can be published
└── requirements.txt
```

Data, Jupyter notebooks, training logs, experiment runs, prediction images, and model weights are not included in the repository. Only final tables and anonymized graphs that can be published are selected for the `comparison/` folder.

## Setup

An environment with Python 3.11 and CUDA support is recommended.

```bash
conda create -n license-plate-detection python=3.11 -y
conda activate license-plate-detection
python -m pip install -r requirements.txt
```

Create a local data configuration file, then change `path` to the absolute path of the preprocessed data.

```bash
cp 01_baseline/configs/license_plate.example.yaml \
  01_baseline/configs/license_plate.yaml
```

After reviewing the distribution terms, place the pretrained model at the local path below. Files with the `.pt` extension are excluded from Git.

```text
common/weights/pretrained/yolo26n.pt
```

## Usage

Set the source data root as an environment variable, then verify the coordinate transformation result for one image. You can also pass `--data-root` directly to each command.

```bash
export LICENSE_PLATE_DATA_ROOT=/absolute/path/to/raw-dataset

python 01_baseline/scripts/01_preprocessing/preprocess_baseline.py \
  --mode visualize_raw --split train

python 01_baseline/scripts/01_preprocessing/preprocess_baseline.py \
  --mode preprocess_one --split train
```

After visual verification, preprocess the entire dataset and train the baseline model.

```bash
python 01_baseline/scripts/01_preprocessing/preprocess_baseline.py \
  --mode preprocess_all

python 01_baseline/scripts/02_training/train_baseline.py
```

You can resume an interrupted training run exactly or start training with a new optimizer from specified weights.

```bash
python 01_baseline/scripts/02_training/train_baseline.py \
  --resume /absolute/path/to/last.pt

python 01_baseline/scripts/02_training/train_baseline.py \
  --weights /absolute/path/to/best.pt
```

## Data and Model Policy

* Source data and preprocessed data are not published due to their size, licensing terms, and potential privacy concerns.
* Prediction and error images containing vehicles or license plates are used only after publication rights and anonymization have been confirmed.
* Training weights and YOLO model files are not included in the repository.
* The exact dataset name, source, and license will be added to the documentation after publication rights have been confirmed.

## Project Context

This is an undergraduate research project conducted in an AI laboratory. It documents data processing accuracy, experiment design, fair comparison among models, and quantitative and qualitative evaluation with a focus on small object detection.
