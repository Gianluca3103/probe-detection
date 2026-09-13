# Final YOLO11-N probe detector

This directory contains the code and artifacts for the final probe detector used
in the report. The model is YOLO11-N; the TensorRT FP16, embedded top-1 NMS,
persistent-buffer, and CUDA Graph inference design is retained.

## Final model

- Architecture: YOLO11-N
- Model input: 640x416 (height x width), from a 640x400 source frame
- Selected checkpoint: epoch 489 of 500 (`weights/yolo11n_final_best.pt`, not `last.pt`)
  — SHA-256 `d9574d854d3cb87b8ea6fa144ac40e86d6defedc545ad7dfddda4ef17a195d90`
- Frozen validation threshold: `0.3985197842121124`
- Matching IoU threshold: `0.5`
- NMS IoU threshold: `0.7`
- Selection rule: maximum validation mAP50:95 to pick the checkpoint, then maximum
  validation F1 to pick the confidence threshold
- Validation mAP50:95: `0.9111456488171076`
- Held-out test mAP50:95: `0.9048262567092171`
- Held-out precision, recall, and F1: `1.0`, `1.0`, and `1.0`
- Held-out mean IoU: `0.9388794252435645`; TP/FP/FN: `46/0/0` over 46 test images
- Training/validation images: 216/46

The epoch and confidence threshold were selected exclusively on validation data
and frozen before the held-out test evaluation. See
`weights/frozen_validation_selection.json` for the exact frozen decision and
`weights/yolo11n_final_run_config.json` for the full training configuration.

The seven run cross-validation results in `results/` are supporting evidence
across the full dataset. Exclusively used for analysis.

## Directory layout

```text
final_code/
|-- README.md
|-- requirements.txt
|-- requirements-deployment.txt
|-- inference.py
|-- evaluate.py
|-- data_loading/
|-- models/yolo11.py
|-- deployment/
|-- splits/benchmark/
|-- splits/fold_1/ ... fold_7/
|-- results/
|-- tests/
`-- weights/
```

The image dataset and `probe_labels.json` annotations are intentionally not
duplicated. Place them inside this directory or pass their locations explicitly
to the training and evaluation tools.

## Installation

Python 3.11, PyTorch 2.7.1 with CUDA 12.8, and Ultralytics 8.4.138 were used.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For engine building and TensorRT inference:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-deployment.txt
```

## TensorRT/CUDA Graph inference

The engine accepts fixed batch-1, 640x400 RGB uint8 frames. It pads eight pixels
above and below the frame to produce the model's 640x416 tensor, normalizes and
runs YOLO11-N in FP16, applies the frozen confidence threshold and top-1 NMS,
then restores coordinates to the original 640x400 frame. 

Build the local engine before the first inference run:

```powershell
.\.venv\Scripts\python.exe -m deployment.build_engine
```

```powershell
.\.venv\Scripts\python.exe inference.py path\to\image_or_folder `
  --output-dir output\inference
```

Input images with dimensions other than 640x400 are rejected. The script writes
annotated images, `detections.csv`, and `inference_summary.json`.

TensorRT engines are tied to their GPU architecture and TensorRT stack, so run
the same build command again on each deployment device.

## Reproduce the final training run

The dataset is external to this repository. Set `$datasetRoot` to the folder
containing `probe_images/` and `probe_labels.json` before starting training.

```powershell
$datasetRoot = 'C:\path\to\probe_dataset'

.\.venv\Scripts\python.exe -u -m models.yolo11 `
  --images "$datasetRoot\probe_images" `
  --annotations "$datasetRoot\probe_labels.json" `
  --pretrained-checkpoint weights\yolo11n_coco.pt `
  --checkpoint-dir checkpoints\yolo11_n_640x416_constant_500_batch8 `
  --train-split splits\benchmark\train.txt `
  --val-split splits\benchmark\val.txt `
  --input-width 640 `
  --input-height 416 `
  --epochs 500 `
  --batch-size 8 `
  --validation-batch-size 16 `
  --learning-rate 3.125e-5 `
  --learning-rate-schedule linear `
  --final-learning-rate-fraction 1.0 `
  --optimizer AdamW `
  --momentum 0.9 `
  --weight-decay 0.0001 `
  --dropout 0 `
  --warmup-epochs 0 `
  --nominal-batch-size 8 `
  --augmentation-policy shared `
  --seed 42 `
  --num-workers 6 `
  --cache-images `
  --amp `
  --device cuda `
  --early-stopping-patience 0 `
  --confidence-threshold 0.25 `
  --matching-iou-threshold 0.5 `
  --nms-iou-threshold 0.7 `
  --defer-validation
```

Deferred validation saves one evaluation-only EMA checkpoint per epoch and does
not use validation performance during optimization. The final run configuration
and frozen selection are preserved in `weights/`.

## Reproduce validation-threshold selection and test evaluation

The dataset is not included in this repository. Set `$datasetRoot` to the folder
containing the provided `probe_images/` directory and `probe_labels.json` file.

```powershell
$datasetRoot = 'C:\path\to\probe_dataset'

.\.venv\Scripts\python.exe evaluate.py `
  --images "$datasetRoot\probe_images" `
  --annotations "$datasetRoot\probe_labels.json" `
  --val-split splits\benchmark\val.txt `
  --test-split splits\benchmark\test.txt `
  --device cuda `
  --output output\final_metrics.json
```

## Third-party considerations

YOLO11 and Ultralytics are third-party software. From what I have seen for commercial uses they might require licensing.
