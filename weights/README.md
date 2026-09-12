# Final model artifacts

`yolo11n_final_best.pt` is the selected YOLO11-N checkpoint from epoch 489 of
the final 500-epoch, 640x416, batch-8 run. Its validation mAP50:95 was
`0.9111456488171076`.

`yolo11n_coco.pt` is the official Ultralytics COCO initialization used to
reconstruct YOLO11-N before loading the trained EMA state.

`frozen_validation_selection.json` records the validation-only epoch and
confidence-threshold selection. The frozen threshold used by inference is
`0.3985197842121124`.

`yolo11n_final_run_config.json` preserves the exact training configuration.
