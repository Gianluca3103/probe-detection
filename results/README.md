# Results

- `final_metrics.json`: validation-selected threshold and held-out test metrics
  for the actual deployed checkpoint (epoch 489, `weights/yolo11n_final_best.pt`).
  This is the checkpoint shipped in `weights/` and built into the TensorRT engine.
- `cross_validation_summary.csv`: operating metrics and latency for all seven
  frozen outer test folds, a separate cross-validation analysis across the full
  dataset (different checkpoints per fold, not the deployed one).
- `latency_summary.json`: compact TensorRT/CUDA Graph latency report without the
  thousands of raw per-frame timing samples.
- `fold4_metrics.json`: validation-selected threshold and held-out Fold 4
  metrics for that fold's own checkpoint (epoch 358) — part of the seven-fold
  analysis above, not the deployed checkpoint.
- `selected_visualizations/`: all seven cross-validation error cases, plus an
  index describing each miss or false positive.

Large diagnostic traces, exploratory experiments, augmentation previews, and
intermediate checkpoints are intentionally excluded.
