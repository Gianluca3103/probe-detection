# Single-probe test failures

This folder contains every failing image from the seven-fold top-1 evaluation.
Each of the 308 dataset images was evaluated in exactly one outer test fold.

- Green: ground-truth probe box.
- Cyan: a correctly matched prediction.
- Orange: the retained highest-confidence prediction when it did not match.
- `FN`: the ground-truth probe had no match at IoU 0.5.
- `FP1`: the one retained prediction was incorrect.

There are seven unique failing images: seven false negatives in total, with two
of those images also producing a false positive. `index.csv` records each image,
fold, ground-truth box, retained prediction, confidence, IoU, and status.
