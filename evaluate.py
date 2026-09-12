"""Evaluate the final YOLO11-N checkpoint with its frozen validation decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from ultralytics.utils.nms import non_max_suppression

from data_loading import (
    BoundingBox,
    ImageRecord,
    Prediction,
    evaluate_detections,
    ground_truth_from_records,
    load_dataset,
    load_split_records,
)
from models.yolo11 import (
    YOLO11ProbeDataset,
    _box_from_letterboxed_coordinates,
    _build_official_model,
    _input_size_from_run_config,
)


#Picks the device to run on
def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


# Hashes the semantic manifest bytes consistently after Git checkout on Windows or Unix.
# Only newline encoding is normalized; changed paths or ordering still fail verification.
def _split_manifest_sha256(path: Path) -> str:
    contents = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(contents).hexdigest()


#Loads a trained checkpoint into a fresh model and returns it ready for inference
def _load_model(
    weights: Path, pretrained: Path, device: torch.device
) -> tuple[Any, int | tuple[int, int]]:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    #Makes sure this file is actually a training checkpoint, not some other .pt file
    if "ema" not in checkpoint or "run_config" not in checkpoint:
        raise ValueError("Weights are not a probe-training checkpoint")
    model = _build_official_model(pretrained, device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    return model.eval(), _input_size_from_run_config(checkpoint["run_config"])


#Runs one image through the model and returns its best box and confidence, or None if nothing found
@torch.inference_mode()
def _predict(
    model: Any,
    image_path: Path,
    device: torch.device,
    input_size: int | tuple[int, int],
    confidence_threshold: float,
    nms_iou_threshold: float,
) -> tuple[BoundingBox, float] | None:
    with Image.open(image_path) as source:
        width, height = source.size
    #Builds a placeholder full-image box just to reuse the shared dataset loader
    record = ImageRecord(0, image_path, width, height, BoundingBox(0, 0, width, height))
    #Reuses the same preprocessing adapter used in training and validation
    sample = YOLO11ProbeDataset(
        [record], training=False, seed=0, input_size=input_size
    )[0]
    #Adds a batch dimension and scales pixels from 0-255 down to 0-1
    image = sample["img"].unsqueeze(0).to(device).float().div_(255.0)
    #Only one class and at most one detection, since each image has at most one probe
    detections = non_max_suppression(
        model(image),
        conf_thres=confidence_threshold,
        iou_thres=nms_iou_threshold,
        classes=[0],
        max_det=1,
        nc=1,
        max_time_img=10.0,
    )[0]
    if len(detections) == 0:
        return None
    detection = detections[0]
    #Converts the box from the padded model space back to the original image size
    box = _box_from_letterboxed_coordinates(
        detection[:4].detach().cpu().tolist(), width, height, input_size
    )
    #A bad box here would mean a real bug, not a normal outcome so raise fail
    if box is None:
        raise RuntimeError(f"Model produced an invalid box for {image_path.name}")
    return box, float(detection[4].detach().cpu())


#Runs every record through the model and collects the results into a list of predictions
def _collect(model, records, device, input_size, nms_iou_threshold):
    predictions = []
    for record in records:
        #Uses a very low threshold here so almost nothing is dropped early
        result = _predict(
            model,
            record.path,
            device,
            input_size,
            confidence_threshold=0.001,
            nms_iou_threshold=nms_iou_threshold,
        )
        if result is not None:
            box, confidence = result
            predictions.append(Prediction(record.image_id, box, confidence))
    return tuple(predictions)


#checks the run matches the frozen decision, then scores the model on the test split
def main() -> None:
    #Folder this script lives in, used for default weight paths
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, default=root / "weights/yolo11n_final_best.pt"
    )
    parser.add_argument(
        "--pretrained", type=Path, default=root / "weights/yolo11n_coco.pt"
    )
    #The file recording the epoch and confidence threshold already picked on validation data
    parser.add_argument(
        "--frozen-selection",
        type=Path,
        default=root / "weights/frozen_validation_selection.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--matching-iou-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=Path("output/metrics.json"))
    args = parser.parse_args()

    frozen = json.loads(args.frozen_selection.read_text(encoding="utf-8-sig"))
    #Refuses to run unless this file confirms the threshold was chosen without looking at test data
    if frozen.get("test_split_used") is not False:
        raise ValueError("Frozen selection does not certify that test data was unused")
    threshold_record = frozen["threshold_selection"]
    #Makes sure the validation split file hasn't changed since the threshold was originally chosen
    if _split_manifest_sha256(args.val_split) != threshold_record["split_sha256"]:
        raise ValueError("Validation manifest differs from the frozen selection")

    all_records = load_dataset(args.images, args.annotations)
    load_split_records(all_records, args.val_split)  # validates the val split against the current dataset
    test_records = load_split_records(all_records, args.test_split)
    device = _resolve_device(args.device)
    model, input_size = _load_model(args.weights, args.pretrained, device)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    #Makes sure this is the exact checkpoint the frozen decision was based on, not a different one
    if int(checkpoint["epoch"]) != int(frozen["checkpoint_epoch"]):
        raise ValueError("Final checkpoint epoch differs from the frozen selection")
    threshold = float(threshold_record["confidence_threshold"])
    #Makes sure today's thresholds match the frozen ones, so results stay comparable
    if args.matching_iou_threshold != float(
        threshold_record["matching_iou_threshold"]
    ):
        raise ValueError("Matching IoU threshold differs from the frozen selection")
    if args.nms_iou_threshold != float(threshold_record["nms_iou_threshold"]):
        raise ValueError("NMS IoU threshold differs from the frozen selection")
    test_predictions = _collect(
        model, test_records, device, input_size, args.nms_iou_threshold
    )
    #Shared settings passed into evaluate_detections below
    common = {
        "confidence_threshold": threshold,
        "matching_iou_threshold": args.matching_iou_threshold,
        "max_detections_per_image": 1,
    }
    test = evaluate_detections(
        ground_truth_from_records(test_records), test_predictions, **common
    )
    #Combines run info, the already-frozen validation numbers, and the computed test numbers
    report = {
        "checkpoint": str(args.weights),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold_selected_on": "validation and frozen before test",
        "confidence_threshold": threshold,
        "validation": frozen["validation"],
        "test": asdict(test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
