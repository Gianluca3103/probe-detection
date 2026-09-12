"""Build the final benchmark-trained YOLO11-N TensorRT FP16 engine.

Run this on the deployment device because TensorRT engines are not generally
portable across GPU architectures and TensorRT versions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from data_loading import sha256_file
from models.yolo11 import _build_official_model, _input_size_from_run_config

#Height and width of source images
SOURCE_HEIGHT = 400
SOURCE_WIDTH = 640
#The actual YOLO input after padding 400->416 so it is stride-32 compatible
MODEL_HEIGHT = 416
MODEL_WIDTH = 640
MODEL_NAME = "YOLO11-N"
#Pins the exact checkpoint files
MODEL_CHECKPOINT_SHA256 = "d9574d854d3cb87b8ea6fa144ac40e86d6defedc545ad7dfddda4ef17a195d90"
PRETRAINED_CHECKPOINT_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
#The frozen validation threshold
CONFIDENCE_THRESHOLD = 0.3985197842121124


#Preprocessing of input so it's yolo compatible
class _RawFrameModel(torch.nn.Module):

    def __init__(self, model: torch.nn.Module, padding_top: int, padding_bottom: int) -> None:
        super().__init__()
        self.model = model
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom

    #Runs one raw uint8 frame through preprocessing then the model
    def forward(self, raw_images: torch.Tensor) -> torch.Tensor:
        #Converts to fp16
        images = raw_images.to(torch.float16).permute(0, 3, 1, 2)
        #Pads only the height (400 -> 416)
        images = functional.pad(
            images, (0, 0, self.padding_top, self.padding_bottom), value=0.0
        )
        #Normalizes raw 0-255 pixel values into the 0-1 range the model was trained on
        images = images.mul(1.0 / 255.0)
        output = self.model(images)
        output = output[0] if isinstance(output, (tuple, list)) else output
        return output.to(torch.float16)


#Loads the checkpoint, confirms it's really a YOLO11-N run at the expected input size,
#and computes how much top/bottom padding is needed to the increase 400 -> 416
def _checkpoint_details(weights: Path) -> tuple[dict, int, int, int, int]:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    configured = _input_size_from_run_config(checkpoint["run_config"])
    model_height, model_width = (
        (configured, configured) if isinstance(configured, int) else configured
    )
    #Fails fast if someone points this script at a checkpoint from a different model or resolution
    if checkpoint["run_config"].get("architecture") != MODEL_NAME:
        raise ValueError(f"Expected a {MODEL_NAME} training checkpoint")
    if (model_height, model_width) != (MODEL_HEIGHT, MODEL_WIDTH):
        raise ValueError(
            f"Expected model input {(MODEL_HEIGHT, MODEL_WIDTH)}, "
            f"got {(model_height, model_width)}"
        )
    #Same centered-padding split used to split the gap between top and bottom
    padding_top = (model_height - SOURCE_HEIGHT) // 2
    padding_bottom = model_height - SOURCE_HEIGHT - padding_top
    return checkpoint, model_height, model_width, padding_top, padding_bottom


#Builds the trained model, wraps it with the fused preprocessing, and exports it to ONNX
def _export_raw_onnx(
    pretrained: Path,
    destination: Path,
    checkpoint: dict,
    padding_top: int,
    padding_bottom: int,
) -> None:
    #Only needed here, so imported locally instead of at module load time
    from ultralytics.nn.modules import Detect

    model = _build_official_model(pretrained, torch.device("cuda:0"))
    model.load_state_dict(checkpoint["ema"], strict=True)
    #FP16 for the engine's precision
    model.eval().half()
    #Switches the detection head to emit raw ONNX-export-friendly output instead of its normal
    #training/inference postprocessing
    for module in model.modules():
        if isinstance(module, Detect):
            module.export = True
            module.format = "onnx"
            module.dynamic = False
    wrapper = _RawFrameModel(model, padding_top, padding_bottom).eval()
    #Dummy input, only used to trace the graph's shapes; its actual pixel values don't matter
    example = torch.zeros(
        (1, SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=torch.uint8, device="cuda:0"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        example,
        destination,
        input_names=["raw_images"],
        output_names=["raw_predictions"],
        opset_version=17,
        do_constant_folding=True,
        #No dynamic axes: forces a fully fixed-shape export, matching the engine's fixed I/O contract
        dynamic_axes=None,
    )


def _plugin_threshold(threshold: float) -> float:

    #Nudges the threshold up to the next representable FP16 value, so the plugin's FP32
    #comparison filters at least as strictly as the FP16 model's own scores would
    return float(np.nextafter(np.float16(threshold), np.float16(np.inf)))


#Manually edits the exported ONNX graph to add NMS filtering, restore coordinates back to the
#original 640x400 frame, and pack everything into the final [count, detection] output the
#deployment runtime (ProbeDetector) expects
def _add_top1_postprocessing(
    source: Path,
    destination: Path,
    confidence_threshold: float,
    padding_top: int,
) -> None:
    import onnx
    from onnx import helper, numpy_helper

    graph = onnx.load(source)
    nodes = graph.graph.node

    #Registers a fixed numeric value as a new graph constant, returns its name for use as a node input
    def constant(name, value, dtype=np.float32):
        graph.graph.initializer.append(
            numpy_helper.from_array(np.asarray(value, dtype=dtype), name)
        )
        return name

    #Appends one computation node to the graph, returns its output name so it can feed the next node
    def node(operation, inputs, name, **attributes):
        nodes.append(
            helper.make_node(operation, inputs, [name], name=name, **attributes)
        )
        return name

    raw = graph.graph.output[0].name
    #YOLO's raw output is channel-first; transpose so each detection becomes its own row
    rows = node("Transpose", [raw], "nms_rows", perm=[0, 2, 1])

    #Slices out specific columns from the detection rows by index (e.g. box center vs size vs score)
    def gather(indices, name):
        return node(
            "Gather",
            [rows, constant(f"{name}_indices", indices, np.int64)],
            name,
            axis=2,
        )

    #Splits the raw center+size box encoding into its two halves
    center = gather([0, 1], "box_center")
    size = gather([2, 3], "box_size")
    #Converts center+size into x1y1x2y2 corners: center minus half-size, center plus half-size
    half = node("Div", [size, constant("two", 2, np.float16)], "half_size")
    low = node("Sub", [center, half], "box_low")
    high = node("Add", [center, half], "box_high")
    corners = node("Concat", [low, high], "boxes_fp16", axis=2)
    #Casts fp16 -> fp32 since the TensorRT NMS plugin expects fp32 boxes/scores
    boxes = node("Cast", [corners], "boxes", to=onnx.TensorProto.FLOAT)
    scores_half = gather([4], "scores_fp16")
    scores = node("Cast", [scores_half], "scores", to=onnx.TensorProto.FLOAT)
    threshold = _plugin_threshold(confidence_threshold)
    #The actual NMS: max_output_boxes=1 enforces the single-probe assumption, background_class=-1
    #means there's no background class to exclude, score_activation=0 means scores are already
    #plain probabilities, not raw logits needing a sigmoid
    nodes.append(
        helper.make_node(
            "EfficientNMS_TRT",
            [boxes, scores],
            ["nms_count", "nms_boxes", "nms_scores", "nms_classes"],
            name="top1_nms",
            domain="trt.plugins",
            plugin_version="1",
            background_class=-1,
            max_output_boxes=1,
            score_threshold=threshold,
            iou_threshold=0.7,
            score_activation=0,
            box_coding=0,
        )
    )
    #remove top/bottom padding back, converting from the padded 640x416
    #model space back toward the original 640x400 source frame (x offset is always 0)
    restored = node(
        "Sub",
        [
            "nms_boxes",
            constant("letterbox_padding", [0, padding_top, 0, padding_top]),
        ],
        "unpad",
    )
    #Sets coordinates into the valid 640x400 frame in case unpadding pushed them slightly out of bounds
    restored = node("Max", [restored, constant("zero", 0)], "clamp_low")
    restored = node(
        "Min",
        [restored, constant("source_bounds", [640, 400, 640, 400])],
        "restored_boxes",
    )
    #Reshapes the per-detection scalars (confidence, class) into a column so they can be
    #concatenated alongside the 4-column box below
    axes = constant("column_axis", [2], np.int64)
    confidence = node("Unsqueeze", ["nms_scores", axes], "confidence")
    classes = node(
        "Cast", ["nms_classes"], "classes_float", to=onnx.TensorProto.FLOAT
    )
    classes = node("Unsqueeze", [classes, axes], "classes_column")
    #Assembles the final [x1, y1, x2, y2, confidence, class] row, matching the documented output contract
    detection = node(
        "Concat", [restored, confidence, classes], "detection", axis=2
    )
    #At very low thresholds the own count can lose precision, so recompute it directly
    #by comparing scores against the threshold instead of trusting the  count output
    count_source = "nms_count"
    if threshold < 0.007:
        valid = node(
            "GreaterOrEqual",
            ["nms_scores", constant("valid_floor", threshold)],
            "valid_scores",
        )
        valid = node("Cast", [valid], "valid_int", to=onnx.TensorProto.INT32)
        count_source = node(
            "ReduceSum",
            [valid, constant("count_axis", [1], np.int64)],
            "valid_count",
            keepdims=1,
        )
    count = node("Cast", [count_source], "count_float", to=onnx.TensorProto.FLOAT)
    count = node("Unsqueeze", [count, axes], "count_column")
    #Builds the header row: the count value followed by 5 zero columns, matching the detection
    #row's 6-column width so the two rows can be stacked together
    header = node(
        "Concat",
        [count, constant("header_zeros", np.zeros((1, 1, 5)))],
        "header",
        axis=2,
    )
    #Stacks header row + detection row into the final [1, 2, 6] packed output tensor
    node("Concat", [header, detection], "packed_detections", axis=1)
    #Replaces the model's original raw output declaration with the new packed output
    del graph.graph.output[:]
    graph.graph.output.append(
        helper.make_tensor_value_info(
            "packed_detections", onnx.TensorProto.FLOAT, [1, 2, 6]
        )
    )
    #Registers the custom TensorRT opset so the graph is recognized as using EfficientNMS_TRT
    graph.opset_import.append(helper.make_opsetid("trt.plugins", 1))
    onnx.save(graph, destination)


#Orchestrates the whole build: validates inputs, exports the ONNX graph, adds NMS, compiles the
#TensorRT engine, and writes a metadata sidecar file recording exactly what went into it
def build_engine(
    weights: Path,
    pretrained: Path,
    destination: Path,
    confidence_threshold: float,
    workspace_gib: float | None,
) -> None:

    #Building requires a GPU: TensorRT compiles the engine for the specific hardware it runs on
    if not torch.cuda.is_available():
        raise RuntimeError("Engine building requires an NVIDIA CUDA device")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1")
    import tensorrt as trt
    from ultralytics.utils.export.engine import onnx2engine

    #Refuses to build from the wrong weights file, so the wrong checkpoint never gets silently baked in
    if sha256_file(weights) != MODEL_CHECKPOINT_SHA256:
        raise ValueError(f"Weights are not the selected {MODEL_NAME} checkpoint")
    if sha256_file(pretrained) != PRETRAINED_CHECKPOINT_SHA256:
        raise ValueError("Pretrained weights are not the official YOLO11-N checkpoint")

    #Registers TensorRT's library, needed since the graph will use the EfficientNMS_TRT plugin
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")

    #Two intermediate ONNX files one before NMS is added, one after
    raw_onnx = destination.with_name(f"{destination.stem}_raw.onnx")
    final_onnx = destination.with_suffix(".onnx")
    checkpoint, model_height, model_width, padding_top, padding_bottom = (
        _checkpoint_details(weights)
    )
    _export_raw_onnx(
        pretrained,
        raw_onnx,
        checkpoint,
        padding_top,
        padding_bottom,
    )
    _add_top1_postprocessing(
        raw_onnx, final_onnx, confidence_threshold, padding_top
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx2engine(
        str(final_onnx),
        output_file=destination,
        workspace=None if workspace_gib is None else int(workspace_gib),
        #quantize=16 builds in FP16 precision; dynamic=False and a fixed shape lock the engine
        #to exactly one input size, matching ProbeDetector's fixed-shape contract
        quantize=16,
        dynamic=False,
        shape=(1, SOURCE_HEIGHT, SOURCE_WIDTH, 3),
        metadata=None,
        verbose=False,
    )
    #Records everything needed to audit or reproduce this exact build later; saved as the
    #engine's own .json sidecar file (deployment_artifacts/probe_detector.json)
    metadata = {
        "model": checkpoint["run_config"].get("architecture", MODEL_NAME),
        "source_checkpoint": weights.name,
        "source_checkpoint_sha256": sha256_file(weights),
        "pretrained_checkpoint": pretrained.name,
        "pretrained_checkpoint_sha256": sha256_file(pretrained),
        "engine_sha256": sha256_file(destination),
        "source_shape": [1, SOURCE_HEIGHT, SOURCE_WIDTH, 3],
        "source_dtype": "uint8",
        "source_color": "RGB",
        "model_input_shape": [1, 3, model_height, model_width],
        "precision": "FP16",
        "confidence_threshold": confidence_threshold,
        "nms_iou_threshold": 0.7,
        "max_detections": 1,
        "postprocessing": "EfficientNMS_TRT",
        "build_gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "portability_note": "Rebuild this engine on the deployment device.",
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# builds the engine from this project's fixed checkpoint paths
def main() -> None:
    #Repo root, two levels up from this file (deployment/ -> final_code/)
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "deployment_artifacts/probe_detector.engine",
    )
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    args = parser.parse_args()
    build_engine(
        root / "weights/yolo11n_final_best.pt",
        root / "weights/yolo11n_coco.pt",
        args.output,
        CONFIDENCE_THRESHOLD,
        args.workspace_gib,
    )
    print(f"Saved TensorRT engine: {args.output}")


if __name__ == "__main__":
    main()
