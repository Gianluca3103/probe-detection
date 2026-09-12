"""Train the official YOLO11-N model with the shared probe benchmark pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_loading import (
    BoundingBox,
    ImageRecord,
    Prediction,
    apply_detection_transform,
    build_evaluation_transform,
    build_training_transform,
    evaluate_detections,
    ground_truth_from_records,
    load_dataset,
    load_split_records,
    sha256_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_FAMILY = "YOLO11"
MODEL_VARIANT = "n"
CHECKPOINT_PREFIX = "yolo11"
#Fixed pretrained checkpoint for the final YOLO11-N model
PRETRAINED_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "pretrained" / "yolo11n.pt"
PRETRAINED_CHECKPOINT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
)
PRETRAINED_CHECKPOINT_SHA256 = (
    "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
)
#Pinned library versions this benchmark was verified against checked at startup
ULTRALYTICS_VERSION = "8.4.138"
TORCH_VERSION = "2.7.1+cu128"
TORCHVISION_VERSION = "0.22.1+cu128"
#Default square input size if none is given on the command line
INPUT_SIZE = 640
#Protects the native-augmentation path random seeding across parallel data-loader workers
_NATIVE_AUGMENTATION_SEED_LOCK = threading.Lock()


#Reads the input size a checkpoint was actually trained with, so other scripts can reuse it exactly
def _input_size_from_run_config(
    run_config: dict[str, Any],
) -> int | tuple[int, int]:
    input_height = run_config.get("input_height")
    input_width = run_config.get("input_width")
    if input_height is not None and input_width is not None:
        height, width = int(input_height), int(input_width)
        return height if height == width else (height, width)
    #Falls back to a single size field for older checkpoints saved before height/width were recorded
    return int(run_config.get("input_size", INPUT_SIZE))


class YOLO11ProbeDataset(Dataset):

    def __init__(
        self,
        records: Sequence[ImageRecord],
        *,
        training: bool,
        seed: int,
        input_size: int | tuple[int, int] = INPUT_SIZE,
        cache_images: bool = False,
    ) -> None:
        self.records = tuple(records)
        self.training = training
        self.seed = seed
        self.input_size = input_size
        #Shared memory so every data-loader worker process sees the same current epoch number
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        #Built once since the evaluation transform is deterministic no need to rebuild per sample
        self._evaluation_transform = build_evaluation_transform(input_size)
        #decode every image once upfront to trade memory for faster repeated epochs
        self._image_cache = self._cache_images() if cache_images else None

    def __len__(self) -> int:
        return len(self.records)

    #Updates the shared epoch counter, used below to vary each sample's augmentation seed per epoch
    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(epoch)

    #Returns one training/validation sample the image and its normalized box
    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self._image_cache is None:
            pixels = self._decode_image(record)
        else:
            pixels = self._image_cache[index]
        image = Image.fromarray(pixels.numpy(), mode="RGB")

        if self.training:
            #A fresh, seeded transform per sample
            transform = build_training_transform(
                seed=self._sample_seed(record.image_id), target_size=self.input_size
            )
        else:
            transform = self._evaluation_transform
        image, box = apply_detection_transform(transform, image, record.box)

        pixels = np.asarray(image, dtype=np.uint8).copy()
        return {
            "img": torch.from_numpy(pixels).permute(2, 0, 1),
            #Always class 0: this is a single-class (probe) detector
            "cls": torch.zeros((1, 1), dtype=torch.float32),
            "bboxes": torch.tensor(
                [_normalized_cxcywh(box, self.input_size)], dtype=torch.float32
            ),
            #Kept so a later prediction can be traced back to its source image
            "image_id": record.image_id,
        }

    #Combines seed, epoch, and image id into one deterministic per-sample seed
    def _sample_seed(self, image_id: int) -> int:
        epoch = int(self._epoch.item())
        return (self.seed + epoch * 1_000_003 + image_id * 97) % (2**32)

    #Loads one image file from disk into a raw RGB pixel tensor
    @staticmethod
    def _decode_image(record: ImageRecord) -> torch.Tensor:
        with Image.open(record.path) as source:
            pixels = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(pixels)

    #Decodes every image once, stacks them into one tensor if they're all the same size,
    #otherwise keeps them as a tuple since differently-shaped images can't be stacked
    def _cache_images(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        decoded = tuple(self._decode_image(record) for record in self.records)
        shapes = {tuple(image.shape) for image in decoded}
        if len(shapes) == 1:
            return torch.stack(decoded).share_memory_()
        return tuple(image.share_memory_() for image in decoded)


#The alternative augmentation path (Ultralytics' own built-in augmentations, chosen via
#--augmentation-policy yolo); tried once for comparison, not the final policy used
class YOLO11NativeAugmentationDataset:

    #Builds and returns an actual Ultralytics YOLODataset instance directly, instead of a plain
    #instance of this class, so Ultralytics' own dataset machinery can run completely unmodified
    def __new__(
        cls,
        records: Sequence[ImageRecord],
        *,
        seed: int,
        input_size: int | tuple[int, int],
        batch_size: int,
    ):
        #Only needed for this rarely-used path, so imported locally
        import albumentations as A
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.utils import IterableSimpleNamespace

        target_height, target_width = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        #Ultralytics' rectangular training mode requires the width to be the longer side
        if target_width != max(target_height, target_width):
            raise ValueError("YOLO rectangular training expects width >= height")

        class _Dataset(YOLODataset):
            def __init__(self) -> None:
                #Lets get_img_files/get_labels below map an Ultralytics-provided path back
                #to this project's own ImageRecord
                self._records_by_path = {
                    str(record.path.resolve()): record for record in records
                }
                self.seed = seed
                self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
                hyp = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
                # its four enabled defaults explicitly for the repository's
                hyp.augmentations = [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                ]
                #Hands off to Ultralytics' own dataset constructor with this project's
                #single-class ("probe") label and rectangular batching enabled
                super().__init__(
                    img_path=list(self._records_by_path),
                    imgsz=target_width,
                    augment=True,
                    hyp=hyp,
                    rect=True,
                    batch_size=batch_size,
                    stride=32,
                    pad=0.0,
                    data={"names": {0: "probe"}, "nc": 1, "channels": 3},
                )
                #Confirms Ultralytics actually produced the exact requested shape, rather than
                #silently picking a different rectangular size on its own
                actual_shapes = {
                    tuple(int(value) for value in shape) for shape in self.batch_shapes
                }
                if actual_shapes != {(target_height, target_width)}:
                    raise ValueError(
                        f"Ultralytics rectangular shape {actual_shapes} does not match "
                        f"requested {(target_height, target_width)}"
                    )

            #Tells Ultralytics which absolute paths to load, matching _records_by_path's keys
            def get_img_files(self, img_path: str | list[str]) -> list[str]:
                return sorted(str(Path(path).resolve()) for path in img_path)

            #Converts each ImageRecord into the label dict format Ultralytics' dataset expects
            def get_labels(self) -> list[dict[str, Any]]:
                labels = []
                for image_path in self.im_files:
                    record = self._records_by_path[str(Path(image_path).resolve())]
                    labels.append(
                        {
                            "im_file": image_path,
                            "shape": (record.height, record.width),
                            "cls": np.zeros((1, 1), dtype=np.float32),
                            "bboxes": np.asarray(
                                [
                                    _normalized_cxcywh(
                                        record.box, (record.height, record.width)
                                    )
                                ],
                                dtype=np.float32,
                            ),
                            "segments": [],
                            "keypoints": None,
                            "normalized": True,
                            "bbox_format": "xywh",
                            "image_id": record.image_id,
                        }
                    )
                return labels

            def set_epoch(self, epoch: int) -> None:
                self._epoch.fill_(epoch)

            #Same deterministic-per-sample seeding idea as YOLO11ProbeDataset, but Ultralytics'
            #own augmentations read Python/NumPy's global random state directly, so it has to be
            #swapped out and restored around this one call instead of passed in as an argument
            def __getitem__(self, index: int) -> dict[str, Any]:
                image_id = int(self.labels[index]["image_id"])
                with _NATIVE_AUGMENTATION_SEED_LOCK:
                    python_state = random.getstate()
                    numpy_state = np.random.get_state()
                    sample_seed = (
                        self.seed
                        + int(self._epoch.item()) * 1_000_003
                        + image_id * 97
                    ) % (2**32)
                    random.seed(sample_seed)
                    np.random.seed(sample_seed)
                    try:
                        sample = super().__getitem__(index)
                    finally:
                        random.setstate(python_state)
                        np.random.set_state(numpy_state)
                sample["image_id"] = image_id
                return sample

        return _Dataset()


def collate_yolo_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:

    return {
        "img": torch.stack([sample["img"] for sample in samples]),
        "cls": torch.cat([sample["cls"] for sample in samples]),
        "bboxes": torch.cat([sample["bboxes"] for sample in samples]),
        #Tags each box with which image in the batch it belongs to, so the loss knows
        #which prediction should be compared against which ground truth
        "batch_idx": torch.cat(
            [
                torch.full((len(sample["cls"]),), index, dtype=torch.float32)
                for index, sample in enumerate(samples)
            ]
        ),
        "image_id": [int(sample["image_id"]) for sample in samples],
    }


#Converts an absolute-pixel xyxy box into YOLO's expected relative center/width/height format
def _normalized_cxcywh(
    box: BoundingBox, input_size: int | tuple[int, int] = INPUT_SIZE
) -> tuple[float, float, float, float]:
    input_height, input_width = (
        (input_size, input_size) if isinstance(input_size, int) else input_size
    )
    return (
        (box.x1 + box.x2) / (2 * input_width),
        (box.y1 + box.y2) / (2 * input_height),
        box.width / input_width,
        box.height / input_height,
    )


#Inverse of the transform: converts a model-space box back to original image pixels
def _box_from_letterboxed_coordinates(
    coordinates: Sequence[float],
    original_width: int,
    original_height: int,
    input_size: int | tuple[int, int] = INPUT_SIZE,
) -> BoundingBox | None:
    input_height, input_width = (
        (input_size, input_size) if isinstance(input_size, int) else input_size
    )
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    padding_x = (input_width - resized_width) // 2
    padding_y = (input_height - resized_height) // 2

    x1, y1, x2, y2 = coordinates
    clipped = (
        max(0.0, min((x1 - padding_x) / scale, original_width)),
        max(0.0, min((y1 - padding_y) / scale, original_height)),
        max(0.0, min((x2 - padding_x) / scale, original_width)),
        max(0.0, min((y2 - padding_y) / scale, original_height)),
    )
    #A degenerate box after clipping (ex: zero area) is reported as no detection, not an error
    try:
        return BoundingBox(*clipped)
    except ValueError:
        return None


#Seeds every RNG source and disables cudnn's non-deterministic autotuning, for a reproducible run
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#Picks the training device
def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


#Downloads the official COCO checkpoint if missing, then checks its checksum, so training
#always starts from the same known, unmodified starting point
def _ensure_pretrained_checkpoint(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading official {MODEL_FAMILY}-{MODEL_VARIANT.upper()} COCO checkpoint "
            f"to {path}"
        )
        urllib.request.urlretrieve(PRETRAINED_CHECKPOINT_URL, path)

    checksum = sha256_file(path)
    if checksum != PRETRAINED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Unexpected {MODEL_FAMILY}-{MODEL_VARIANT.upper()} checkpoint SHA-256: "
            f"{checksum}; expected {PRETRAINED_CHECKPOINT_SHA256}"
        )
    return checksum


#Fails fast if the required libraries are missing or their versions don't match what this
#benchmark was verified against
def _validate_training_runtime() -> None:
    try:
        import torchvision
        import ultralytics
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"The YOLO11 training environment is incomplete: {error}. "
            f"Interpreter: {sys.executable}. Install requirements-yolo11.txt "
            "into this interpreter before training."
        ) from error
    versions = {
        "torch": (torch.__version__, TORCH_VERSION),
        "torchvision": (torchvision.__version__, TORCHVISION_VERSION),
        "ultralytics": (ultralytics.__version__, ULTRALYTICS_VERSION),
    }
    mismatches = [
        f"{name}={actual} (expected {expected})"
        for name, (actual, expected) in versions.items()
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError(
            "The YOLO11 benchmark requires the pinned runtime: "
            + ", ".join(mismatches)
            + f". Interpreter: {sys.executable}"
        )


#Builds a fresh single-class YOLO11 model and loads the pretrained weights into it
def _build_official_model(checkpoint_path: Path, device: torch.device) -> Any:
    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint
    from ultralytics.utils import DEFAULT_CFG

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Expected ultralytics {ULTRALYTICS_VERSION}, got {ultralytics.__version__}. "
            "Install requirements-yolo11.txt."
        )

    pretrained, _ = load_checkpoint(checkpoint_path, device="cpu")
    #Rebuilds the architecture from the pretrained checkpoint own config
    #(one class) instead of default 80 classes
    model = DetectionModel(pretrained.yaml, ch=3, nc=1, verbose=False)
    model.names = {0: "probe"}
    model.args = get_cfg(
        DEFAULT_CFG,
        overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5},
    )
    #Loads only the shape-compatible pretrained weights; the class-specific
    #head is left freshly initialized for this new single-class task
    model.load(pretrained)
    #Clears any cached loss so it gets rebuilt fresh for this run's settings
    model.criterion = None
    return model.to(device)


def _register_detection_head_input_dropout(model: Any, probability: float) -> Any:
    if probability == 0.0:
        return None
    detection_head = model.model[-1]

    #Runs right before the detection head each forward pass; only drops features while
    #training, never during validation or inference
    def drop_features(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if not module.training:
            return None
        features = inputs[0]
        if not isinstance(features, (list, tuple)):
            raise TypeError("YOLO detection head input must be a feature sequence")
        dropped = type(features)(
            F.dropout(feature, p=probability, training=True) for feature in features
        )
        return (dropped, *inputs[1:])

    return detection_head.register_forward_pre_hook(drop_features)


#Builds the optimizer, scaling weight decay and iteration count for the effective batch size
def _build_official_optimizer(
    model: Any,
    *,
    name: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    nominal_batch_size: int,
    epochs: int,
    training_size: int,
) -> tuple[Any, int]:
    from ultralytics.engine.trainer import BaseTrainer

    #How many small batches get accumulated to simulate one bigger "nominal" batch
    accumulation = max(round(nominal_batch_size / batch_size), 1)
    #Scales weight decay to match the real batch size actually used, standard practice
    scaled_decay = weight_decay * batch_size * accumulation / nominal_batch_size
    iterations = math.ceil(training_size / max(batch_size, nominal_batch_size)) * epochs
    #A stand-in object: Ultralytics' optimizer builder expects a full Trainer instance
    #but only actually reads these few fields off it
    context = SimpleNamespace(
        data={"nc": 1},
        args=SimpleNamespace(
            lr0=learning_rate,
            momentum=momentum,
            warmup_bias_lr=0.1,
        ),
    )
    optimizer = BaseTrainer.build_optimizer(
        context,
        model,
        name=name,
        lr=learning_rate,
        momentum=momentum,
        decay=scaled_decay,
        iterations=iterations,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer, accumulation


#Moves every tensor in a batch to the training device, leaving other values untouched
def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


#Linearly ramps the learning rate (and momentum) up from a small starting value during the
#first warmup steps, a standard technique to stabilize the very start of training
def _apply_warmup(
    optimizer: Any,
    *,
    global_step: int,
    warmup_steps: int,
    epoch_factor: float,
    warmup_momentum: float,
    momentum: float,
    warmup_bias_lr: float,
) -> None:
    if global_step >= warmup_steps or warmup_steps == 0:
        return
    fraction = global_step / warmup_steps
    for group in optimizer.param_groups:
        #Bias parameters start warmup from a higher learning rate than the rest, standard YOLO practice
        starting_lr = warmup_bias_lr if group.get("param_group") == "bias" else 0.0
        target_lr = group["initial_lr"] * epoch_factor
        group["lr"] = starting_lr + fraction * (target_lr - starting_lr)
        if "momentum" in group:
            group["momentum"] = warmup_momentum + fraction * (
                momentum - warmup_momentum
            )


#Runs one full pass over the training data with gradient accumulation and mixed precision,
#returning the average loss and how many real optimizer steps actually happened
def _train_one_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: Any,
    ema: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    accumulation: int,
    warmup_steps: int,
    learning_rate_factor: float,
    warmup_momentum: float,
    momentum: float,
    warmup_bias_lr: float,
) -> tuple[float, int]:
    model.train()
    dataset = loader.dataset
    #Only the shared-augmentation dataset needs the epoch number for its per-sample seeding
    if isinstance(dataset, YOLO11ProbeDataset):
        dataset.set_epoch(epoch)

    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    amp_enabled = scaler.is_enabled()
    pending_batches = 0
    optimizer_steps = 0
    for step, batch in enumerate(loader):
        global_step = epoch * len(loader) + step
        _apply_warmup(
            optimizer,
            global_step=global_step,
            warmup_steps=warmup_steps,
            epoch_factor=learning_rate_factor,
            warmup_momentum=warmup_momentum,
            momentum=momentum,
            warmup_bias_lr=warmup_bias_lr,
        )
        batch = _move_batch(batch, device)
        #Pixels are normalized per-batch on the device, not upfront, to save memory
        batch["img"] = batch["img"].float().div_(255.0)

        #Runs the forward pass and loss in mixed precision when enabled
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss_components, loss_items = model(batch)
            loss = loss_components.sum()
        #Catches a diverged/broken run immediately instead of continuing on bad numbers
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}, step {step}")

        #Scales the loss before backprop so small gradients don't underflow to zero in fp16
        scaler.scale(loss).backward()
        pending_batches += 1
        #Gradient accumulation: only actually step the optimizer every few batches, not every one
        should_step = pending_batches >= accumulation or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            #The scaler silently skips the step if it detects overflowed gradients (scale shrinks);
            #only count it as a real step when that didn't happen
            if scaler.get_scale() >= scale_before_step:
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
            #Updates the exponential-moving-average weights, which are what actually get validated/saved
            ema.update(model)
            pending_batches = 0

        batch_loss = sum(float(value.detach()) for value in loss_items.values())
        loss_total += batch_loss
        #Progress logging every 25 steps and at the start/end of the epoch
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == len(loader):
            print(
                f"Epoch {epoch + 1}/{epochs} step {step + 1}/{len(loader)} "
                f"loss={batch_loss:.4f}"
            )
    #If every single step was skipped, something is fundamentally broken, not a normal epoch
    if optimizer_steps == 0:
        raise RuntimeError(
            "Every optimizer update was skipped because AMP detected non-finite "
            "gradients; retry with --no-amp or inspect the loss scale"
        )
    return loss_total / len(loader), optimizer_steps


#Runs the model over a validation loader, scores its predictions, and optionally also
#computes validation loss (used only as a diagnostic signal for early stopping)
@torch.inference_mode()
def _validation_pass(
    model: Any,
    loader: DataLoader,
    records: Sequence[ImageRecord],
    device: torch.device,
    confidence_threshold: float,
    matching_iou_threshold: float,
    nms_iou_threshold: float,
    input_size: int | tuple[int, int] = INPUT_SIZE,
    *,
    calculate_loss: bool = False,
    max_loss_batches: int | None = None,
) -> tuple[Any, float | None]:
    from ultralytics.utils.nms import non_max_suppression

    model.eval()
    #Lets predictions below be matched back to the right record for coordinate restoration
    records_by_id = {record.image_id: record for record in records}
    predictions: list[Prediction] = []
    loss_total = 0.0
    loss_batches = 0

    for batch_index, batch in enumerate(loader):
        image_ids = batch["image_id"]
        batch = _move_batch(batch, device)
        images = batch["img"].float().div_(255.0)
        batch["img"] = images
        raw_predictions = model(images)
        #Loss is only a diagnostic for early stopping, so it's capped to a limited number of batches
        if calculate_loss and (
            max_loss_batches is None or batch_index < max_loss_batches
        ):
            _, loss_items = model.loss(batch, preds=raw_predictions)
            batch_loss = sum(float(value.detach()) for value in loss_items.values())
            if not math.isfinite(batch_loss):
                raise RuntimeError("Non-finite validation loss")
            loss_total += batch_loss
            loss_batches += 1
        #Deliberately permissive here (very low confidence floor, many detections kept); the
        #real confidence threshold is applied later, inside evaluate_detections
        detections = non_max_suppression(
            raw_predictions,
            conf_thres=0.001,
            iou_thres=nms_iou_threshold,
            classes=[0],
            max_det=100,
            nc=1,
            max_time_img=10.0,
        )

        for image_id, image_detections in zip(image_ids, detections):
            record = records_by_id[image_id]
            for detection in image_detections:
                box = _box_from_letterboxed_coordinates(
                    detection[:4].detach().cpu().tolist(),
                    record.width,
                    record.height,
                    input_size,
                )
                if box is not None:
                    predictions.append(
                        Prediction(
                            image_id=image_id,
                            box=box,
                            confidence=float(detection[4].detach().cpu()),
                        )
                    )

    result = evaluate_detections(
        ground_truth_from_records(records), predictions,
        confidence_threshold=confidence_threshold,
        matching_iou_threshold=matching_iou_threshold,
    )
    if calculate_loss and loss_batches == 0:
        raise ValueError("Validation loss requires at least one batch")
    return result, loss_total / loss_batches if calculate_loss else None


#Thin wrapper around _validation_pass for the common case where validation loss isn't needed
def _validate(
    model: Any,
    loader: DataLoader,
    records: Sequence[ImageRecord],
    device: torch.device,
    confidence_threshold: float,
    matching_iou_threshold: float,
    nms_iou_threshold: float,
    input_size: int | tuple[int, int] = INPUT_SIZE,
):
    result, _ = _validation_pass(
        model, loader, records, device, confidence_threshold,
        matching_iou_threshold, nms_iou_threshold, input_size,
    )
    return result


#Saves everything needed to fully resume training later, plus the run's own configuration
def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: Any,
    ema: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    best_map50_95: float | None,
    run_config: dict[str, Any],
) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.ema.state_dict(),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_validation_map50_95": best_map50_95,
        "selection_metric": "validation_map50_95",
        "run_config": run_config,
    }
    torch.save(state, path)


def _save_deferred_validation_checkpoint(
    path: Path,
    *,
    epoch: int,
    ema: Any,
    run_config: dict[str, Any],
) -> None:
    """Save only the full-precision EMA weights needed for post-hoc validation."""

    #Lighter than _save_checkpoint: these never need to resume training, only be evaluated later
    torch.save(
        {
            "epoch": epoch,
            "ema": ema.ema.state_dict(),
            "ema_updates": ema.updates,
            "best_validation_map50_95": None,
            "selection_metric": "validation_map50_95_pending",
            "checkpoint_role": "deferred_validation_candidate",
            "run_config": run_config,
        },
        path,
    )


#Checks that a resumed run's settings match the checkpoint's original settings, so a run can't
#silently continue with different, now-inconsistent hyperparameters
def _validate_resume_configuration(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    required = ("model", "ema", "optimizer", "lr_scheduler", "scaler", "run_config")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Resume checkpoint is missing training state: {missing}")
    start_epoch = int(checkpoint["epoch"])
    if args.epochs <= start_epoch:
        raise ValueError(
            f"--epochs is the total target and must exceed resumed epoch {start_epoch}"
        )
    previous = dict(checkpoint["run_config"])
    #Compares every simple setting directly between the checkpoint and the current command
    scalar_fields = (
        "batch_size", "validation_batch_size", "learning_rate",
        "optimizer", "momentum", "weight_decay", "dropout", "warmup_epochs",
        "nominal_batch_size", "augmentation_policy", "seed", "defer_validation",
        "amp",
    )
    mismatches = []
    for field in scalar_fields:
        requested = getattr(args, field)
        if previous.get(field) != requested:
            mismatches.append(
                f"{field}: checkpoint={previous.get(field)!r}, command={requested!r}"
            )
    #Compares the schedule type using the same name translation used when it was first recorded
    previous_schedule = str(previous["learning_rate_schedule"]["type"])
    requested_schedule = (
        "validation_map_plateau"
        if args.learning_rate_schedule == "plateau"
        else "linear"
    )
    if previous_schedule != requested_schedule:
        mismatches.append(
            f"learning_rate_schedule: checkpoint={previous_schedule!r}, "
            f"command={requested_schedule!r}"
        )
    #Compares by resolved absolute path so equivalent paths spelled differently still match
    for field in ("pretrained_checkpoint", "checkpoint_dir", "train_split", "val_split"):
        previous_path = Path(str(previous[field])).resolve()
        requested_path = Path(getattr(args, field)).resolve()
        if previous_path != requested_path:
            mismatches.append(
                f"{field}: checkpoint={previous_path}, command={requested_path}"
            )
    requested_height = args.input_size if args.input_height is None else args.input_height
    requested_width = args.input_size if args.input_width is None else args.input_width
    if int(previous["input_height"]) != requested_height:
        mismatches.append("input_height differs from the checkpoint")
    if int(previous["input_width"]) != requested_width:
        mismatches.append("input_width differs from the checkpoint")
    #Reports every mismatch at once, not just the first one found
    if mismatches:
        raise ValueError("Resume configuration mismatch: " + "; ".join(mismatches))
    return start_epoch, previous


#Builds one CSV row for an epoch when validation was skipped (deferred-validation mode)
def _training_only_metrics_row(
    epoch: int,
    train_loss: float,
    optimizer_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "optimizer_steps": optimizer_steps,
        "learning_rate": learning_rate,
    }


#Builds one CSV row combining training and validation metrics for a normal training epoch
def _metrics_row(
    epoch: int,
    train_loss: float,
    optimizer_steps: int,
    learning_rate: float,
    result: Any,
    validation_loss: float | None = None,
) -> dict[str, Any]:
    overall = result.overall
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "optimizer_steps": optimizer_steps,
        "learning_rate": learning_rate,
        "validation_loss": validation_loss if validation_loss is not None else "",
        "val_map50_95": overall.map50_95,
        "val_precision": overall.precision,
        "val_recall": overall.recall,
        "val_f1": overall.f1,
        "val_mean_iou": overall.mean_iou,
    }


#Defines every training CLI flag and its default; most have their own help= text below
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("probe_images"))
    parser.add_argument("--annotations", type=Path, default=Path("probe_labels.json"))
    parser.add_argument("--train-split", type=Path, default=Path("splits/train.txt"))
    parser.add_argument("--val-split", type=Path, default=Path("splits/val.txt"))
    parser.add_argument(
        "--pretrained-checkpoint", type=Path
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Restore model, EMA, optimizer, scheduler, and AMP state and continue "
            "until the total epoch count given by --epochs."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        help="Validation-only batch size; defaults to the training batch size.",
    )
    #--input-size sets a square size; --input-width/--input-height together override it with a
    #rectangular one (both required if either is given, checked later in main)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--input-height", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("linear", "plateau"),
        default="linear",
        help="Learning-rate schedule; plateau responds to validation mAP50:95.",
    )
    parser.add_argument("--lr-plateau-factor", type=float, default=0.8)
    parser.add_argument("--lr-plateau-patience", type=int, default=20)
    parser.add_argument("--lr-plateau-cooldown", type=int, default=5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=("AdamW", "SGD"), default="AdamW")
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help=(
            "Training-only elementwise dropout probability applied to the "
            "multiscale feature maps entering the Detect head."
        ),
    )
    parser.add_argument("--final-learning-rate-fraction", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-momentum", type=float, default=0.8)
    parser.add_argument("--warmup-bias-learning-rate", type=float, default=0.1)
    #The "effective" batch size gradients are accumulated up to before each optimizer step
    parser.add_argument("--nominal-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decode images once into shared CPU memory for reuse across epochs.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--matching-iou-threshold", type=float, default=0.50)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.70)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without higher validation mAP50:95; 0 disables it.",
    )
    parser.add_argument(
        "--validation-loss-batches",
        type=int,
        help="Maximum deterministic validation batches used for early stopping.",
    )
    parser.add_argument(
        "--defer-validation",
        action="store_true",
        help=(
            "Skip all validation during training and save one evaluation-only EMA "
            "checkpoint per epoch for post-hoc validation selection."
        ),
    )
    parser.add_argument(
        "--epoch-checkpoint-dir",
        type=Path,
        help=(
            "Directory for deferred-validation epoch candidates; defaults to "
            "CHECKPOINT_DIR/epoch_candidates."
        ),
    )
    parser.add_argument(
        "--augmentation-policy",
        choices=("shared", "yolo"),
        default="shared",
        help="Training augmentation implementation; validation remains deterministic.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


#CLI entry point: validates every setting, builds the model/data/optimizer, then runs training
def main() -> None:
    args = _parse_args()
    _validate_training_runtime()
    if args.pretrained_checkpoint is None:
        args.pretrained_checkpoint = Path(PRETRAINED_CHECKPOINT)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(f"checkpoints/{CHECKPOINT_PREFIX}_{MODEL_VARIANT}")
    #The rest of this block checks every argument is in a sane range before doing any real work
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")
    if args.validation_batch_size is not None and args.validation_batch_size <= 0:
        raise ValueError("Validation batch size must be positive")
    if (args.input_width is None) != (args.input_height is None):
        raise ValueError("--input-width and --input-height must be provided together")
    #Falls back to a square size from --input-size unless a rectangular one was explicitly given
    input_height, input_width = (
        (args.input_size, args.input_size)
        if args.input_width is None
        else (args.input_height, args.input_width)
    )
    #Multiples of 32 are required so the resolution stays compatible with the model's stride
    if any(size <= 0 or size % 32 for size in (input_height, input_width)):
        raise ValueError("Input width and height must be positive multiples of 32")
    #Collapses to a plain int when square, since most of this file's functions accept either form
    input_size: int | tuple[int, int] = (
        input_height if input_height == input_width else (input_height, input_width)
    )
    if not 0.0 < args.final_learning_rate_fraction <= 1.0:
        raise ValueError("Final learning-rate fraction must be in (0, 1]")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("Dropout probability must be in [0, 1)")
    if not 0.0 < args.lr_plateau_factor < 1.0:
        raise ValueError("LR plateau factor must be in (0, 1)")
    if args.lr_plateau_patience < 0 or args.lr_plateau_cooldown < 0:
        raise ValueError("LR plateau patience and cooldown cannot be negative")
    if not 0.0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError(
            "Minimum learning rate must be positive and no greater than the initial rate"
        )
    if args.nominal_batch_size <= 0 or args.warmup_epochs < 0:
        raise ValueError("Nominal batch size must be positive and warmup cannot be negative")
    if args.num_workers < 0:
        raise ValueError("Number of data-loader workers cannot be negative")
    if args.early_stopping_patience < 0:
        raise ValueError("Early-stopping patience cannot be negative")
    if args.validation_loss_batches is not None and args.validation_loss_batches <= 0:
        raise ValueError("Validation-loss batches must be positive")
    if args.defer_validation and args.learning_rate_schedule != "linear":
        raise ValueError("Deferred validation requires the validation-independent linear schedule")
    if args.defer_validation and args.early_stopping_patience:
        raise ValueError("Deferred validation is incompatible with early stopping")
    if args.defer_validation and args.validation_loss_batches is not None:
        raise ValueError("Deferred validation cannot calculate validation loss")
    if args.epoch_checkpoint_dir is not None and not args.defer_validation:
        raise ValueError("--epoch-checkpoint-dir requires --defer-validation")

    #If resuming, load the old checkpoint now and check its settings match this run's
    resume_checkpoint = None
    start_epoch = 0
    previous_run_config = None
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(args.resume_checkpoint)
        resume_checkpoint = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        start_epoch, previous_run_config = _validate_resume_configuration(
            args, resume_checkpoint
        )

    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    checkpoint_sha256 = _ensure_pretrained_checkpoint(args.pretrained_checkpoint)

    all_records = load_dataset(args.images, args.annotations)
    train_records = load_split_records(all_records, args.train_split)
    val_records = load_split_records(all_records, args.val_split)
    #Makes sure no image accidentally ended up in both splits
    if {record.image_id for record in train_records} & {
        record.image_id for record in val_records
    }:
        raise ValueError("Training and validation splits overlap")

    #Picks the shared (Albumentations) or native (Ultralytics) augmentation dataset per policy
    train_dataset = (
        YOLO11ProbeDataset(
            train_records,
            training=True,
            seed=args.seed,
            input_size=input_size,
            cache_images=args.cache_images,
        )
        if args.augmentation_policy == "shared"
        else YOLO11NativeAugmentationDataset(
            train_records,
            seed=args.seed,
            input_size=input_size,
            batch_size=args.batch_size,
        )
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_yolo_batch,
        generator=generator,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if not args.defer_validation:
        val_dataset = YOLO11ProbeDataset(
            val_records,
            training=False,
            seed=args.seed,
            input_size=input_size,
            cache_images=args.cache_images,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.validation_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_yolo_batch,
            drop_last=False,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    #Builds the model, optionally with feature dropout, plus a matching optimizer
    model = _build_official_model(args.pretrained_checkpoint, device)
    _register_detection_head_input_dropout(model, args.dropout)
    optimizer, accumulation = _build_official_optimizer(
        model,
        name=args.optimizer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        nominal_batch_size=args.nominal_batch_size,
        epochs=args.epochs,
        training_size=len(train_records),
    )
    from ultralytics.utils.torch_utils import ModelEMA

    ema = ModelEMA(model)
    #When resuming, the schedule still targets the originally planned epoch count, not a new one
    schedule_horizon_epochs = (
        int(previous_run_config["epochs"])
        if previous_run_config is not None
        else args.epochs
    )
    #Two schedule choices: plateau reacts to validation mAP, linear just decays over the run
    if args.learning_rate_schedule == "plateau":
        learning_rate_function = lambda epoch: 1.0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            cooldown=args.lr_plateau_cooldown,
            min_lr=args.minimum_learning_rate,
        )
    else:
        learning_rate_function = lambda epoch: max(
            1 - epoch / schedule_horizon_epochs, 0
        ) * (
            1.0 - args.final_learning_rate_fraction
        ) + args.final_learning_rate_fraction
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=learning_rate_function
        )
    amp_enabled = args.amp and device.type == "cuda"
    # YOLO's loss is summed across the effective (accumulated) batch. A lower
    # initial scale prevents the first nominal-batch updates from overflowing.
    amp_initial_scale = 128.0
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled, init_scale=amp_initial_scale
    )
    #Restores every piece of state so training continues exactly where it left off
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        ema.ema.load_state_dict(resume_checkpoint["ema"], strict=True)
        ema.updates = int(resume_checkpoint["ema_updates"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["lr_scheduler"])
        scaler.load_state_dict(resume_checkpoint["scaler"])
        print(
            f"Resuming complete training state from epoch {start_epoch}; "
            f"target epoch={args.epochs}"
        )
    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
    warmup_steps = round(warmup_epochs * len(train_loader))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    #Records exactly which augmentation settings were used, for the saved run_config below
    if args.augmentation_policy == "yolo":
        training_augmentation = {
            "implementation": "ultralytics",
            "rectangular_training": True,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "blur_probability": 0.01,
            "median_blur_probability": 0.01,
            "grayscale_probability": 0.01,
            "clahe_probability": 0.01,
        }
    else:
        training_augmentation = {"implementation": "shared"}

    epoch_checkpoint_dir = args.epoch_checkpoint_dir or (
        args.checkpoint_dir / "epoch_candidates"
    )
    #A resumed run keeps recording the original schedule, not a newly built one
    if previous_run_config is not None:
        schedule_record = previous_run_config["learning_rate_schedule"]
    elif args.learning_rate_schedule == "plateau":
        schedule_record = {
            "type": "validation_map_plateau",
            "initial": args.learning_rate,
            "factor": args.lr_plateau_factor,
            "patience": args.lr_plateau_patience,
            "cooldown": args.lr_plateau_cooldown,
            "minimum": args.minimum_learning_rate,
            "warmup_epochs": warmup_epochs,
        }
    else:
        schedule_record = {
            "type": "linear",
            "initial": args.learning_rate,
            "final": args.learning_rate * args.final_learning_rate_fraction,
            "warmup_epochs": warmup_epochs,
        }
    #Records every setting used for this run, so it can be audited or exactly reproduced later.
    #Starts from every CLI argument, then adds computed/derived values below
    run_config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_device": str(device),
        "input_height": input_height,
        "input_width": input_width,
        "variant": MODEL_VARIANT,
        "architecture": f"{MODEL_FAMILY}-{MODEL_VARIANT.upper()}",
        "ultralytics_version": ULTRALYTICS_VERSION,
        "pretrained_checkpoint_url": PRETRAINED_CHECKPOINT_URL,
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "train_images": len(train_records),
        "validation_images": len(val_records),
        "gradient_accumulation": accumulation,
        "amp_initial_scale": amp_initial_scale if amp_enabled else None,
        "test_split_used": False,
        "validation_during_training": not args.defer_validation,
        "deferred_validation": {
            "enabled": args.defer_validation,
            "candidate_checkpoint_dir": (
                str(epoch_checkpoint_dir) if args.defer_validation else None
            ),
            "candidate_checkpoint_format": (
                "evaluation_only_full_precision_ema" if args.defer_validation else None
            ),
            "selection_metric": "validation_map50_95",
        },
        "native_ultralytics_augmentations_used": args.augmentation_policy == "yolo",
        "learning_rate_schedule": schedule_record,
        "training_augmentation": training_augmentation,
        "training_regularization": {
            "detection_head_input_dropout": args.dropout,
            "implementation": "elementwise feature dropout before Detect; training only",
        },
        "resume": (
            {
                "source_checkpoint": str(args.resume_checkpoint.resolve()),
                "from_epoch": start_epoch,
                "target_epoch": args.epochs,
                "restored_state": [
                    "model", "ema", "optimizer", "lr_scheduler", "amp_scaler"
                ],
                "schedule_horizon_epochs": schedule_horizon_epochs,
                "continued_lr_policy": "hold original linear schedule floor",
                "rng_note": (
                    "The source checkpoint predates RNG-state persistence; the "
                    "data-loader sequence restarts deterministically from the run seed."
                ),
            }
            if resume_checkpoint is not None
            else None
        ),
        "early_stopping": {
            "enabled": args.early_stopping_patience > 0,
            "criterion": "maximum full-validation mAP50:95",
            "patience": args.early_stopping_patience,
            "validation_loss_batches": args.validation_loss_batches,
            "validation_loss_role": "diagnostic only",
        },
    }
    (args.checkpoint_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    #Refuses to silently overwrite deferred-validation checkpoints from a previous run
    if args.defer_validation:
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        collisions = [
            epoch_checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
            for epoch in range(start_epoch, args.epochs)
            if (epoch_checkpoint_dir / f"epoch_{epoch + 1:04d}.pt").exists()
        ]
        if collisions:
            raise FileExistsError(
                f"Refusing to overwrite deferred candidates: {collisions[:3]}"
            )

    metrics_path = args.checkpoint_dir / "metrics.csv"
    best_map50_95 = float("-inf")
    epochs_without_improvement = 0
    early_stopping = args.early_stopping_patience > 0
    #When resuming, append to the existing metrics.csv instead of starting a new one,
    #after checking it actually ends at the epoch being resumed from
    if resume_checkpoint is not None:
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Resume metrics file is missing: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8", newline="") as existing_file:
            existing_rows = list(csv.DictReader(existing_file))
        if not existing_rows or int(existing_rows[-1]["epoch"]) != start_epoch:
            raise ValueError(
                f"Metrics history does not end at resumed epoch {start_epoch}"
            )
        metrics_mode = "a"
        existing_fields = list(existing_rows[-1])
    else:
        metrics_mode = "w"
        existing_fields = None
    with metrics_path.open(metrics_mode, encoding="utf-8", newline="") as metrics_file:
        writer: csv.DictWriter[str] | None = (
            csv.DictWriter(metrics_file, fieldnames=existing_fields)
            if existing_fields is not None
            else None
        )
        #The main training loop: one epoch of training, then (unless deferred) one validation pass
        for epoch in range(start_epoch, args.epochs):
            train_loss, optimizer_steps = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                ema,
                scaler,
                device,
                epoch=epoch,
                epochs=args.epochs,
                accumulation=accumulation,
                warmup_steps=warmup_steps,
                learning_rate_factor=learning_rate_function(epoch),
                warmup_momentum=args.warmup_momentum,
                momentum=args.momentum,
                warmup_bias_lr=args.warmup_bias_learning_rate,
            )
            #Keeps the EMA copy's non-weight attributes (like stride) in sync with the live model
            ema.update_attr(
                model, include=("yaml", "nc", "args", "names", "stride")
            )
            #Three modes: skip validation entirely, validate and also track loss for early
            #stopping, or validate without the extra loss computation
            if args.defer_validation:
                row = _training_only_metrics_row(
                    epoch + 1,
                    train_loss,
                    optimizer_steps,
                    optimizer.param_groups[-1]["lr"],
                )
                result = None
            elif early_stopping:
                assert val_loader is not None
                result, validation_loss = _validation_pass(
                    ema.ema,
                    val_loader,
                    val_records,
                    device,
                    args.confidence_threshold,
                    args.matching_iou_threshold,
                    args.nms_iou_threshold,
                    input_size,
                    calculate_loss=True,
                    max_loss_batches=args.validation_loss_batches,
                )
                assert validation_loss is not None
                row = _metrics_row(
                    epoch + 1,
                    train_loss,
                    optimizer_steps,
                    optimizer.param_groups[-1]["lr"],
                    result,
                    validation_loss=validation_loss,
                )
            else:
                assert val_loader is not None
                result = _validate(
                    ema.ema,
                    val_loader,
                    val_records,
                    device,
                    args.confidence_threshold,
                    args.matching_iou_threshold,
                    args.nms_iou_threshold,
                    input_size,
                )
                row = _metrics_row(
                    epoch + 1,
                    train_loss,
                    optimizer_steps,
                    optimizer.param_groups[-1]["lr"],
                    result,
                )
            #Creates the CSV header from the first row's own keys, since the columns differ
            #depending on which of the three modes above ran
            if writer is None:
                writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            metrics_file.flush()

            improved = (
                result is not None and result.overall.map50_95 > best_map50_95
            )
            #Deferred mode saves every epoch as a candidate; otherwise only a new best gets saved
            if args.defer_validation:
                _save_deferred_validation_checkpoint(
                    epoch_checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                    epoch=epoch + 1,
                    ema=ema,
                    run_config=run_config,
                )
            elif improved:
                best_map50_95 = result.overall.map50_95
                if early_stopping:
                    epochs_without_improvement = 0
                _save_checkpoint(
                    args.checkpoint_dir / "best.pt",
                    epoch=epoch + 1,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_map50_95=best_map50_95,
                    run_config=run_config,
                )
            elif early_stopping:
                epochs_without_improvement += 1
            #The plateau schedule needs this epoch's validation score to decide whether to react
            if args.learning_rate_schedule == "plateau":
                assert result is not None
                scheduler.step(result.overall.map50_95)
            #Always overwrites "last.pt" so training can resume from the most recent epoch
            _save_checkpoint(
                args.checkpoint_dir / "last.pt",
                epoch=epoch + 1,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_map50_95=(None if args.defer_validation else best_map50_95),
                run_config=run_config,
            )
            if args.defer_validation:
                print(
                    f"Training epoch {epoch + 1}: loss={train_loss:.4f}, "
                    f"lr={optimizer.param_groups[-1]['lr']:.8f}; validation deferred"
                )
            elif early_stopping:
                print(
                    f"Validation epoch {epoch + 1}: loss={validation_loss:.4f}, "
                    f"mAP50:95={result.overall.map50_95:.4f}, "
                    f"best_mAP50:95={best_map50_95:.4f}, "
                    f"lr={optimizer.param_groups[-1]['lr']:.8f}, "
                    f"patience={epochs_without_improvement}/"
                    f"{args.early_stopping_patience}, "
                    f"precision={result.overall.precision:.4f}, "
                    f"recall={result.overall.recall:.4f}, "
                    f"F1={result.overall.f1:.4f}, "
                    f"IoU={result.overall.mean_iou:.4f}"
                )
            else:
                print(
                    f"Validation epoch {epoch + 1}: "
                    f"mAP50:95={result.overall.map50_95:.4f}, "
                    f"precision={result.overall.precision:.4f}, "
                    f"recall={result.overall.recall:.4f}"
                )
            #The linear schedule just advances every epoch, independent of validation results
            if args.learning_rate_schedule == "linear":
                scheduler.step()
            if early_stopping and (
                epochs_without_improvement >= args.early_stopping_patience
            ):
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if args.defer_validation:
        print(f"Deferred-validation candidates: {epoch_checkpoint_dir}")
        print("No best checkpoint exists until post-hoc validation selection completes.")
    else:
        print(f"Best validation mAP50:95: {best_map50_95:.4f}")
    print(f"Checkpoints and metrics: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
