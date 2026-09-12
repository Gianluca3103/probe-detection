"""Shared image-space augmentation for probe detection models."""

from __future__ import annotations

import random
import threading
from typing import Any, Protocol

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from albumentations import Compose

from .probe_types import BoundingBox

TARGET_SIZE = (416, 640)  # height, width - if no size is given it defaults to the fixed 640x416 model input
_MAX_BOX_ATTEMPTS = 10  # Maximum attempts to create a valid augmentation
# Makes sure that augmentation seeds don't interfere with each other global random state
_SEED_LOCK = threading.Lock()

# Defines a transformation of an argument to a dictonary
class _Transform(Protocol):
    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        ...

# Creates deterministic Sequence of seeds that will be used to create the augmentations
class _SeededCompose:
    """Give Albumentations 1.x an isolated, reproducible random sequence."""

    def __init__(self, transform: A.Compose, seed: int | None) -> None:
        self._transform = transform
        self._seeds = random.Random(seed) if seed is not None else None

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if self._seeds is None:
            return self._transform(**kwargs)

        invocation_seed = self._seeds.randrange(2**32)
        with _SEED_LOCK:
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            random.seed(invocation_seed)
            np.random.seed(invocation_seed)
            try:
                return self._transform(**kwargs)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

# Tells Albumentations how to interpret Bbox
def _bbox_params() -> A.BboxParams:
    return A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_area=1.0,
        min_width=1.0,
        min_height=1.0,
    )

# Transforms an image and its bbox into a model input size without stretching the image
class _Letterbox(A.DualTransform):
    """Resize to fit a target canvas, then add centered constant padding."""

    def __init__(self, height: int, width: int) -> None:
        super().__init__(always_apply=True, p=1.0)
        self.height = height
        self.width = width
    # Calculates values necessary to keep aspect ratio
    def _geometry(self, rows: int, cols: int) -> tuple[float, int, int, int, int]:
        scale = min(self.width / cols, self.height / rows)
        resized_width = round(cols * scale)
        resized_height = round(rows * scale)
        padding_left = (self.width - resized_width) // 2
        padding_top = (self.height - resized_height) // 2
        return scale, resized_width, resized_height, padding_left, padding_top

    # Adds padding to keep the correct aspect ratio
    # The goal is to keep the tensor size compatible with the model stride without distoring the image
    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        _, resized_width, resized_height, padding_left, padding_top = self._geometry(
            *image.shape[:2]
        )
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        return cv2.copyMakeBorder(
            resized,
            padding_top,
            self.height - resized_height - padding_top,
            padding_left,
            self.width - resized_width - padding_left,
            cv2.BORDER_CONSTANT,
            value=0,
        )

    # Apply the same resize and padding to the bounding box to keep the image bbox ratio the same
    def apply_to_bbox(
        self,
        bbox: tuple[float, float, float, float],
        rows: int = 0,
        cols: int = 0,
        **params: Any,
    ) -> tuple[float, float, float, float]:
        scale, _, _, padding_left, padding_top = self._geometry(rows, cols)
        x1, y1, x2, y2 = bbox
        return (
            (x1 * cols * scale + padding_left) / self.width,
            (y1 * rows * scale + padding_top) / self.height,
            (x2 * cols * scale + padding_left) / self.width,
            (y2 * rows * scale + padding_top) / self.height,
        )

    def apply_to_keypoint(self, keypoint: tuple[float, ...], **params: Any) -> tuple[float, ...]:
        raise NotImplementedError("Letterboxing keypoints is not supported")

    # Tells albumentation which constructors define the transformation
    def get_transform_init_args_names(self) -> tuple[str, str]:
        return ("height", "width")

# Converts target size into height width tuple
def _target_dimensions(target_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(target_size, int):
        return target_size, target_size
    return target_size

# Create the letterbox transform for the requested target dimensions
def _letterbox(target_size: int | tuple[int, int]) -> list[A.BasicTransform]:
    target_height, target_width = _target_dimensions(target_size)
    return [
        _Letterbox(target_height, target_width),
    ]

# When called apply random augmentations to the sample during training
def build_training_transform(
    seed: int | None = None,
    target_size: int | tuple[int, int] = TARGET_SIZE,
) -> _Transform:
    """Build the shared stochastic training transform.

    Two transforms created with the same seed produce the same sequence of
    augmented samples when called in the same order.
    """

    transform = A.Compose(
        [
            *_letterbox(target_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                rotate=(-10.0, 10.0),
                translate_percent=0.0,
                shear=0.0,
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                fit_output=False,
                keep_ratio=True,
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.5,
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.0), p=0.1),
            A.MotionBlur(blur_limit=(3, 5), p=0.1),
            A.GaussNoise(var_limit=(5.0, 20.0), mean=0, p=0.1),
        ],
        bbox_params=_bbox_params(),
    )
    return _SeededCompose(transform, seed)

#  given the imagize size defined by letterbox applies the same padding to bbox
def build_evaluation_transform(
    target_size: int | tuple[int, int] = TARGET_SIZE,
) -> Compose:

    return A.Compose(
        _letterbox(target_size),
        bbox_params=_bbox_params(),
    )

#Apply an image transform and return its synchronized, valid box
def apply_detection_transform(
    transform: _Transform,
    image: Image.Image,
    box: BoundingBox,
) -> tuple[Image.Image, BoundingBox]:

    box.validate_within(*image.size)
    image_array = np.asarray(image.convert("RGB"))

    for _ in range(_MAX_BOX_ATTEMPTS):
        result = transform(
            image=image_array,
            bboxes=[(box.x1, box.y1, box.x2, box.y2)],
            labels=[0],
        )
        transformed_boxes = result["bboxes"]
        if not transformed_boxes:
            continue
        if len(transformed_boxes) != 1:
            raise RuntimeError(
                f"Expected one transformed bounding box, got {len(transformed_boxes)}"
            )

        transformed_image = result["image"]
        height, width = transformed_image.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in transformed_boxes[0])
        clipped = (
            max(0.0, min(x1, width)),
            max(0.0, min(y1, height)),
            max(0.0, min(x2, width)),
            max(0.0, min(y2, height)),
        )

        try:
            transformed_box = BoundingBox(*clipped)
        except ValueError:
            continue
        transformed_box.validate_within(width, height)
        return Image.fromarray(transformed_image), transformed_box

    raise RuntimeError(
        f"Augmentation removed the only probe box after {_MAX_BOX_ATTEMPTS} attempts"
    )
