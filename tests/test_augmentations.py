from __future__ import annotations

import unittest

import albumentations as A
import numpy as np
from PIL import Image

from data_loading.augmentations import (
    apply_detection_transform,
    build_evaluation_transform,
    build_training_transform,
)
from data_loading.probe_types import BoundingBox


def _compose(*transforms: A.BasicTransform) -> A.Compose:
    return A.Compose(
        list(transforms),
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),
    )


def _test_image() -> Image.Image:
    y, x = np.mgrid[:48, :64]
    pixels = np.stack((x * 3, y * 5, (x + y) * 2), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels)


class AugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = _test_image()
        self.box = BoundingBox(8, 10, 28, 34)

    #Verifies a horizontal flip mirrors the box's x-coordinates while leaving y untouched
    def test_horizontal_flip_updates_bbox(self) -> None:
        transform = _compose(A.HorizontalFlip(p=1.0))

        _, box = apply_detection_transform(transform, self.image, self.box)

        self.assertAlmostEqual(box.x1, 64 - self.box.x2)
        self.assertAlmostEqual(box.x2, 64 - self.box.x1)
        self.assertAlmostEqual(box.y1, self.box.y1)
        self.assertAlmostEqual(box.y2, self.box.y2)

    #Verifies rotation and scaling still produce a positive-area box inside the image
    def test_rotation_and_scale_produce_valid_bbox(self) -> None:
        transform = _compose(
            A.Affine(
                scale=1.15,
                rotate=10.0,
                translate_percent=0.0,
                shear=0.0,
                keep_ratio=True,
                p=1.0,
            )
        )

        image, box = apply_detection_transform(transform, self.image, self.box)

        self.assertGreater(box.width, 0)
        self.assertGreater(box.height, 0)
        box.validate_within(*image.size)

    #Verifies a brightness/contrast-only transform changes pixels but never the box
    def test_photometric_transform_leaves_bbox_unchanged(self) -> None:
        transform = _compose(
            A.RandomBrightnessContrast(
                brightness_limit=(0.2, 0.2),
                contrast_limit=(0.2, 0.2),
                p=1.0,
            )
        )

        transformed_image, box = apply_detection_transform(
            transform, self.image, self.box
        )

        self.assertEqual(box, self.box)
        self.assertFalse(
            np.array_equal(np.asarray(transformed_image), np.asarray(self.image))
        )

    #Verifies two training transforms built with the same seed produce identical output
    def test_seed_reproduces_same_training_result(self) -> None:
        first_transform = build_training_transform(seed=42)
        second_transform = build_training_transform(seed=42)

        first_image, first_box = apply_detection_transform(
            first_transform, self.image, self.box
        )
        second_image, second_box = apply_detection_transform(
            second_transform, self.image, self.box
        )

        self.assertEqual(first_box, second_box)
        self.assertTrue(
            np.array_equal(np.asarray(first_image), np.asarray(second_image))
        )

    #Verifies the evaluation transform is deterministic and resolution is the fixed 640x416 model input
    def test_evaluation_transform_is_deterministic(self) -> None:
        transform = build_evaluation_transform()

        first_image, first_box = apply_detection_transform(
            transform, self.image, self.box
        )
        second_image, second_box = apply_detection_transform(
            transform, self.image, self.box
        )

        self.assertEqual(first_box, second_box)
        self.assertTrue(
            np.array_equal(np.asarray(first_image), np.asarray(second_image))
        )
        self.assertEqual(first_image.size, (640, 416))

    #Verifies it is non-square target size keeps the box geometry correct
    def test_rectangular_evaluation_transform_preserves_geometry(self) -> None:
        image = Image.new("RGB", (640, 400))
        box = BoundingBox(100, 50, 300, 250)

        transformed_image, transformed_box = apply_detection_transform(
            build_evaluation_transform((320, 512)), image, box
        )

        self.assertEqual(transformed_image.size, (512, 320))
        self.assertEqual(transformed_box, BoundingBox(80, 40, 240, 200))


if __name__ == "__main__":
    unittest.main()
