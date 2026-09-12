from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from data_loading import BoundingBox, ImageRecord
from models.yolo11 import (
    YOLO11ProbeDataset,
    _box_from_letterboxed_coordinates,
    collate_yolo_batch,
)


class YOLO11AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary_directory.name) / "image.jpg"
        Image.new("RGB", (64, 40), color=(128, 64, 32)).save(self.image_path)
        self.record = ImageRecord(
            image_id=7,
            path=self.image_path,
            width=64,
            height=40,
            box=BoundingBox(1, 2, 3, 4),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    #Verifies the validation adapter resize the image and normalizes the box to [0,1]
    def test_validation_adapter_letterboxes_and_normalizes_target(self) -> None:
        sample = YOLO11ProbeDataset(
            [self.record], training=False, seed=42, input_size=(416, 640)
        )[0]

        self.assertEqual(tuple(sample["img"].shape), (3, 416, 640))
        self.assertEqual(sample["img"].dtype, torch.uint8)
        expected = torch.tensor([[20 / 640, 38 / 416, 20 / 640, 20 / 416]])
        self.assertTrue(torch.allclose(sample["bboxes"], expected))
        self.assertEqual(sample["cls"].tolist(), [[0.0]])

    #Verifies a resized prediction converts back to the original image's box
    def test_letterbox_predictions_return_to_original_coordinates(self) -> None:
        box = _box_from_letterboxed_coordinates([10, 140, 30, 160], 64, 40)

        self.assertEqual(box, self.record.box)

    #Verifies a smaller square input size still produces correctly scaled box geometry
    def test_lower_resolution_preserves_bbox_geometry(self) -> None:
        sample = YOLO11ProbeDataset(
            [self.record], training=False, seed=42, input_size=320
        )[0]
        box = _box_from_letterboxed_coordinates(
            [5, 70, 15, 80], 64, 40, input_size=320
        )

        self.assertEqual(tuple(sample["img"].shape), (3, 320, 320))
        expected = torch.tensor([[10 / 320, 75 / 320, 10 / 320, 10 / 320]])
        self.assertTrue(torch.allclose(sample["bboxes"], expected))
        self.assertEqual(box, self.record.box)

    #Verifies a non-square input size still produces correctly scaled box geometry
    def test_rectangular_resolution_preserves_bbox_geometry(self) -> None:
        sample = YOLO11ProbeDataset(
            [self.record], training=False, seed=42, input_size=(320, 512)
        )[0]
        box = _box_from_letterboxed_coordinates(
            [8, 16, 24, 32], 64, 40, input_size=(320, 512)
        )

        self.assertEqual(tuple(sample["img"].shape), (3, 320, 512))
        expected = torch.tensor([[16 / 512, 24 / 320, 16 / 512, 16 / 320]])
        self.assertTrue(torch.allclose(sample["bboxes"], expected))
        self.assertEqual(box, self.record.box)

    #Verifies collation assigns each target to the correct image index within the batch
    def test_collate_assigns_each_target_to_its_batch_image(self) -> None:
        dataset = YOLO11ProbeDataset(
            [self.record], training=False, seed=42, input_size=(416, 640)
        )

        batch = collate_yolo_batch([dataset[0], dataset[0]])

        self.assertEqual(tuple(batch["img"].shape), (2, 3, 416, 640))
        self.assertEqual(batch["batch_idx"].tolist(), [0.0, 1.0])
        self.assertEqual(batch["image_id"], [7, 7])

    #Verifies the training adapter produces identical augmented output for the same seed and epoch
    def test_training_adapter_is_reproducible_for_seed_and_epoch(self) -> None:
        first = YOLO11ProbeDataset([self.record], training=True, seed=42)
        second = YOLO11ProbeDataset([self.record], training=True, seed=42)
        first.set_epoch(3)
        second.set_epoch(3)

        first_sample = first[0]
        second_sample = second[0]

        self.assertTrue(torch.equal(first_sample["img"], second_sample["img"]))
        self.assertTrue(
            torch.equal(first_sample["bboxes"], second_sample["bboxes"])
        )

if __name__ == "__main__":
    unittest.main()
