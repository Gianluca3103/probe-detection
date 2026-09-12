from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data_loading.probe_data import load_dataset, load_split_records
from data_loading.probe_types import BoundingBox, ImageRecord


class BoundingBoxTests(unittest.TestCase):
    #Verifies xywh coordinates convert to xyxy with the correct width/height
    def test_xywh_conversion(self) -> None:
        box = BoundingBox.from_xywh(10, 20, 30, 40)

        self.assertEqual(box, BoundingBox(x1=10, y1=20, x2=40, y2=60))
        self.assertEqual(box.width, 30)
        self.assertEqual(box.height, 40)

    #Verifies if the bounding box is inside the image area
    def test_valid_box_passes_image_validation(self) -> None:
        record = ImageRecord(
            image_id=1,
            path=Path("image.jpg"),
            width=100,
            height=80,
            box=BoundingBox(0, 0, 100, 80),
        )

        self.assertEqual(record.box, BoundingBox(0, 0, 100, 80))

    #Verifies doesn't have zero or negative area raises error
    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid bounding box"):
            BoundingBox(x1=10, y1=5, x2=10, y2=20)

    #Verifies a box that extends past the image bounds raises Error
    def test_box_outside_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside image dimensions"):
            ImageRecord(
                image_id=1,
                path=Path("image.jpg"),
                width=100,
                height=80,
                box=BoundingBox(0, 0, 101, 80),
            )


class DatasetLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.images_dir = self.root / "images"
        self.images_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_image(self, name: str, size: tuple[int, int] = (100, 80)) -> None:
        Image.new("RGB", size, color="black").save(self.images_dir / name)

    def _write_annotations(self, payload: dict[str, object]) -> Path:
        annotations_path = self.root / "annotations.json"
        annotations_path.write_text(json.dumps(payload), encoding="utf-8")
        return annotations_path

    #Verifies annotations are matched to their image by id, not by JSON list order
    def test_annotations_are_associated_by_image_id(self) -> None:
        self._create_image("first.jpg")
        self._create_image("second.jpg")
        annotations_path = self._write_annotations(
            {
                "images": [
                    {"id": 10, "file_name": "first.jpg", "width": 100, "height": 80},
                    {"id": 20, "file_name": "second.jpg", "width": 100, "height": 80},
                ],
                "annotations": [
                    {"image_id": 20, "bbox": [5, 6, 10, 12]},
                    {"image_id": 10, "bbox": [1, 2, 3, 4]},
                ],
            }
        )

        records = load_dataset(self.images_dir, annotations_path)

        self.assertEqual(records[0].box, BoundingBox(1, 2, 4, 6))
        self.assertEqual(records[1].box, BoundingBox(5, 6, 15, 18))

    #Verifies an annotation referencing a nonexistent image id raises ValueError
    def test_unknown_annotation_image_id_is_rejected(self) -> None:
        annotations_path = self._write_annotations(
            {
                "images": [],
                "annotations": [{"image_id": 99, "bbox": [1, 2, 3, 4]}],
            }
        )

        with self.assertRaisesRegex(ValueError, "unknown image id"):
            load_dataset(self.images_dir, annotations_path)

    #Verifies split records come back in the manifest file's listed order, not load order
    def test_split_records_follow_manifest_order(self) -> None:
        records = (
            ImageRecord(1, Path("first.jpg"), 10, 10, BoundingBox(0, 0, 1, 1)),
            ImageRecord(2, Path("second.jpg"), 10, 10, BoundingBox(0, 0, 1, 1)),
        )
        split_path = self.root / "split.txt"
        split_path.write_text("second.jpg\nfirst.jpg\n", encoding="utf-8")

        selected = load_split_records(records, split_path)

        self.assertEqual([record.image_id for record in selected], [2, 1])


if __name__ == "__main__":
    unittest.main()
