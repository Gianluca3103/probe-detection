#Load and validation for the supplied dataset

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from PIL import Image
from .probe_types import BoundingBox, ImageRecord


#Computes the SHA256 checksum of a file, shared by evaluation, training, and deployment scripts
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Load JSON file into proper coordinates
def load_dataset(images_dir: Path, annotations_path: Path) -> tuple[ImageRecord, ...]:

    images_dir = Path(images_dir)
    annotations_path = Path(annotations_path)

    with annotations_path.open(encoding="utf-8") as file:
        payload: dict[str, Any] = json.load(file)

    image_entries: list[dict[str, Any]] = payload["images"]
    annotation_entries: list[dict[str, Any]] = payload["annotations"]
    #Checks for duplicate IDs in the JSON file
    entries_by_id: dict[int, dict[str, Any]] = {}
    for entry in image_entries:
        image_id = entry["id"]
        if image_id in entries_by_id:
            raise ValueError(f"Duplicate image id: {image_id!r}")
        entries_by_id[image_id] = entry

    box_by_image_id: dict[int, BoundingBox] = {}
    for annotation in annotation_entries:
        image_id = annotation["image_id"]
        if image_id not in entries_by_id:
            raise ValueError(f"Annotation references unknown image id: {image_id!r}")
        if image_id in box_by_image_id:
            raise ValueError(f"Multiple annotations found for image id: {image_id!r}")

        source_box = annotation["bbox"]
        if len(source_box) != 4:
            raise ValueError(
                f"Expected four xywh coordinates for image {image_id!r}, got {source_box!r}"
            )
        box_by_image_id[image_id] = BoundingBox.from_xywh(
            *(float(value) for value in source_box)
        )

    records: list[ImageRecord] = []
    for entry in image_entries:
        image_id = entry["id"]
        declared_width = entry["width"]
        declared_height = entry["height"]
        image_path = images_dir / entry["file_name"]

        with Image.open(image_path) as image:
            actual_size = image.size
            image.verify()

        declared_size = (declared_width, declared_height)
        if actual_size != declared_size:
            raise ValueError(
                f"Image dimensions do not match metadata for {image_path}: "
                f"actual={actual_size}, declared={declared_size}"
            )

        records.append(
            ImageRecord(
                image_id=image_id,
                path=image_path,
                width=declared_width,
                height=declared_height,
                box=box_by_image_id[image_id],
            )
        )

    return tuple(records)

#Select records in the exact order necessary
def load_split_records(
    records: Sequence[ImageRecord], split_path: Path
) -> tuple[ImageRecord, ...]:


    split_path = Path(split_path)
    records_by_name = {record.path.name: record for record in records}
    names = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate image filename in split: {split_path}")

    missing = [name for name in names if name not in records_by_name]
    if missing:
        raise ValueError(f"Split references unknown image: {missing[0]}")
    return tuple(records_by_name[name] for name in names)
