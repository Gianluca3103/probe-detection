# Data structures for bounding boxes and images
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

#Stores and Derive bounding box information
@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned bounding box in ``xyxy`` coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    #Quick check to make sure values of the coords are positive
    def __post_init__(self) -> None:
        if not (0 <= self.x1 < self.x2 and 0 <= self.y1 < self.y2):
            raise ValueError(
                f"Invalid bounding box: {(self.x1, self.y1, self.x2, self.y2)}"
            )
    #Given the coordinates converts to bounding box format
    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> BoundingBox:
        """Convert a source ``[x, y, width, height]`` box to ``xyxy``."""

        return cls(x1=x, y1=y, x2=x + width, y2=y + height)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def validate_within(self, image_width: int, image_height: int) -> None:

    #Checks if the bounding box extends past the image boundaries
        if self.x2 > image_width or self.y2 > image_height:
            raise ValueError(
                "Bounding box lies outside image dimensions: "
                f"box={(self.x1, self.y1, self.x2, self.y2)}, "
                f"image_size={(image_width, image_height)}"
            )

#Storing Image Information
@dataclass(frozen=True)
class ImageRecord:

    image_id: int
    path: Path
    width: int
    height: int
    box: BoundingBox

    def __post_init__(self) -> None:
        self.box.validate_within(self.width, self.height)
