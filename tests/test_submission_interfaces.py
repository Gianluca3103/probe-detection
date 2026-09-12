from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from PIL import Image

from deployment import ProbeDetection
from inference import (
    _draw_result,
    _fill_engine_input,
    _latency_summary,
)


class _FakeDetector:
    frame_shape = (400, 640, 3)

    def __init__(self, detection: ProbeDetection | None = None) -> None:
        self.input_buffer = np.empty(self.frame_shape, dtype=np.uint8)
        self.detection = detection

    def detect_buffer(self) -> ProbeDetection | None:
        return self.detection


class SubmissionInterfaceTests(unittest.TestCase):
    #Verifies ProbeDetection.bbox packages the four coordinates as one tuple
    def test_detection_exposes_xyxy_tuple(self) -> None:
        detection = ProbeDetection(1, 2, 3, 4, 0.9)
        self.assertEqual(detection.bbox, (1, 2, 3, 4))

    #Verifies an annotated image is still saved when no probe is detected
    def test_no_detection_image_is_still_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jpg"
            destination = Path(directory) / "result.jpg"
            Image.new("RGB", (64, 40), "black").save(source)

            _draw_result(source, destination, None)

            self.assertTrue(destination.is_file())

    #Verifies a correctly-sized image is copied into the engine's input buffer unchanged
    def test_fixed_size_image_is_copied_into_engine_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            Image.new("RGB", (640, 400), "white").save(image_path)
            detector = _FakeDetector()

            _fill_engine_input(detector, image_path)

            self.assertTrue(np.all(detector.input_buffer == 255))

    #Verifies an image with the wrong dimensions raises ValueError instead of being resized
    def test_incorrect_image_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "square.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            detector = _FakeDetector()

            with self.assertRaisesRegex(ValueError, "Expected a 640x400 image"):
                _fill_engine_input(detector, image_path)

    #Verifies latency values convert from seconds to milliseconds with correct summary stats
    def test_latency_summary_uses_milliseconds(self) -> None:
        summary = _latency_summary([0.001, 0.002, 0.003])
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["minimum"], 1.0)
        self.assertEqual(summary["maximum"], 3.0)


if __name__ == "__main__":
    unittest.main()
