from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from evaluate import _split_manifest_sha256_candidates

from data_loading.evaluation import (
    GroundTruthDetection,
    Prediction,
    evaluate_detections,
    intersection_over_union,
)
from data_loading.probe_types import BoundingBox


def _ground_truth(
    image_id: int,
    box: BoundingBox = BoundingBox(0, 0, 10, 10),
    image_size: tuple[int, int] = (100, 100),
) -> GroundTruthDetection:
    return GroundTruthDetection(image_id, box, *image_size)


class EvaluationTests(unittest.TestCase):
    # Git may check the same manifest out with LF or CRLF; provenance must remain portable.
    def test_split_manifest_hash_ignores_line_ending_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lf_path = Path(directory) / "lf.txt"
            crlf_path = Path(directory) / "crlf.txt"
            changed_path = Path(directory) / "changed.txt"
            lf_path.write_bytes(b"image_1.jpg\nimage_2.jpg\n")
            crlf_path.write_bytes(b"image_1.jpg\r\nimage_2.jpg\r\n")
            changed_path.write_bytes(b"image_1.jpg\nimage_3.jpg\n")

            self.assertEqual(
                _split_manifest_sha256_candidates(lf_path),
                _split_manifest_sha256_candidates(crlf_path),
            )
            self.assertFalse(
                _split_manifest_sha256_candidates(lf_path)
                & _split_manifest_sha256_candidates(changed_path)
            )

    #Verifies IoU is computed correctly on a simple, hand-checkable 50% overlap
    def test_iou_has_obvious_half_overlap(self) -> None:
        first = BoundingBox(0, 0, 10, 10)
        second = BoundingBox(0, 0, 5, 10)

        self.assertAlmostEqual(intersection_over_union(first, second), 0.5)

    #Verifies an exact-match prediction scores 1.0 on every metric
    def test_perfect_prediction_has_perfect_metrics(self) -> None:
        ground_truth = [_ground_truth(1)]
        predictions = [Prediction(1, BoundingBox(0, 0, 10, 10), 0.9)]

        metrics = evaluate_detections(ground_truth, predictions).overall

        self.assertEqual(metrics.map50_95, 1.0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.mean_iou, 1.0)

    #Verifies mAP50:95 correctly averages precision across all ten IoU thresholds
    def test_map_averages_the_ten_coco_iou_thresholds(self) -> None:
        ground_truth = [_ground_truth(1)]
        predictions = [Prediction(1, BoundingBox(0, 0, 5, 10), 0.9)]

        metrics = evaluate_detections(ground_truth, predictions).overall

        self.assertAlmostEqual(metrics.map50_95, 0.1)

    #Verifies precision, recall, and F1 on a mixed case with one TP, one FP, and one FN
    def test_precision_recall_and_f1_with_one_tp_one_fp_one_fn(self) -> None:
        ground_truth = [_ground_truth(1), _ground_truth(2)]
        predictions = [
            Prediction(1, BoundingBox(0, 0, 10, 10), 0.9),
            Prediction(2, BoundingBox(50, 50, 60, 60), 0.8),
        ]

        metrics = evaluate_detections(ground_truth, predictions).overall

        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.f1, 0.5)
        self.assertEqual(metrics.mean_iou, 1.0)

    #Verifies a borderline prediction is included or excluded based on the confidence threshold
    def test_confidence_and_matching_iou_thresholds_are_configurable(self) -> None:
        ground_truth = [_ground_truth(1)]
        predictions = [Prediction(1, BoundingBox(0, 0, 5, 10), 0.4)]

        included = evaluate_detections(
            ground_truth,
            predictions,
            confidence_threshold=0.4,
            matching_iou_threshold=0.5,
        ).overall
        excluded = evaluate_detections(
            ground_truth,
            predictions,
            confidence_threshold=0.5,
            matching_iou_threshold=0.5,
        ).overall

        self.assertEqual(included.true_positives, 1)
        self.assertAlmostEqual(included.mean_iou, 0.5)
        self.assertEqual(excluded.true_positives, 0)
        self.assertEqual(excluded.recall, 0.0)

    #Verifies a second prediction on an already matched image counts as a false positive
    def test_duplicate_prediction_is_false_positive(self) -> None:
        ground_truth = [_ground_truth(1)]
        predictions = [
            Prediction(1, BoundingBox(0, 0, 10, 10), 0.9),
            Prediction(1, BoundingBox(0, 0, 10, 10), 0.8),
        ]

        metrics = evaluate_detections(ground_truth, predictions).overall

        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 1.0)

if __name__ == "__main__":
    unittest.main()
