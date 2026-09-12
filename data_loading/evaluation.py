"""Model-independent evaluation for single-class probe detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .probe_types import BoundingBox, ImageRecord

#The ten COCO IoU thresholds 0.50 to 0.95 averaged together to get mAP50:95
_COCO_IOU_THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(10))
#The recall points used for interpolated AP
_COCO_RECALL_THRESHOLDS = tuple(index / 100 for index in range(101))
#Only class supported
_PROBE_CLASS = "probe"


#Stores the correct answer for one image: its ground-truth box and the image's own size
@dataclass(frozen=True, slots=True)
class GroundTruthDetection:
    image_id: int
    box: BoundingBox
    image_width: int
    image_height: int
    class_name: str = _PROBE_CLASS

    #Checks class name, image size, and that the box fits inside the image
    def __post_init__(self) -> None:
        if self.class_name != _PROBE_CLASS:
            raise ValueError(f"Only the probe class is supported, got {self.class_name!r}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        self.box.validate_within(self.image_width, self.image_height)


#Stores one model prediction for one image: its box and confidence score
@dataclass(frozen=True, slots=True)
class Prediction:
    image_id: int
    box: BoundingBox
    confidence: float
    class_name: str = _PROBE_CLASS

    #Checks class name and that confidence is a valid probability
    def __post_init__(self) -> None:
        if self.class_name != _PROBE_CLASS:
            raise ValueError(f"Only the probe class is supported, got {self.class_name!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence!r}"
            )


#Holds every computed metric for one evaluation run
@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    map50_95: float
    precision: float
    recall: float
    f1: float
    mean_iou: float
    ground_truth_count: int
    prediction_count: int
    true_positives: int
    false_positives: int
    false_negatives: int


#Stores the overall metrics and the thresholds used to get them
@dataclass(frozen=True, slots=True)
class EvaluationResult:
    overall: DetectionMetrics
    confidence_threshold: float
    matching_iou_threshold: float


#Turns loaded dataset records into the ground-truth format the evaluator expects
def ground_truth_from_records(
    records: Sequence[ImageRecord],
) -> tuple[GroundTruthDetection, ...]:

    return tuple(
        GroundTruthDetection(
            image_id=record.image_id,
            box=record.box,
            image_width=record.width,
            image_height=record.height,
        )
        for record in records
    )


#Computes IoU between two boxes overlap area divided by combined area
def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    #Sets 0 so non-overlapping boxes give zero overlap instead of a negative width/height
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(
        0.0, min(first.y2, second.y2) - max(first.y1, second.y1)
    )
    intersection = intersection_width * intersection_height
    #Union counts the overlap once, add both areas then subtract the double-counted overlap
    union = first.area + second.area - intersection
    return intersection / union


#scores a full set of predictions against ground truth and returns every metric
def evaluate_detections(
    ground_truth: Sequence[GroundTruthDetection],
    predictions: Sequence[Prediction],
    *,
    confidence_threshold: float = 0.25,
    matching_iou_threshold: float = 0.50,
    max_detections_per_image: int = 100,
) -> EvaluationResult:

    #Checks every argument is in a valid, sane range before doing any real work
    if not ground_truth:
        raise ValueError("Ground truth must not be empty")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1")
    if not 0.0 < matching_iou_threshold <= 1.0:
        raise ValueError("Matching IoU threshold must be greater than 0 and at most 1")
    if max_detections_per_image <= 0:
        raise ValueError("Maximum detections per image must be positive")

    ground_truth_by_id = _ground_truth_by_id(ground_truth)
    #Makes sure no prediction points at an image that has no ground truth at all
    unknown_ids = sorted(
        {prediction.image_id for prediction in predictions} - ground_truth_by_id.keys()
    )
    if unknown_ids:
        raise ValueError(f"Prediction references unknown image id: {unknown_ids[0]!r}")

    #Caps predictions per image, then scores the full dataset
    limited_predictions = _limit_predictions(predictions, max_detections_per_image)
    overall = _evaluate_subset(
        ground_truth,
        limited_predictions,
        confidence_threshold,
        matching_iou_threshold,
    )

    return EvaluationResult(
        overall=overall,
        confidence_threshold=confidence_threshold,
        matching_iou_threshold=matching_iou_threshold,
    )


#Indexes ground truth by image id for fast lookup, and rejects a duplicate box for the same image
def _ground_truth_by_id(
    ground_truth: Sequence[GroundTruthDetection],
) -> dict[int, GroundTruthDetection]:
    by_id: dict[int, GroundTruthDetection] = {}
    for detection in ground_truth:
        if detection.image_id in by_id:
            raise ValueError(
                f"Multiple ground-truth boxes found for image id: {detection.image_id!r}"
            )
        by_id[detection.image_id] = detection
    return by_id


#Keeps only the top highest-confidence predictions per image, dropping the rest
def _limit_predictions(
    predictions: Sequence[Prediction], max_detections_per_image: int
) -> tuple[Prediction, ...]:
    #Groups predictions by image first, since the cap applies per image, not overall
    by_image: dict[int, list[Prediction]] = {}
    for prediction in predictions:
        by_image.setdefault(prediction.image_id, []).append(prediction)

    limited: list[Prediction] = []
    for image_predictions in by_image.values():
        #Sorts by confidence and slices to the cap, so only the weakest extra predictions are dropped
        limited.extend(
            sorted(
                image_predictions,
                key=lambda prediction: prediction.confidence,
                reverse=True,
            )[:max_detections_per_image]
        )
    return tuple(limited)


#Computes one full DetectionMetrics for a given ground-truth/prediction pair
def _evaluate_subset(
    ground_truth: Sequence[GroundTruthDetection],
    predictions: Sequence[Prediction],
    confidence_threshold: float,
    matching_iou_threshold: float,
) -> DetectionMetrics:
    #Precision/recall/F1/IoU only look at predictions that clear the confidence threshold
    operating_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.confidence >= confidence_threshold
    )
    true_positives, false_positives, matched_ious = _match_predictions(
        ground_truth, operating_predictions, matching_iou_threshold
    )
    #Every ground truth not matched counts as a miss
    false_negatives = len(ground_truth) - true_positives

    #The 0.0 fallbacks below avoid dividing by zero when there are no predictions or matches
    precision_denominator = true_positives + false_positives
    precision = (
        true_positives / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positives / len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    #mAP needs the full ranked prediction list, not the confidence-filtered one, to build its curve
    average_precisions = tuple(
        _average_precision(ground_truth, predictions, threshold)
        for threshold in _COCO_IOU_THRESHOLDS
    )
    return DetectionMetrics(
        map50_95=sum(average_precisions) / len(average_precisions),
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        ground_truth_count=len(ground_truth),
        prediction_count=len(operating_predictions),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


#Greedily matches predictions to ground truth at one IoU threshold, highest confidence first
def _match_predictions(
    ground_truth: Sequence[GroundTruthDetection],
    predictions: Sequence[Prediction],
    iou_threshold: float,
) -> tuple[int, int, list[float]]:
    ground_truth_by_id = _ground_truth_by_id(ground_truth)
    matched_image_ids: set[int] = set()
    matched_ious: list[float] = []
    true_positives = 0
    false_positives = 0

    #Goes through predictions best-confidence-first, so the strongest guess gets first claim on a match
    for prediction in sorted(
        predictions, key=lambda detection: detection.confidence, reverse=True
    ):
        target = ground_truth_by_id[prediction.image_id]
        iou = intersection_over_union(prediction.box, target.box)
        #An image can only be matched once: a second prediction on it is always a false positive
        if prediction.image_id not in matched_image_ids and iou >= iou_threshold:
            matched_image_ids.add(prediction.image_id)
            matched_ious.append(iou)
            true_positives += 1
        else:
            false_positives += 1

    return true_positives, false_positives, matched_ious


#Computes COCO-style average precision (AP) at one IoU threshold from the full ranked prediction list
def _average_precision(
    ground_truth: Sequence[GroundTruthDetection],
    predictions: Sequence[Prediction],
    iou_threshold: float,
) -> float:
    ground_truth_by_id = _ground_truth_by_id(ground_truth)
    matched_image_ids: set[int] = set()
    cumulative_true_positives = 0
    cumulative_false_positives = 0
    recalls: list[float] = []
    precisions: list[float] = []

    #Records precision/recall after each prediction, building the curve this function integrates over
    for prediction in sorted(
        predictions, key=lambda detection: detection.confidence, reverse=True
    ):
        target = ground_truth_by_id[prediction.image_id]
        iou = intersection_over_union(prediction.box, target.box)
        if prediction.image_id not in matched_image_ids and iou >= iou_threshold:
            matched_image_ids.add(prediction.image_id)
            cumulative_true_positives += 1
        else:
            cumulative_false_positives += 1

        recalls.append(cumulative_true_positives / len(ground_truth))
        precisions.append(
            cumulative_true_positives
            / (cumulative_true_positives + cumulative_false_positives)
        )

    #No predictions at all means zero AP
    if not precisions:
        return 0.0

    #Standard COCO smoothing: replaces each precision with the best one seen at any higher recall
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    #101-point interpolation: averages the best precision found at each of the 101 recall thresholds
    interpolated_precision = 0.0
    for recall_threshold in _COCO_RECALL_THRESHOLDS:
        precision = 0.0
        for recall, candidate_precision in zip(recalls, precisions):
            if recall >= recall_threshold:
                precision = candidate_precision
                break
        interpolated_precision += precision
    return interpolated_precision / len(_COCO_RECALL_THRESHOLDS)
