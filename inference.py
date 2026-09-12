#Run the images through the final model to detect the probe

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from deployment import ProbeDetection, ProbeDetector
#What type of images it takes
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
#Iterations ignored by the inference time measurement due to GPU "warming up"
WARMUP_ITERATIONS = 3
MODEL_NAME = "YOLO11-N"
CONFIDENCE_THRESHOLD = 0.3985197842121124

#Purely visualization, draws the bounding box on the image after it detects the probe
def _draw_result(
    image_path: Path,
    output_path: Path,
    detection: ProbeDetection | None,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    if detection is None:
        draw.rectangle((0, 0, 190, 22), fill="black")
        draw.text((5, 5), "NO PROBE DETECTED", fill="red")
    else:
        draw.rectangle(detection.bbox, outline="lime", width=3)
        label_y = max(0, detection.y1 - 20)
        draw.rectangle(
            (detection.x1, label_y, detection.x1 + 105, detection.y1),
            fill="black",
        )
        draw.text(
            (detection.x1 + 3, max(0, detection.y1 - 17)),
            f"probe {detection.confidence:.3f}",
            fill="lime",
        )
    image.save(output_path)

# Read each image and checks if the image input size is correct and copies pixel into input buffer
def _fill_engine_input(detector: ProbeDetector, image_path: Path) -> None:

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        target_height, target_width, _ = detector.frame_shape
        if image.size != (target_width, target_height):
            raise ValueError(
                f"Expected a {target_width}x{target_height} image, "
                f"got {image.width}x{image.height}: {image_path}"
            )
        image_array = np.asarray(image, dtype=np.uint8)

    detector.input_buffer[...] = image_array


# Gives the latency data of the run
def _latency_summary(seconds: list[float]) -> dict[str, float]:
    milliseconds = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "mean": float(milliseconds.mean()),
        "median": float(np.median(milliseconds)),
        "p95": float(np.percentile(milliseconds, 95)),
        "p99": float(np.percentile(milliseconds, 99)),
        "minimum": float(milliseconds.min()),
        "maximum": float(milliseconds.max()),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    engine_path = root / "deployment_artifacts/probe_detector.engine"

    #Breaks down input arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/inference"))
    args = parser.parse_args()

    # Validate input arguments
    if args.input_path.is_file():
        image_paths = [args.input_path]
    elif args.input_path.is_dir():
        if args.input_path.resolve() == args.output_dir.resolve():
            raise ValueError("Input and output directories must be different")
        image_paths = sorted(
            path
            for path in args.input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        raise FileNotFoundError(f"Image or folder not found: {args.input_path}")
    if not image_paths:
        raise ValueError(f"No supported image files found in {args.input_path}")
    if any(path.suffix.lower() not in IMAGE_SUFFIXES for path in image_paths):
        raise ValueError(f"Unsupported image format: {image_paths[0].suffix}")

    #Creates and initializes probe detection instance
    detector = ProbeDetector(engine_path)
    #Creates output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _fill_engine_input(detector, image_paths[0])
    #executing the warmup and setting up cuda graph capture
    detector.prepare()

    #Initializing some variables for results
    rows: list[dict[str, object]] = []
    runtime_times: list[float] = []
    pipeline_times: list[float] = []
    complete_times: list[float] = []
    detected_count = 0

    #Main loop, processes every image
    for image_path in tqdm(image_paths, desc="Detecting", unit="image"):
        #Starts clocks
        complete_start = time.perf_counter()
        pipeline_start = time.perf_counter()
        #Loads and validate the images
        _fill_engine_input(detector, image_path)
        runtime_start = time.perf_counter()
        #Runs the trained model
        detection = detector.detect_buffer()
        #Records the time taken
        runtime_times.append(time.perf_counter() - runtime_start)
        pipeline_times.append(time.perf_counter() - pipeline_start)

        #Draws bounding box on the image
        _draw_result(image_path, args.output_dir / image_path.name, detection)
        complete_times.append(time.perf_counter() - complete_start)
        #Checks if there's detection
        if detection is None:
            rows.append({"file_name": image_path.name, "detected": False})
            continue
        #If detection set to false, don't update the detected count
        detected_count += 1
        #If detected append bbox coordinates, and confidence into rows dict
        rows.append(
            {
                "file_name": image_path.name,
                "detected": True,
                "confidence": detection.confidence,
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
            }
        )
    #Storing the results in CSV
    fieldnames = ["file_name", "detected", "confidence", "x1", "y1", "x2", "y2"]
    with (args.output_dir / "detections.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    detector_latency = _latency_summary(runtime_times)
    input_to_detection_latency = _latency_summary(pipeline_times)
    folder_processing_latency = _latency_summary(complete_times)
    #Report results of run
    report: dict[str, object] = {
        "backend": "TensorRT FP16 with embedded NMS and CUDA Graph replay",
        "model": MODEL_NAME,
        "engine": str(engine_path),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "warmup_iterations": WARMUP_ITERATIONS,
        "images_processed": len(image_paths),
        "images_detected": detected_count,
        "images_not_detected": len(image_paths) - detected_count,
        "latency_ms": {
            "optimized_detector": detector_latency,
            "decoded_file_to_detection": input_to_detection_latency,
            "folder_workflow_with_visualization": folder_processing_latency,
        },
        "latency_scope": {
            "optimized_detector": (
                "Pinned H2D copy, fused preprocessing, TensorRT inference, embedded "
                "confidence filtering and NMS, coordinate restoration, compact D2H "
                "copy, synchronization, and CPU result unpacking"
            ),
            "decoded_file_to_detection": (
                "JPEG/file decode and RGB array creation plus optimized_detector"
            ),
            "folder_workflow_with_visualization": (
                "decoded_file_to_detection plus drawing and saving the annotated image"
            ),
        },
    }
    #Writes results in the output file
    summary_path = args.output_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    #Gives image detected summary
    print("\nInference summary")
    print(f"Images processed:             {len(image_paths)}")
    print(f"Images detected:              {detected_count}")
    print(f"Images not detected:          {len(image_paths) - detected_count}")
    print(f"Average optimized detector:    {detector_latency['mean']:.3f} ms/image")
    print(
        "Average file-to-detection:    "
        f"{input_to_detection_latency['mean']:.3f} ms/image"
    )
    print(
        "Average with visualization:  "
        f"{folder_processing_latency['mean']:.3f} ms/image"
    )
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
