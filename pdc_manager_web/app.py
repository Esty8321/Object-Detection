from __future__ import annotations

import os
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Make the existing project package importable when this file is started directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person_detector_classifier.config import AttributeConfig, ClassifierConfig, DetectorConfig, PipelineConfig
from person_detector_classifier.src.pipeline import PersonDetectorClassifierPipeline
from person_detector_classifier.src.utils import read_image_bgr, write_image


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# The ML objects are expensive to load. Create them once, not once per request.
pipeline = PersonDetectorClassifierPipeline(
    PipelineConfig(
        detector=DetectorConfig(
            backend=os.getenv("PDC_DETECTOR_BACKEND", "yolo"),
            model_path=os.getenv("PDC_YOLO_MODEL", "person_detector_classifier/models/yolo11x-pose.pt"),
            image_size=int(os.getenv("PDC_IMAGE_SIZE", "1280")),
            conf_threshold=float(os.getenv("PDC_DETECTION_CONFIDENCE", "0.35")),
            iou_threshold=float(os.getenv("PDC_IOU", "0.70")),
            device=os.getenv("PDC_DEVICE") or None,
        ),
        classifier=ClassifierConfig(
            backend=os.getenv("PDC_CLASSIFIER_BACKEND", "torch"),
            pth_path=Path(os.getenv("PDC_PTH_MODEL", "person_detector_classifier/models/mobilenetv4_utkface_age_gender_best.pth")),
            onnx_path=Path(os.getenv("PDC_ONNX_MODEL", "person_detector_classifier/models/mobilenetv4_utkface_age_gender.onnx")),
            device=os.getenv("PDC_DEVICE") or None,
        ),
        attributes=AttributeConfig(
            keypoint_conf_threshold=float(os.getenv("PDC_KEYPOINT_CONFIDENCE", "0.35")),
            save_body_region_crops=True,
        ),
        save_visualization=True,
        include_debug=True,
    )
)
pipeline_lock = threading.Lock()


REGION_COLORS = {
    "head": (72, 202, 228),
    "upper_body": (80, 200, 120),
    "left_hand": (242, 166, 64),
    "right_hand": (242, 166, 64),
    "waist": (205, 119, 255),
    "hips": (120, 170, 255),
    "left_leg": (70, 220, 210),
    "right_leg": (70, 220, 210),
}


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _draw_region_overlay(source_path: Path, payload: dict[str, Any], output_path: Path) -> None:
    image = read_image_bgr(source_path)
    thickness = max(2, round(sum(image.shape[:2]) / 900))
    for person_number, annotation in enumerate(payload.get("annotations", []), start=1):
        regions = annotation.get("attributes", {}).get("body_regions", {})
        for name, info in regions.items():
            if info.get("status") != "available":
                continue
            bbox = info.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            color = REGION_COLORS.get(name, (230, 230, 230))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            label = f"P{person_number} · {name}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, thickness)
            label_top = max(0, y1 - text_height - baseline - 10)
            cv2.rectangle(image, (x1, label_top), (x1 + text_width + 12, y1), color, -1)
            cv2.putText(image, label, (x1 + 6, y1 - baseline - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 20, 28), thickness, cv2.LINE_AA)
    write_image(output_path, image)


def _public_result(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    people: list[dict[str, Any]] = []
    for person_number, annotation in enumerate(payload.get("annotations", []), start=1):
        attrs = annotation.get("attributes", {})
        region_items: list[dict[str, Any]] = []
        for name, info in attrs.get("body_regions", {}).items():
            item = {
                "name": name,
                "status": info.get("status", "unavailable"),
                "quality": info.get("quality", "unknown"),
                "reason": info.get("reason", ""),
                "bbox_xyxy": info.get("bbox_xyxy"),
                "image_url": None,
            }
            crop_path = f"body_parts/input/person_{person_number:03d}/{name}.jpg"
            if item["status"] == "available" and (RUNS_DIR / run_id / crop_path).is_file():
                item["image_url"] = url_for("run_file", run_id=run_id, filename=crop_path)
            region_items.append(item)
        people.append({
            "number": person_number,
            "detection_bbox": annotation.get("bbox"),
            "gender": attrs.get("gender", "unknown"),
            "visibility": attrs.get("visibility", "unknown"),
            "occlusion": attrs.get("occlusion", "unknown"),
            "quality": attrs.get("quality", "unknown"),
            "rotation": attrs.get("rotation", 0),
            "regions": region_items,
        })
    return {
        "run_id": run_id,
        "people_count": len(people),
        "annotated_image_url": url_for("run_file", run_id=run_id, filename="regions_annotated.jpg"),
        "people": people,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose an image before starting the analysis."}), 400
    if not _allowed_file(upload.filename):
        return jsonify({"error": "Unsupported file type. Upload JPG, PNG, WEBP, BMP, or TIFF."}), 400

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    input_path = run_dir / f"input{Path(secure_filename(upload.filename)).suffix.lower()}"
    upload.save(input_path)

    try:
        # A lock avoids concurrent GPU access and model-state contention.
        with pipeline_lock:
            payload = pipeline.process_image(
                input_path,
                output_dir=run_dir,
                include_keypoints=True,
                save_visualization=True,
            )
        _draw_region_overlay(input_path, payload, run_dir / "regions_annotated.jpg")
        return jsonify(_public_result(run_id, payload))
    except Exception as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        app.logger.exception("Image analysis failed")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.get("/runs/<run_id>/<path:filename>")
def run_file(run_id: str, filename: str):
    if not run_id.isalnum():
        return jsonify({"error": "Invalid result identifier."}), 400
    return send_from_directory(RUNS_DIR / run_id, filename)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "The image is larger than 25 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=False)