from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from person_detector_classifier.src.detector import PersonDetection
from person_detector_classifier.src.utils import to_float_list


def xyxy_to_coco_bbox(
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    digits: int = 2,
) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return to_float_list([x1, y1, w, h], digits)


def bbox_area_coco(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return float(max(0.0, bbox[2]) * max(0.0, bbox[3]))


def make_image_record(image_id: str, file_name: str, width: int, height: int) -> dict[str, Any]:
    return {
        "id": image_id,
        "file_name": file_name,
        "width": int(width),
        "height": int(height),
    }


def make_annotation(
    annotation_id: int,
    image_id: str,
    detection: PersonDetection,
    attributes: dict[str, Any],
    digits: int = 2,
    include_keypoints: bool = False,
) -> dict[str, Any]:
    bbox = xyxy_to_coco_bbox(detection.bbox_xyxy, digits=digits)
    ann: dict[str, Any] = {
        "id": int(annotation_id),
        "image_id": image_id,
        "area": round(bbox_area_coco(bbox), 6),
        "bbox": bbox,
        "iscrowd": 0,
        "attributes": attributes,
    }
    if include_keypoints and detection.keypoints is not None:
        # COCO-style flattened x,y,v. Here v stores confidence rounded, not a binary visibility flag.
        flat: list[float] = []
        for kp in detection.keypoints:
            if len(kp) >= 3:
                flat.extend([round(float(kp[0]), digits), round(float(kp[1]), digits), round(float(kp[2]), 4)])
        ann["keypoints"] = flat
        ann["num_keypoints"] = len(flat) // 3
    return ann


def make_dataset_payload(
    images: List[dict[str, Any]],
    annotations: List[dict[str, Any]],
    errors: List[dict[str, Any]] | None = None,
    version: str = "v2",
) -> dict[str, Any]:
    payload = {
        "info": {
            "name": "PDC_v3 Person Detector and Classifier output",
            "schema": "coco_like_person_attributes_v3",
            "version": version,
        },
        "images": images,
        "annotations": annotations,
    }
    if errors:
        payload["errors"] = errors
    return payload


def safe_image_id(path: str | Path) -> str:
    return Path(path).name
