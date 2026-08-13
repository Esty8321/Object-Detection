from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from person_detector_classifier.src.detector import PersonDetection


# COCO skeleton pairs for 17-keypoint layout.
SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]


def _auto_thickness(image_bgr: np.ndarray) -> int:
    h, w = image_bgr.shape[:2]
    return max(1, round((h + w) / 900))


def _color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (68, 221, 255),
        (85, 210, 120),
        (255, 170, 80),
        (220, 120, 255),
        (120, 170, 255),
        (70, 240, 200),
    ]
    return palette[index % len(palette)]


def _draw_label(image: np.ndarray, text: str, x: int, y: int, thickness: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, thickness * 0.35)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + baseline + 4, y)
    cv2.rectangle(image, (x, y - th - baseline - 8), (x + tw + 12, y + 5), (18, 22, 30), -1)
    cv2.rectangle(image, (x, y - th - baseline - 8), (x + tw + 12, y + 5), color, max(1, thickness // 2))
    cv2.putText(image, text, (x + 6, y - baseline - 2), font, scale, (245, 248, 255), thickness, cv2.LINE_AA)


def draw_detection(
    image_bgr: np.ndarray,
    detection: PersonDetection,
    annotation: Dict[str, Any] | None = None,
    keypoint_threshold: float = 0.35,
    index: int = 0,
    preview_only: bool = False,
) -> None:
    thickness = _auto_thickness(image_bgr)
    color = _color_for_index(index)
    x1, y1, x2, y2 = [int(round(v)) for v in detection.bbox_xyxy]
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, max(1, thickness))
    cv2.addWeighted(overlay, 0.82, image_bgr, 0.18, 0, image_bgr)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, max(1, thickness + 1))

    if detection.keypoints is not None:
        kps = detection.keypoints
        for a, b in SKELETON:
            if a < len(kps) and b < len(kps):
                ka, kb = kps[a], kps[b]
                if len(ka) >= 3 and len(kb) >= 3 and ka[2] >= keypoint_threshold and kb[2] >= keypoint_threshold:
                    pa = (int(round(float(ka[0]))), int(round(float(ka[1]))))
                    pb = (int(round(float(kb[0]))), int(round(float(kb[1]))))
                    cv2.line(image_bgr, pa, pb, (60, 190, 255), thickness, cv2.LINE_AA)
        for kp in kps:
            if len(kp) >= 3 and kp[2] >= keypoint_threshold:
                p = (int(round(float(kp[0]))), int(round(float(kp[1]))))
                cv2.circle(image_bgr, p, max(2, thickness + 2), (20, 240, 255), -1, cv2.LINE_AA)
                cv2.circle(image_bgr, p, max(3, thickness + 3), (15, 25, 35), 1, cv2.LINE_AA)

    if preview_only:
        label = f"#{index + 1}  person  {detection.confidence:.2f}"
    else:
        label = f"person {detection.confidence:.2f}"
        if annotation:
            attrs = annotation.get("attributes", {})
            age = attrs.get("age_group", "unknown")
            quality = attrs.get("quality", "unknown")
            ann_id = annotation.get("id", "?")
            label = f"ID {ann_id} | {age} | {quality}"
    _draw_label(image_bgr, label, max(0, x1), max(0, y1 - 8), thickness, color)


def draw_annotations(
    image_bgr: np.ndarray,
    detections: List[PersonDetection],
    annotations: List[Dict[str, Any]],
    keypoint_threshold: float = 0.35,
) -> np.ndarray:
    out = image_bgr.copy()
    for idx, (det, ann) in enumerate(zip(detections, annotations)):
        draw_detection(out, det, ann, keypoint_threshold=keypoint_threshold, index=idx)
    return out


def draw_detections_preview(
    image_bgr: np.ndarray,
    detections: List[PersonDetection],
    keypoint_threshold: float = 0.35,
) -> np.ndarray:
    out = image_bgr.copy()
    for idx, det in enumerate(detections):
        draw_detection(out, det, None, keypoint_threshold=keypoint_threshold, index=idx, preview_only=True)
    if not detections:
        h, w = out.shape[:2]
        msg = "No person detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.7, min(h, w) / 900)
        thickness = max(1, round(scale * 2))
        (tw, th), base = cv2.getTextSize(msg, font, scale, thickness)
        x = max(10, (w - tw) // 2)
        y = max(th + 20, h // 2)
        cv2.rectangle(out, (x - 16, y - th - base - 16), (x + tw + 16, y + base + 16), (18, 22, 30), -1)
        cv2.putText(out, msg, (x, y), font, scale, (240, 245, 255), thickness, cv2.LINE_AA)
    return out
