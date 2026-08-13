from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from person_detector_classifier.config import AttributeConfig
from person_detector_classifier.src.detector import PersonDetection
from person_detector_classifier.src.preprocessing import laplacian_blur_score


# COCO keypoint indices used by Ultralytics pose models.
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

HEAD = {NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR}
SHOULDERS = {LEFT_SHOULDER, RIGHT_SHOULDER}
ARMS = {LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST}
HIPS = {LEFT_HIP, RIGHT_HIP}
LEGS = {LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE}
UPPER_BODY = HEAD | SHOULDERS | ARMS
LOWER_BODY = HIPS | LEGS


@dataclass
class AttributeResult:
    visibility: str
    occlusion: str
    occluded: bool
    quality: str
    rotation: float
    debug: Dict[str, float | int | bool | str] | None = None


def _visible_indices(keypoints: np.ndarray | None, threshold: float) -> set[int]:
    if keypoints is None or len(keypoints) == 0:
        return set()
    visible: set[int] = set()
    for idx, kp in enumerate(keypoints):
        if len(kp) >= 3 and float(kp[2]) >= threshold:
            visible.add(idx)
    return visible


def _count_visible(indices: Iterable[int], visible: set[int]) -> int:
    return sum(1 for i in indices if i in visible)


def estimate_visibility(keypoints: np.ndarray | None, threshold: float) -> str:
    visible = _visible_indices(keypoints, threshold)
    if not visible:
        return "unknown"

    head_count = _count_visible(HEAD, visible)
    shoulder_count = _count_visible(SHOULDERS, visible)
    upper_count = _count_visible(UPPER_BODY, visible)
    hip_count = _count_visible(HIPS, visible)
    leg_count = _count_visible(LEGS, visible)
    lower_count = _count_visible(LOWER_BODY, visible)
    total_count = len(visible)

    if shoulder_count >= 1 and hip_count >= 1 and leg_count >= 2:
        return "full_body"
    if (head_count >= 1 or shoulder_count >= 1) and upper_count >= 3 and lower_count <= 2:
        return "upper_body"
    if lower_count >= 3 and upper_count <= 2:
        return "lower_body"
    if total_count >= 3:
        return "partial"
    return "unknown"


def _bbox_touches_boundary(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin_px: int,
) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        x1 <= margin_px
        or y1 <= margin_px
        or x2 >= image_width - 1 - margin_px
        or y2 >= image_height - 1 - margin_px
    )


def estimate_occlusion(
    detection: PersonDetection,
    image_width: int,
    image_height: int,
    threshold: float,
    boundary_margin_px: int,
) -> tuple[str, bool, float, bool]:
    kps = detection.keypoints
    if kps is None or len(kps) == 0:
        boundary = _bbox_touches_boundary(detection.bbox_xyxy, image_width, image_height, boundary_margin_px)
        return ("partial" if boundary else "unknown", boundary, 1.0, boundary)

    visible = _visible_indices(kps, threshold)
    missing_ratio = 1.0 - (len(visible) / 17.0)
    boundary = _bbox_touches_boundary(detection.bbox_xyxy, image_width, image_height, boundary_margin_px)

    if missing_ratio <= 0.30 and not boundary:
        return "none", False, missing_ratio, boundary
    if missing_ratio <= 0.70:
        return "partial", True, missing_ratio, boundary
    return "heavy", True, missing_ratio, boundary


def estimate_rotation(keypoints: np.ndarray | None, threshold: float) -> float:
    """Estimate in-plane body rotation conservatively.

    v2 could show 90 degrees for upright portrait people when only a weak pair was usable. v3 only
    trusts near-horizontal shoulder/hip pairs. Suspicious vertical angles are neutralized to 0.0 so
    the UI stops accusing standing people of being furniture.
    """
    if keypoints is None or len(keypoints) < 17:
        return 0.0

    def point_pair_angle(idx_a: int, idx_b: int) -> Optional[float]:
        a = keypoints[idx_a]
        b = keypoints[idx_b]
        if len(a) < 3 or len(b) < 3:
            return None
        if float(a[2]) < threshold or float(b[2]) < threshold:
            return None
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        if abs(dx) < 4.0 and abs(dy) < 4.0:
            return None
        angle = math.degrees(math.atan2(dy, dx))
        # Shoulder/hip lines for upright people should be roughly horizontal. Reject vertical-ish
        # lines caused by wrong keypoint pairing or severe pose failure.
        if abs(angle) > 45.0:
            return None
        return angle

    angle = point_pair_angle(LEFT_SHOULDER, RIGHT_SHOULDER)
    if angle is None:
        angle = point_pair_angle(LEFT_HIP, RIGHT_HIP)
    if angle is None:
        return 0.0
    return float(max(-45.0, min(45.0, angle)))


def estimate_quality(
    crop_bgr: np.ndarray | None,
    detection: PersonDetection,
    keypoint_threshold: float,
    blur_low_threshold: float,
    blur_medium_threshold: float,
    high_quality_min_height: int,
    medium_quality_min_height: int,
) -> tuple[str, float, int]:
    if crop_bgr is None or crop_bgr.size == 0:
        return "low", 0.0, 0

    crop_h, crop_w = crop_bgr.shape[:2]
    blur = laplacian_blur_score(crop_bgr)
    visible_count = len(_visible_indices(detection.keypoints, keypoint_threshold))
    conf = detection.confidence

    if (
        crop_h >= high_quality_min_height
        and crop_w >= 48
        and blur >= blur_medium_threshold
        and conf >= 0.60
        and visible_count >= 8
    ):
        return "high", blur, visible_count

    if crop_h >= medium_quality_min_height and crop_w >= 32 and blur >= blur_low_threshold and conf >= 0.35:
        return "medium", blur, visible_count

    return "low", blur, visible_count


def extract_attributes(
    detection: PersonDetection,
    image_width: int,
    image_height: int,
    crop_bgr: np.ndarray | None,
    config: AttributeConfig | None = None,
    include_debug: bool = False,
) -> AttributeResult:
    cfg = config or AttributeConfig()
    visibility = estimate_visibility(detection.keypoints, cfg.keypoint_conf_threshold)
    occlusion, occluded, missing_ratio, touches_boundary = estimate_occlusion(
        detection,
        image_width=image_width,
        image_height=image_height,
        threshold=cfg.keypoint_conf_threshold,
        boundary_margin_px=cfg.bbox_boundary_margin_px,
    )
    rotation = estimate_rotation(detection.keypoints, cfg.keypoint_conf_threshold)
    quality, blur, visible_count = estimate_quality(
        crop_bgr,
        detection,
        keypoint_threshold=cfg.keypoint_conf_threshold,
        blur_low_threshold=cfg.blur_low_threshold,
        blur_medium_threshold=cfg.blur_medium_threshold,
        high_quality_min_height=cfg.high_quality_min_height,
        medium_quality_min_height=cfg.medium_quality_min_height,
    )

    debug = None
    if include_debug:
        debug = {
            "bbox_confidence": round(float(detection.confidence), 6),
            "keypoints_visible": int(visible_count),
            "keypoints_missing_ratio": round(float(missing_ratio), 6),
            "touches_image_boundary": bool(touches_boundary),
            "blur_laplacian_var": round(float(blur), 6),
        }

    return AttributeResult(
        visibility=visibility,
        occlusion=occlusion,
        occluded=occluded,
        quality=quality,
        rotation=rotation,
        debug=debug,
    )
