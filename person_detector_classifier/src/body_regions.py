from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from person_detector_classifier.src.preprocessing import (
    build_classifier_crop,
    crop_bgr,
    expand_bbox_xyxy,
    laplacian_blur_score,
)


# COCO / Ultralytics 17-keypoint indices.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

REGION_ORDER = (
    "head",
    "upper_body",
    "left_hand",
    "right_hand",
    "waist",
    "hips",
    "left_leg",
    "right_leg",
)


@dataclass
class BodyRegionResult:
    name: str
    crop_bgr: np.ndarray | None = field(repr=False)
    bbox_xyxy: tuple[int, int, int, int] | None
    status: str                       # available | unavailable
    source: str
    quality: str = "unknown"
    reason: str = ""
    keypoints_used: list[int] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "source": self.source,
            "quality": self.quality,
            "reason": self.reason,
            "keypoints_used": self.keypoints_used,
            "candidates": self.candidates,
        }


def _point(keypoints: np.ndarray | None, index: int, threshold: float) -> tuple[float, float] | None:
    if keypoints is None or index >= len(keypoints) or len(keypoints[index]) < 3:
        return None
    kp = keypoints[index]
    if float(kp[2]) < threshold:
        return None
    return float(kp[0]), float(kp[1])


def _quality(crop: np.ndarray | None, min_width: int, min_height: int) -> tuple[str, str]:
    if crop is None or crop.size == 0:
        return "reject", "empty_crop"
    height, width = crop.shape[:2]
    if width < min_width or height < min_height:
        return "reject", "crop_too_small"
    blur = laplacian_blur_score(crop)
    if blur < 8.0:
        return "weak", "blur_low"
    if width >= 90 and height >= 90 and blur >= 35.0:
        return "high", "ok"
    if width >= 56 and height >= 56:
        return "medium", "ok"
    return "low", "ok"


def _box_from_points(
    points: Iterable[tuple[float, float]],
    image_width: int,
    image_height: int,
    pad_x: float,
    pad_y: float,
    minimum_span: float = 12.0,
) -> tuple[int, int, int, int] | None:
    pts = list(points)
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    span_x = max(max(xs) - min(xs), minimum_span)
    span_y = max(max(ys) - min(ys), minimum_span)
    box = (
        min(xs) - span_x * pad_x,
        min(ys) - span_y * pad_y,
        max(xs) + span_x * pad_x,
        max(ys) + span_y * pad_y,
    )
    return expand_bbox_xyxy(box, image_width, image_height, padding_ratio=0.0)


def _segment_box(
    a: tuple[float, float],
    b: tuple[float, float],
    image_width: int,
    image_height: int,
    width_ratio: float,
) -> tuple[int, int, int, int] | None:
    """Axis-aligned envelope around an oriented limb segment."""
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if length < 6.0:
        return None
    pad = max(6.0, length * width_ratio)
    return expand_bbox_xyxy(
        (min(a[0], b[0]) - pad, min(a[1], b[1]) - pad,
         max(a[0], b[0]) + pad, max(a[1], b[1]) + pad),
        image_width,
        image_height,
        padding_ratio=0.0,
    )


def _build_result(
    name: str,
    image_bgr: np.ndarray,
    candidates: list[tuple[str, tuple[int, int, int, int] | None, list[int], str]],
    min_width: int,
    min_height: int,
) -> BodyRegionResult:
    history: list[dict[str, Any]] = []
    weak: BodyRegionResult | None = None
    for source, box, used, unavailable_reason in candidates:
        crop = crop_bgr(image_bgr, box) if box else None
        quality, reason = _quality(crop, min_width, min_height)
        if box is None:
            reason = unavailable_reason
        history.append({
            "source": source,
            "bbox_xyxy": list(box) if box else None,
            "quality": quality,
            "reason": reason,
            "keypoints_used": used,
        })
        if box is None or crop is None or quality == "reject":
            continue
        result = BodyRegionResult(name, crop, box, "available", source, quality, reason, used, history.copy())
        if quality != "weak":
            return result
        if weak is None:
            weak = result
    if weak is not None:
        weak.candidates = history
        return weak
    reason = history[-1]["reason"] if history else "no_candidate"
    return BodyRegionResult(name, None, None, "unavailable", "none", "reject", reason, [], history)


def extract_body_regions(
    image_bgr: np.ndarray,
    person_bbox_xyxy: tuple[float, float, float, float] | list[float],
    keypoints: np.ndarray | None,
    all_bboxes_xyxy: Iterable[tuple[float, float, float, float] | list[float]] | None = None,
    keypoint_threshold: float = 0.35,
    min_width: int = 24,
    min_height: int = 24,
) -> dict[str, BodyRegionResult]:
    """Extract reliable per-person regions; unavailable parts remain explicit in JSON."""
    if image_bgr is None or image_bgr.size == 0:
        return {name: BodyRegionResult(name, None, None, "unavailable", "none", "reject", "empty_image") for name in REGION_ORDER}

    height, width = image_bgr.shape[:2]
    p = lambda idx: _point(keypoints, idx, keypoint_threshold)
    ls, rs, le, re, lw, rw = p(LEFT_SHOULDER), p(RIGHT_SHOULDER), p(LEFT_ELBOW), p(RIGHT_ELBOW), p(LEFT_WRIST), p(RIGHT_WRIST)
    lh, rh, lk, rk, la, ra = p(LEFT_HIP), p(RIGHT_HIP), p(LEFT_KNEE), p(RIGHT_KNEE), p(LEFT_ANKLE), p(RIGHT_ANKLE)

    results: dict[str, BodyRegionResult] = {}

    # Reuse the proven ordered head strategy: detector -> face keypoints -> shoulders.
    head = build_classifier_crop(
        image_bgr=image_bgr,
        bbox_xyxy=person_bbox_xyxy,
        keypoints=keypoints,
        all_bboxes_xyxy=all_bboxes_xyxy,
        mode="head",
        keypoint_threshold=keypoint_threshold,
        min_classify_crop_width=min_width,
        min_classify_crop_height=min_height,
    )
    if head.crop_bgr is not None and head.bbox_xyxy is not None:
        results["head"] = BodyRegionResult("head", head.crop_bgr, head.bbox_xyxy, "available", head.source, head.quality, head.reason, [], head.candidates)
    else:
        results["head"] = BodyRegionResult("head", None, None, "unavailable", head.source, "reject", head.reason, [], head.candidates)

    shoulder_points = [x for x in (ls, rs) if x]
    hip_points = [x for x in (lh, rh) if x]
    torso_points = shoulder_points + hip_points
    upper_box = _box_from_points(torso_points, width, height, 0.16, 0.08) if len(shoulder_points) >= 1 and len(hip_points) >= 1 else None
    upper_box_wide = _box_from_points(torso_points, width, height, 0.24, 0.14) if len(shoulder_points) >= 1 and len(hip_points) >= 1 else None
    results["upper_body"] = _build_result("upper_body", image_bgr, [
        ("shoulders_to_waist_keypoints", upper_box,
         [i for i, v in ((LEFT_SHOULDER, ls), (RIGHT_SHOULDER, rs), (LEFT_HIP, lh), (RIGHT_HIP, rh)) if v],
         "need_visible_shoulder_and_hip"),
        ("shoulders_to_waist_wide", upper_box_wide,
         [i for i, v in ((LEFT_SHOULDER, ls), (RIGHT_SHOULDER, rs), (LEFT_HIP, lh), (RIGHT_HIP, rh)) if v],
         "need_visible_shoulder_and_hip"),
    ], min_width, min_height)

    for name, elbow, wrist, indices in (
        ("left_hand", le, lw, [LEFT_ELBOW, LEFT_WRIST]),
        ("right_hand", re, rw, [RIGHT_ELBOW, RIGHT_WRIST]),
    ):
        tight = _segment_box(elbow, wrist, width, height, 0.18) if elbow and wrist else None
        box = _segment_box(elbow, wrist, width, height, 0.28) if elbow and wrist else None
        results[name] = _build_result(name, image_bgr, [
            ("elbow_to_wrist_tight", tight, indices if tight else [], "need_visible_elbow_and_wrist"),
            ("elbow_to_wrist_padded", box, indices if box else [], "need_visible_elbow_and_wrist"),
        ], min_width, min_height)

    # COCO has no waist joint. Estimate a narrow band at 72%-100% of shoulder-to-hip axes.
    waist_box = None
    if len(shoulder_points) >= 1 and len(hip_points) >= 1:
        shoulder_center = np.mean(np.asarray(shoulder_points), axis=0)
        hip_center = np.mean(np.asarray(hip_points), axis=0)
        axis = hip_center - shoulder_center
        waist_top = shoulder_center + axis * 0.70
        waist_bottom = shoulder_center + axis * 1.02
        shoulder_span = abs(rs[0] - ls[0]) if ls and rs else 0.0
        hip_span = abs(rh[0] - lh[0]) if lh and rh else 0.0
        torso_width = max(shoulder_span, hip_span, 18.0)
        axis_length = max(float(np.hypot(axis[0], axis[1])), 1.0)
        perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64) / axis_length
        half_width = torso_width * 0.56
        corners = [
            tuple(waist_top + perpendicular * half_width),
            tuple(waist_top - perpendicular * half_width),
            tuple(waist_bottom + perpendicular * half_width),
            tuple(waist_bottom - perpendicular * half_width),
        ]
        waist_box = _box_from_points(corners, width, height, 0.06, 0.10)
    results["waist"] = _build_result("waist", image_bgr, [
        ("interpolated_shoulders_to_hips", waist_box,
         [i for i, v in ((LEFT_SHOULDER, ls), (RIGHT_SHOULDER, rs), (LEFT_HIP, lh), (RIGHT_HIP, rh)) if v],
         "need_visible_shoulder_and_hip"),
    ], min_width, min_height)

    hips_points = hip_points + [x for x in (lk, rk) if x]
    hips_box = _box_from_points(hips_points, width, height, 0.14, 0.08) if hip_points and (lk or rk) else None
    hips_box_wide = _box_from_points(hips_points, width, height, 0.22, 0.13) if hip_points and (lk or rk) else None
    results["hips"] = _build_result("hips", image_bgr, [
        ("hips_to_knees_keypoints", hips_box,
         [i for i, v in ((LEFT_HIP, lh), (RIGHT_HIP, rh), (LEFT_KNEE, lk), (RIGHT_KNEE, rk)) if v],
         "need_visible_hip_and_knee"),
        ("hips_to_knees_wide", hips_box_wide,
         [i for i, v in ((LEFT_HIP, lh), (RIGHT_HIP, rh), (LEFT_KNEE, lk), (RIGHT_KNEE, rk)) if v],
         "need_visible_hip_and_knee"),
    ], min_width, min_height)

    for name, knee, ankle, indices in (
        ("left_leg", lk, la, [LEFT_KNEE, LEFT_ANKLE]),
        ("right_leg", rk, ra, [RIGHT_KNEE, RIGHT_ANKLE]),
    ):
        tight = _segment_box(knee, ankle, width, height, 0.20) if knee and ankle else None
        box = _segment_box(knee, ankle, width, height, 0.30) if knee and ankle else None
        results[name] = _build_result(name, image_bgr, [
            ("knee_to_ankle_tight", tight, indices if tight else [], "need_visible_knee_and_ankle"),
            ("knee_to_ankle_padded", box, indices if box else [], "need_visible_knee_and_ankle"),
        ], min_width, min_height)

    return {name: results[name] for name in REGION_ORDER}