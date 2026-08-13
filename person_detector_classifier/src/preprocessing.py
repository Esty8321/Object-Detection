from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Tuple

import cv2
import numpy as np
import torch


@dataclass
class ClassifierCropResult:
    crop_bgr: np.ndarray | None
    bbox_xyxy: tuple[int, int, int, int] | None
    source: str
    quality: str = "unknown"
    reason: str = ""
    crowded: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)


def clamp_bbox_xyxy(
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def expand_bbox_xyxy(
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clamp_bbox_xyxy(bbox_xyxy, image_width, image_height)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    pad = padding_ratio * max(bw, bh)
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    x1, y1, x2, y2 = clamp_bbox_xyxy((x1, y1, x2, y2), image_width, image_height)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def bbox_area_xyxy(bbox: tuple[float, float, float, float] | list[float]) -> float:
    x1, y1, x2, y2 = map(float, bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection_xyxy(
    a: tuple[float, float, float, float] | list[float],
    b: tuple[float, float, float, float] | list[float],
) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def bbox_iou_xyxy(
    a: tuple[float, float, float, float] | list[float],
    b: tuple[float, float, float, float] | list[float],
) -> float:
    inter = bbox_intersection_xyxy(a, b)
    union = bbox_area_xyxy(a) + bbox_area_xyxy(b) - inter
    return 0.0 if union <= 0 else float(inter / union)


def crop_bgr(image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def preprocess_bgr_crop_for_mobilenet(
    crop_bgr: np.ndarray,
    input_size: Tuple[int, int] = (256, 256),
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Preprocess BGR crop to NCHW torch tensor matching the training script."""
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty crop passed to MobileNetV4 preprocessor")
    resized = cv2.resize(crop_bgr, input_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean_arr) / std_arr
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).float()


def laplacian_blur_score(crop_bgr: np.ndarray) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness_score(crop_bgr: np.ndarray | None) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _crop_quality_for_classifier(crop: np.ndarray | None, min_w: int = 32, min_h: int = 32, blur_min: float = 8.0) -> tuple[str, str, dict[str, float | int]]:
    if crop is None or crop.size == 0:
        return "reject", "empty", {"width": 0, "height": 0, "blur": 0.0, "brightness": 0.0}
    h, w = crop.shape[:2]
    blur = laplacian_blur_score(crop)
    brightness = _brightness_score(crop)
    metrics: dict[str, float | int] = {"width": int(w), "height": int(h), "blur": round(float(blur), 3), "brightness": round(float(brightness), 3)}
    if w < min_w or h < min_h:
        return "reject", "too_small", metrics
    if brightness < 18 or brightness > 242:
        return "weak", "extreme_brightness", metrics
    if blur < blur_min:
        return "weak", "blur_low", metrics
    if w >= 90 and h >= 90 and blur >= 35:
        return "high", "ok", metrics
    if w >= 56 and h >= 56:
        return "medium", "ok", metrics
    return "low", "ok", metrics


def _visible_keypoint_xy(keypoints: np.ndarray | None, idx: int, threshold: float) -> tuple[float, float] | None:
    if keypoints is None or idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    if len(kp) >= 3 and float(kp[2]) >= threshold:
        return float(kp[0]), float(kp[1])
    return None


def _keypoint_head_center(keypoints: np.ndarray | None, threshold: float) -> tuple[float, float] | None:
    if keypoints is None:
        return None
    pts = []
    for idx in (0, 1, 2, 3, 4):
        pt = _visible_keypoint_xy(keypoints, idx, threshold)
        if pt is not None:
            pts.append(pt)
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def _box_center(box: tuple[float, float, float, float] | list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _detect_faces_opencv(
    image_bgr: np.ndarray,
    min_neighbors: int = 4,
    scale_factor: float = 1.08,
    min_face_ratio: float = 0.035,
    max_faces: int = 6,
) -> list[tuple[int, int, int, int]]:
    """Return face boxes in xyxy coordinates in the crop coordinate system."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    min_side = max(20, int(min(h, w) * min_face_ratio))
    cascades = []
    try:
        frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        if not frontal.empty():
            cascades.append(frontal)
        if not profile.empty():
            cascades.append(profile)
    except Exception:
        return []

    faces: list[tuple[int, int, int, int]] = []
    for cascade in cascades:
        found = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, fw, fh in found:
            if fw <= 0 or fh <= 0:
                continue
            faces.append((int(x), int(y), int(x + fw), int(y + fh)))

    if not faces:
        return []

    # Merge duplicates using lightweight NMS.
    boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in faces]
    scores = [float((x2 - x1) * (y2 - y1)) for x1, y1, x2, y2 in faces]
    keep = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=0.35)
    if keep is None or len(keep) == 0:
        return []
    kept = [faces[int(i)] for i in np.array(keep).reshape(-1).tolist()]
    kept.sort(key=lambda b: ((b[2] - b[0]) * (b[3] - b[1])), reverse=True)
    return kept[:max_faces]


def _candidate_record(source: str, bbox: tuple[int, int, int, int] | None, quality: str, reason: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "source": source,
        "bbox_xyxy": list(bbox) if bbox is not None else None,
        "quality": quality,
        "reason": reason,
        "metrics": metrics or {},
    }


def _score_face_box(
    face_box_abs: tuple[int, int, int, int],
    selected_bbox: tuple[float, float, float, float] | list[float],
    keypoints: np.ndarray | None,
    threshold: float,
) -> float:
    fx1, fy1, fx2, fy2 = face_box_abs
    area = max(1.0, float((fx2 - fx1) * (fy2 - fy1)))
    fc = _box_center(face_box_abs)
    head_center = _keypoint_head_center(keypoints, threshold)
    if head_center is not None:
        dx = fc[0] - head_center[0]
        dy = fc[1] - head_center[1]
        dist_penalty = float(np.hypot(dx, dy)) * 2.0
    else:
        x1, y1, x2, y2 = map(float, selected_bbox)
        target = ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.20)
        dist_penalty = float(np.hypot(fc[0] - target[0], fc[1] - target[1])) * 0.8
    return area - dist_penalty


def _face_detector_crop(
    image_bgr: np.ndarray,
    person_box: tuple[int, int, int, int],
    selected_bbox: tuple[float, float, float, float] | list[float],
    keypoints: np.ndarray | None,
    keypoint_threshold: float,
    face_padding_ratio: float,
    min_neighbors: int,
    scale_factor: float,
    min_face_ratio: float,
    max_faces_inside_person: int,
    min_w: int,
    min_h: int,
    blur_min: float,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, str, str, dict[str, Any]]:
    px1, py1, px2, py2 = person_box
    person_crop = crop_bgr(image_bgr, person_box)
    if person_crop is None:
        return None, None, "reject", "empty_person_crop", {}
    faces_rel = _detect_faces_opencv(
        person_crop,
        min_neighbors=min_neighbors,
        scale_factor=scale_factor,
        min_face_ratio=min_face_ratio,
        max_faces=max_faces_inside_person,
    )
    if not faces_rel:
        return None, None, "reject", "no_face_detected", {"faces": 0}
    faces_abs = [(px1 + x1, py1 + y1, px1 + x2, py1 + y2) for x1, y1, x2, y2 in faces_rel]
    faces_abs.sort(key=lambda b: _score_face_box(b, selected_bbox, keypoints, keypoint_threshold), reverse=True)
    image_h, image_w = image_bgr.shape[:2]
    best = faces_abs[0]
    x1, y1, x2, y2 = best
    fw = max(1.0, x2 - x1)
    fh = max(1.0, y2 - y1)
    pcw = max(1.0, px2 - px1)
    pch = max(1.0, py2 - py1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    desired_w = fw * (1.0 + 2.0 * face_padding_ratio)
    desired_h = fh * (1.0 + face_padding_ratio * 1.95)
    # Prevent close-up portraits or bad Haar boxes from expanding into the whole person/image.
    # The classifier needs face/head context, not an accidental full-body crop wearing a fake mustache.
    max_w = max(64.0, min(pcw * 0.64, image_w * 0.92))
    max_h = max(64.0, min(pch * 0.54, image_h * 0.82))
    if pch > pcw * 1.25:
        max_h = min(max_h, pch * 0.46)
    crop_w = max(min(desired_w, max_w), min(64.0, pcw))
    crop_h = max(min(desired_h, max_h), min(64.0, pch))
    face_crop_box = expand_bbox_xyxy(
        (cx - crop_w / 2.0, cy - crop_h * 0.45, cx + crop_w / 2.0, cy + crop_h * 0.55),
        image_width=image_w,
        image_height=image_h,
        padding_ratio=0.0,
    )
    crop = crop_bgr(image_bgr, face_crop_box)
    quality, reason, metrics = _crop_quality_for_classifier(crop, min_w=min_w, min_h=min_h, blur_min=blur_min)
    metrics["faces_inside_person"] = len(faces_abs)
    return crop, face_crop_box, quality, reason, metrics


def _keypoint_face_crop(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    keypoints: np.ndarray | None,
    keypoint_threshold: float,
    face_padding_ratio: float,
    min_w: int,
    min_h: int,
    blur_min: float,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, str, str, dict[str, Any]]:
    if keypoints is None or len(keypoints) < 5:
        return None, None, "reject", "no_keypoints", {}
    image_h, image_w = image_bgr.shape[:2]
    points = []
    for idx in (0, 1, 2, 3, 4):
        pt = _visible_keypoint_xy(keypoints, idx, keypoint_threshold)
        if pt is not None:
            points.append(pt)
    if len(points) < 2:
        return None, None, "reject", "too_few_face_keypoints", {"points": len(points)}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    person_box = expand_bbox_xyxy(bbox_xyxy, image_width=image_w, image_height=image_h, padding_ratio=0.02)
    bw = max(float(person_box[2] - person_box[0]), 1.0)
    bh = max(float(person_box[3] - person_box[1]), 1.0)
    head_w = max(x2 - x1, bw * 0.15, 34.0)
    head_h = max(y2 - y1, bh * 0.14, 34.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(head_w, head_h) * (1.0 + face_padding_ratio)
    crop_box = (cx - side / 2.0, cy - side * 0.42, cx + side / 2.0, cy + side * 0.62)
    crop_box_i = expand_bbox_xyxy(crop_box, image_width=image_w, image_height=image_h, padding_ratio=0.0)
    crop = crop_bgr(image_bgr, crop_box_i)
    quality, reason, metrics = _crop_quality_for_classifier(crop, min_w=min_w, min_h=min_h, blur_min=blur_min)
    metrics["face_keypoints"] = len(points)
    return crop, crop_box_i, quality, reason, metrics


def _shoulder_head_crop(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    keypoints: np.ndarray | None,
    keypoint_threshold: float,
    head_padding_ratio: float,
    min_w: int,
    min_h: int,
    blur_min: float,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, str, str, dict[str, Any]]:
    image_h, image_w = image_bgr.shape[:2]
    person_box = expand_bbox_xyxy(bbox_xyxy, image_width=image_w, image_height=image_h, padding_ratio=0.02)
    left_shoulder = _visible_keypoint_xy(keypoints, 5, keypoint_threshold)
    right_shoulder = _visible_keypoint_xy(keypoints, 6, keypoint_threshold)
    if left_shoulder is None and right_shoulder is None:
        return None, None, "reject", "no_shoulder_keypoints", {}
    shoulders = [p for p in (left_shoulder, right_shoulder) if p is not None]
    sx = float(np.mean([p[0] for p in shoulders]))
    sy = float(np.mean([p[1] for p in shoulders]))
    x1, y1, x2, y2 = map(float, person_box)
    if len(shoulders) == 2:
        shoulder_span = max(36.0, abs(right_shoulder[0] - left_shoulder[0]))  # type: ignore[index]
    else:
        shoulder_span = max(36.0, (x2 - x1) * 0.34)
    head_h = max(42.0, shoulder_span * 1.28)
    head_w = max(42.0, shoulder_span * 1.12)
    crop_box = (
        sx - head_w * (0.5 + head_padding_ratio),
        sy - head_h * (1.15 + head_padding_ratio),
        sx + head_w * (0.5 + head_padding_ratio),
        sy + head_h * 0.18,
    )
    crop_box_i = expand_bbox_xyxy(crop_box, image_width=image_w, image_height=image_h, padding_ratio=0.0)
    crop = crop_bgr(image_bgr, crop_box_i)
    quality, reason, metrics = _crop_quality_for_classifier(crop, min_w=min_w, min_h=min_h, blur_min=blur_min)
    metrics["shoulder_points"] = len(shoulders)
    return crop, crop_box_i, quality, reason, metrics


def _upper_body_crop(
    image_bgr: np.ndarray,
    person_box: tuple[int, int, int, int],
    upper_body_fraction: float,
    min_w: int,
    min_h: int,
    blur_min: float,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, str, str, dict[str, Any]]:
    x1, y1, x2, y2 = person_box
    h = max(1, y2 - y1)
    upper_y2 = int(round(y1 + h * max(0.25, min(0.85, upper_body_fraction))))
    crop_box_i = (x1, y1, x2, max(y1 + 1, upper_y2))
    crop = crop_bgr(image_bgr, crop_box_i)
    quality, reason, metrics = _crop_quality_for_classifier(crop, min_w=min_w, min_h=min_h, blur_min=blur_min)
    return crop, crop_box_i, quality, reason, metrics


def _person_crop(
    image_bgr: np.ndarray,
    person_box: tuple[int, int, int, int],
    min_w: int,
    min_h: int,
    blur_min: float,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, str, str, dict[str, Any]]:
    crop = crop_bgr(image_bgr, person_box)
    quality, reason, metrics = _crop_quality_for_classifier(crop, min_w=min_w, min_h=min_h, blur_min=blur_min)
    return crop, person_box, quality, reason, metrics


def _detect_crowded_selected_bbox(
    selected_bbox: tuple[float, float, float, float] | list[float],
    all_bboxes: Iterable[tuple[float, float, float, float] | list[float]] | None,
    overlap_threshold: float,
) -> bool:
    if all_bboxes is None:
        return False
    selected_area = max(1.0, bbox_area_xyxy(selected_bbox))
    sx1, sy1, sx2, sy2 = map(float, selected_bbox)
    scx, scy = _box_center(selected_bbox)
    for other in all_bboxes:
        if list(map(float, other)) == list(map(float, selected_bbox)):
            continue
        inter = bbox_intersection_xyxy(selected_bbox, other)
        if inter / selected_area >= overlap_threshold:
            return True
        ox1, oy1, ox2, oy2 = map(float, other)
        ocx, ocy = _box_center(other)
        # Another person's center lies inside selected bbox or selected center lies inside another bbox.
        if sx1 <= ocx <= sx2 and sy1 <= ocy <= sy2:
            return True
        if ox1 <= scx <= ox2 and oy1 <= scy <= oy2 and inter > 0:
            return True
    return False


def _make_tta_crop_variants(crop_bgr: np.ndarray, center_crop_ratio: float = 0.90, expand_canvas_ratio: float = 1.10, use_hflip: bool = True) -> list[np.ndarray]:
    """Build cheap test-time augmentation variants for classifier probability averaging."""
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    variants = [crop_bgr]
    h, w = crop_bgr.shape[:2]
    if use_hflip:
        variants.append(cv2.flip(crop_bgr, 1))
    if 0.50 <= center_crop_ratio < 1.0 and h > 8 and w > 8:
        ch = max(1, int(round(h * center_crop_ratio)))
        cw = max(1, int(round(w * center_crop_ratio)))
        y1 = max(0, (h - ch) // 2)
        x1 = max(0, (w - cw) // 2)
        variants.append(crop_bgr[y1:y1 + ch, x1:x1 + cw].copy())
    if expand_canvas_ratio > 1.0:
        pad_y = int(round(h * (expand_canvas_ratio - 1.0) / 2.0))
        pad_x = int(round(w * (expand_canvas_ratio - 1.0) / 2.0))
        expanded = cv2.copyMakeBorder(crop_bgr, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE)
        variants.append(expanded)
    return variants


def build_classifier_crop(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    keypoints: np.ndarray | None = None,
    all_bboxes_xyxy: Iterable[tuple[float, float, float, float] | list[float]] | None = None,
    mode: str = "auto_face",
    keypoint_threshold: float = 0.35,
    face_padding_ratio: float = 0.72,
    head_padding_ratio: float = 0.28,
    upper_body_fraction: float = 0.52,
    min_classify_crop_width: int = 32,
    min_classify_crop_height: int = 32,
    blur_min_threshold: float = 8.0,
    enable_face_detector_crop: bool = True,
    face_detector_min_neighbors: int = 4,
    face_detector_scale_factor: float = 1.08,
    face_detector_min_face_ratio: float = 0.035,
    max_faces_inside_person: int = 6,
    multi_person_overlap_threshold: float = 0.12,
    multi_person_force_face_crop: bool = True,
) -> ClassifierCropResult:
    """Build the crop that is actually fed to MobileNetV4.

    The no-retraining strategy is simple: make inference crops look like the training distribution.
    Face/head crops come before upper/full-body crops, and crowded scenes force face/head-only crops
    so the classifier never sees two different people while pretending this is science.
    """
    if image_bgr is None or image_bgr.size == 0:
        return ClassifierCropResult(None, None, "empty_image", "reject", "empty_image")

    image_h, image_w = image_bgr.shape[:2]
    mode = (mode or "auto_face").lower().strip()
    if mode == "auto":
        mode = "auto_face"

    person_box = expand_bbox_xyxy(bbox_xyxy, image_width=image_w, image_height=image_h, padding_ratio=0.08)
    crowded = _detect_crowded_selected_bbox(bbox_xyxy, all_bboxes_xyxy, multi_person_overlap_threshold)
    candidates: list[dict[str, Any]] = []
    accepted_weak: ClassifierCropResult | None = None

    def try_candidate(source: str, fn) -> ClassifierCropResult | None:
        nonlocal accepted_weak
        crop, box, quality, reason, metrics = fn()
        candidates.append(_candidate_record(source, box, quality, reason, metrics))
        if crop is None or box is None or quality == "reject":
            return None
        result = ClassifierCropResult(crop, box, source, quality, reason, crowded, candidates)
        if quality in {"high", "medium", "low"}:
            return result
        # Keep weak crop if nothing better is found. Better weak face than full-body soup.
        if accepted_weak is None:
            accepted_weak = result
        return None

    force_face_like = crowded and multi_person_force_face_crop

    if mode in {"auto_face", "face", "head"}:
        if enable_face_detector_crop:
            result = try_candidate(
                "face_detector",
                lambda: _face_detector_crop(
                    image_bgr=image_bgr,
                    person_box=person_box,
                    selected_bbox=bbox_xyxy,
                    keypoints=keypoints,
                    keypoint_threshold=keypoint_threshold,
                    face_padding_ratio=face_padding_ratio,
                    min_neighbors=face_detector_min_neighbors,
                    scale_factor=face_detector_scale_factor,
                    min_face_ratio=face_detector_min_face_ratio,
                    max_faces_inside_person=max_faces_inside_person,
                    min_w=min_classify_crop_width,
                    min_h=min_classify_crop_height,
                    blur_min=blur_min_threshold,
                ),
            )
            if result is not None:
                return result

        result = try_candidate(
            "face_keypoints",
            lambda: _keypoint_face_crop(
                image_bgr=image_bgr,
                bbox_xyxy=bbox_xyxy,
                keypoints=keypoints,
                keypoint_threshold=keypoint_threshold,
                face_padding_ratio=face_padding_ratio,
                min_w=min_classify_crop_width,
                min_h=min_classify_crop_height,
                blur_min=blur_min_threshold,
            ),
        )
        if result is not None:
            return result

        result = try_candidate(
            "head_from_shoulders",
            lambda: _shoulder_head_crop(
                image_bgr=image_bgr,
                bbox_xyxy=bbox_xyxy,
                keypoints=keypoints,
                keypoint_threshold=keypoint_threshold,
                head_padding_ratio=head_padding_ratio,
                min_w=min_classify_crop_width,
                min_h=min_classify_crop_height,
                blur_min=blur_min_threshold,
            ),
        )
        if result is not None:
            return result

    if force_face_like:
        if accepted_weak is not None:
            accepted_weak.candidates = candidates
            accepted_weak.reason = accepted_weak.reason or "crowded_weak_face_like"
            return accepted_weak
        return ClassifierCropResult(None, None, "crowded_no_clean_face", "reject", "crowded_scene_requires_face_or_head", crowded, candidates)

    if mode in {"auto_face", "upper_body"}:
        result = try_candidate(
            "upper_body_fallback",
            lambda: _upper_body_crop(
                image_bgr=image_bgr,
                person_box=person_box,
                upper_body_fraction=upper_body_fraction,
                min_w=min_classify_crop_width,
                min_h=min_classify_crop_height,
                blur_min=blur_min_threshold,
            ),
        )
        if result is not None:
            return result

    if mode in {"auto_face", "person"}:
        result = try_candidate(
            "person_bbox_fallback",
            lambda: _person_crop(
                image_bgr=image_bgr,
                person_box=person_box,
                min_w=min_classify_crop_width,
                min_h=min_classify_crop_height,
                blur_min=blur_min_threshold,
            ),
        )
        if result is not None:
            return result

    if accepted_weak is not None:
        accepted_weak.candidates = candidates
        return accepted_weak

    return ClassifierCropResult(None, None, "no_valid_classifier_crop", "reject", "all_candidates_rejected", crowded, candidates)


# Public helper used by classifier.py. Kept here so preprocessing/crop policy stays in one place.
def make_tta_crop_variants(crop_bgr: np.ndarray, center_crop_ratio: float = 0.90, expand_canvas_ratio: float = 1.10, use_hflip: bool = True) -> list[np.ndarray]:
    return _make_tta_crop_variants(crop_bgr, center_crop_ratio=center_crop_ratio, expand_canvas_ratio=expand_canvas_ratio, use_hflip=use_hflip)
