from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_ROOT / "models"


@dataclass(frozen=True)
class DetectorConfig: 
    """Configuration for person detection / pose estimation.

    backend:
        - "yolo": production YOLO11 pose backend. Requires yolo11n-pose.pt locally or internet on first run.
        - "opencv_hog": offline fallback detector. No keypoints, useful for dependency tests.
        - "auto": try YOLO first, then fall back to OpenCV HOG if YOLO weights are unavailable.
    """

    backend: str = "yolo"
    model_path: str = "yolo11n-pose.pt"
    image_size: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.70
    person_class_id: int = 0
    max_det: int = 300
    device: str | None = None
    use_person_class_filter: bool = True
    allow_auto_fallback: bool = False
    hog_hit_threshold: float = 0.0
    hog_nms_threshold: float = 0.45


@dataclass(frozen=True)
class ClassifierConfig:
    """Configuration for MobileNetV4 attribute inference.

    The bundled checkpoint may be age-only, gender-only, or an age+gender head depending on the
    checkpoint you provide. PDC v3 auto-detects common layouts:
        - 6 logits: age groups
        - 8 logits: age[0:6] + gender[6:8]
        - 2 logits: gender
        - two ONNX outputs: age output + gender output

    No-retraining stabilizers:
        - TTA averages probabilities across original/flipped/center-cropped/expanded crop variants.
        - Low-confidence unknown behavior is opt-in through strict_unknown flags.
        - Margin thresholds can hide genuinely ambiguous outputs without punishing ordinary low-ish softmax.
    """

    backend: str = "torch"  # torch or onnx
    pth_path: Path = MODELS_DIR / "mobilenetv4_utkface_age_gender_best.pth"
    onnx_path: Path = MODELS_DIR / "mobilenetv4_utkface_age_gender.onnx"
    timm_model_name: str = "mobilenetv4_conv_small.e3600_r256_in1k"
    num_classes: int = 6
    input_size: Tuple[int, int] = (256, 256)
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    min_confidence: float = 0.15
    min_gender_confidence: float = 0.15
    min_age_margin: float = 0.00
    min_gender_margin: float = 0.00
    unknown_on_low_confidence: bool = False
    layout: str = "auto"  # auto, age6, age6_gender2, gender2, age6_gender2_multi_output
    device: str | None = None
    use_tta: bool = True
    tta_use_hflip: bool = True
    tta_center_crop_ratio: float = 0.90
    tta_expand_canvas_ratio: float = 1.10


@dataclass(frozen=True)
class AttributeConfig:
    """Configuration for heuristic person-attribute extraction and classifier crop selection."""

    keypoint_conf_threshold: float = 0.35
    min_crop_width: int = 25
    min_crop_height: int = 40
    min_classify_crop_width: int = 32
    min_classify_crop_height: int = 32
    crop_padding_ratio: float = 0.08
    classifier_crop_mode: str = "auto_face"  # auto_face, face, head, upper_body, person
    face_crop_padding_ratio: float = 0.72
    head_crop_padding_ratio: float = 0.28
    upper_body_fraction: float = 0.52
    blur_low_threshold: float = 35.0
    blur_medium_threshold: float = 80.0
    classifier_blur_min_threshold: float = 8.0
    high_quality_min_height: int = 160
    medium_quality_min_height: int = 80
    bbox_boundary_margin_px: int = 2
    low_quality_still_classify: bool = True
    # v3 crop rectification controls
    enable_face_detector_crop: bool = True
    face_detector_min_neighbors: int = 4
    face_detector_scale_factor: float = 1.08
    face_detector_min_face_ratio: float = 0.035
    max_faces_inside_person: int = 6
    multi_person_overlap_threshold: float = 0.12
    multi_person_force_face_crop: bool = True
    save_classifier_crop_debug: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    attributes: AttributeConfig = field(default_factory=AttributeConfig)
    save_visualization: bool = True
    round_digits: int = 2
    include_debug: bool = False
