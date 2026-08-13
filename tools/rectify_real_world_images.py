from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from person_detector_classifier.config import AttributeConfig, ClassifierConfig, DetectorConfig, PipelineConfig
from person_detector_classifier.src.pipeline import PersonDetectorClassifierPipeline
from person_detector_classifier.src.utils import list_images, save_json


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        detector=DetectorConfig(
            backend=args.detector_backend,
            model_path=args.yolo_model,
            conf_threshold=args.det_conf,
            iou_threshold=args.iou,
            image_size=args.imgsz,
            device=args.device,
            use_person_class_filter=not args.no_class_filter,
        ),
        classifier=ClassifierConfig(
            backend=args.classifier_backend,
            pth_path=Path(args.pth_model),
            onnx_path=Path(args.onnx_model),
            min_confidence=args.cls_conf,
            device=args.device,
        ),
        attributes=AttributeConfig(
            keypoint_conf_threshold=args.kp_conf,
            crop_padding_ratio=args.crop_padding,
            low_quality_still_classify=args.classify_low_quality,
        ),
        include_debug=True,
        save_visualization=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PDC_v2 over the provided real-world rectification images.")
    p.add_argument("--input-dir", default="test_assets/images_for_rectifying")
    p.add_argument("--output-dir", default="outputs/real_world_rectification")
    p.add_argument("--detector-backend", choices=["yolo", "opencv_hog", "auto"], default="opencv_hog")
    p.add_argument("--classifier-backend", choices=["torch", "onnx"], default="onnx")
    p.add_argument("--yolo-model", default="models/yolo11n-pose.pt")
    p.add_argument("--pth-model", default="person_detector_classifier/models/mobilenetv4_utkface.pth")
    p.add_argument("--onnx-model", default="person_detector_classifier/models/mobilenetv4_utkface.onnx")
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--det-conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--kp-conf", type=float, default=0.35)
    p.add_argument("--cls-conf", type=float, default=0.45)
    p.add_argument("--crop-padding", type=float, default=0.08)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--no-class-filter", action="store_true")
    p.add_argument("--classify-low-quality", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    pipe = PersonDetectorClassifierPipeline(config)
    images = list_images(args.input_dir, recursive=args.recursive)
    payload = pipe.process_folder(args.input_dir, args.output_dir, recursive=args.recursive, include_keypoints=True)

    summary = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "detector_backend": args.detector_backend,
        "classifier_backend": args.classifier_backend,
        "images_requested": len(images),
        "images_recorded": len(payload.get("images", [])),
        "annotations": len(payload.get("annotations", [])),
        "errors": payload.get("errors", []),
        "per_image_annotation_counts": {},
    }
    for img in payload.get("images", []):
        image_id = img["id"]
        summary["per_image_annotation_counts"][image_id] = sum(
            1 for ann in payload.get("annotations", []) if ann.get("image_id") == image_id
        )
    save_json(Path(args.output_dir) / "rectification_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
