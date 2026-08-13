from __future__ import annotations

import argparse
from pathlib import Path

from person_detector_classifier.config import (
    AttributeConfig,
    ClassifierConfig,
    DetectorConfig,
    PipelineConfig,
)
from person_detector_classifier.src.pipeline import PersonDetectorClassifierPipeline
from person_detector_classifier.src.utils import save_json


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        detector=DetectorConfig(
            backend=args.detector_backend,
            model_path=args.yolo_model,
            image_size=args.imgsz,
            conf_threshold=args.det_conf,
            iou_threshold=args.iou,
            device=args.device,
            use_person_class_filter=not args.no_class_filter,
            allow_auto_fallback=args.detector_backend == "auto",
        ),
        classifier=ClassifierConfig(
            backend=args.backend,
            pth_path=Path(args.pth_model),
            onnx_path=Path(args.onnx_model),
            min_confidence=args.cls_conf,
            min_gender_confidence=args.gender_conf,
            min_age_margin=args.age_margin,
            min_gender_margin=args.gender_margin,
            unknown_on_low_confidence=args.strict_unknown,
            device=args.device,
            use_tta=not args.no_tta,
        ),
        attributes=AttributeConfig(
            keypoint_conf_threshold=args.kp_conf,
            crop_padding_ratio=args.crop_padding,
            classifier_crop_mode=args.classifier_crop_mode,
            enable_face_detector_crop=not args.disable_face_detector,
            multi_person_overlap_threshold=args.multi_overlap_threshold,
            save_classifier_crop_debug=args.save_classifier_crops,
        ),
        save_visualization=not args.no_visualization,
        include_debug=args.debug,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDC_v3 on one image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="outputs", help="Directory for JSON and annotated image")
    parser.add_argument("--detector-backend", choices=["yolo", "opencv_hog", "auto"], default="yolo", help="Detector backend. Use auto/offline HOG only for testing when YOLO weights are unavailable.")
    parser.add_argument("--yolo-model", default="person_detector_classifier/models/yolo11n-pose.pt", help="YOLO pose model path/name")
    parser.add_argument("--pth-model", default="person_detector_classifier/models/mobilenetv4_utkface_age_gender_best.pth")
    parser.add_argument("--onnx-model", default="person_detector_classifier/models/mobilenetv4_utkface_age_gender.onnx")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu. Default auto-selects.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--kp-conf", type=float, default=0.35)
    parser.add_argument("--cls-conf", type=float, default=0.15)
    parser.add_argument("--gender-conf", type=float, default=0.15)
    parser.add_argument("--age-margin", type=float, default=0.0, help="Optional top1-top2 age margin below which age becomes unknown")
    parser.add_argument("--gender-margin", type=float, default=0.0, help="Optional top1-top2 gender margin below which gender becomes unknown")
    parser.add_argument("--crop-padding", type=float, default=0.08)
    parser.add_argument("--classifier-crop-mode", choices=["auto_face", "face", "head", "person", "upper_body"], default="auto_face", help="Crop sent into MobileNet. auto_face is best for face-trained MobileNet classifiers.")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA probability averaging")
    parser.add_argument("--disable-face-detector", action="store_true", help="Skip OpenCV face detector crop and rely on pose/keypoint crops")
    parser.add_argument("--multi-overlap-threshold", type=float, default=0.12, help="Overlap ratio that marks a selected bbox as crowded")
    parser.add_argument("--save-classifier-crops", action="store_true", help="With --debug, save the actual crops sent to MobileNet")
    parser.add_argument("--strict-unknown", action="store_true", help="Return unknown when classifier confidence is below --cls-conf/--gender-conf. Default returns top-1 with confidence in debug.")
    parser.add_argument("--include-keypoints", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--no-class-filter", action="store_true", help="Disable classes=[0] during YOLO inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    pipeline: PersonDetectorClassifierPipeline = PersonDetectorClassifierPipeline(config)
    payload = pipeline.process_image(
        args.image,
        output_dir=args.output_dir,
        include_keypoints=args.include_keypoints,
        save_visualization=not args.no_visualization,
    )
    save_json(Path(args.output_dir) / "annotations.json", payload)
    print(f"[PDC] Done. Annotations: {len(payload.get('annotations', []))}")
    print(f"[PDC] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()