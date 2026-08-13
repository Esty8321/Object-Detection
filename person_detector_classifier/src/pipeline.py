from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from person_detector_classifier.config import PipelineConfig
from person_detector_classifier.src.attributes import extract_attributes
from person_detector_classifier.src.classifier import AttributeClassifier, ClassificationResult
from person_detector_classifier.src.detector import PersonDetection, build_detector
from person_detector_classifier.src.preprocessing import build_classifier_crop, crop_bgr, expand_bbox_xyxy
from person_detector_classifier.src.schema import make_annotation, make_dataset_payload, make_image_record, safe_image_id
from person_detector_classifier.src.utils import ensure_dir, list_images, read_image_bgr, save_json, write_image
from person_detector_classifier.src.visualizer import draw_annotations, draw_detections_preview


class PersonDetectorClassifierPipeline:
    """Production orchestration for PDC v3.

    v3 classifies exactly the selected/detected person. It prefers face/head crops, refuses to feed
    crowded full-body crops into MobileNet, and uses TTA in the classifier to reduce spurious unknowns
    without retraining.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.detector = build_detector(self.config.detector)
        self.classifier = AttributeClassifier(self.config.classifier)

    def detect_image(self, image_path: str | Path) -> tuple[np.ndarray, str, list[PersonDetection]]:
        image_path = Path(image_path)
        image_bgr = read_image_bgr(image_path)
        image_id = safe_image_id(image_path)
        detections = self.detect_array(image_bgr)
        return image_bgr, image_id, detections

    def detect_array(self, image_bgr: np.ndarray) -> list[PersonDetection]:
        detections = self.detector.predict(image_bgr)
        filtered: list[PersonDetection] = []
        for det in detections:
            if det.width < self.config.attributes.min_crop_width or det.height < self.config.attributes.min_crop_height:
                continue
            filtered.append(det)
        filtered.sort(key=lambda d: (d.bbox_xyxy[0], d.bbox_xyxy[1], -d.confidence))
        return filtered

    def analyze_detection(
        self,
        image_bgr: np.ndarray,
        image_id: str,
        detection: PersonDetection,
        annotation_id: int = 1,
        include_keypoints: bool = False,
        all_detections: list[PersonDetection] | None = None,
    ) -> dict[str, Any]:
        image_h, image_w = image_bgr.shape[:2]
        padded_bbox = expand_bbox_xyxy(
            detection.bbox_xyxy,
            image_width=image_w,
            image_height=image_h,
            padding_ratio=self.config.attributes.crop_padding_ratio,
        )
        person_crop = crop_bgr(image_bgr, padded_bbox)
        attr = extract_attributes(
            detection,
            image_width=image_w,
            image_height=image_h,
            crop_bgr=person_crop,
            config=self.config.attributes,
            include_debug=self.config.include_debug,
        )

        all_bboxes = [d.bbox_xyxy for d in all_detections] if all_detections else None
        crop_result = build_classifier_crop(
            image_bgr=image_bgr,
            bbox_xyxy=detection.bbox_xyxy,
            keypoints=detection.keypoints,
            all_bboxes_xyxy=all_bboxes,
            mode=self.config.attributes.classifier_crop_mode,
            keypoint_threshold=self.config.attributes.keypoint_conf_threshold,
            face_padding_ratio=self.config.attributes.face_crop_padding_ratio,
            head_padding_ratio=self.config.attributes.head_crop_padding_ratio,
            upper_body_fraction=self.config.attributes.upper_body_fraction,
            min_classify_crop_width=self.config.attributes.min_classify_crop_width,
            min_classify_crop_height=self.config.attributes.min_classify_crop_height,
            blur_min_threshold=self.config.attributes.classifier_blur_min_threshold,
            enable_face_detector_crop=self.config.attributes.enable_face_detector_crop,
            face_detector_min_neighbors=self.config.attributes.face_detector_min_neighbors,
            face_detector_scale_factor=self.config.attributes.face_detector_scale_factor,
            face_detector_min_face_ratio=self.config.attributes.face_detector_min_face_ratio,
            max_faces_inside_person=self.config.attributes.max_faces_inside_person,
            multi_person_overlap_threshold=self.config.attributes.multi_person_overlap_threshold,
            multi_person_force_face_crop=self.config.attributes.multi_person_force_face_crop,
        )

        classification = self._classify_if_valid(crop_result.crop_bgr, detection, attr.quality)
        attributes = self._compose_attributes(classification, attr, crop_result)
        return make_annotation(
            annotation_id=annotation_id,
            image_id=image_id,
            detection=detection,
            attributes=attributes,
            digits=self.config.round_digits,
            include_keypoints=include_keypoints,
        )

    def preview_image(self, image_bgr: np.ndarray, detections: list[PersonDetection]) -> np.ndarray:
        return draw_detections_preview(
            image_bgr,
            detections,
            keypoint_threshold=self.config.attributes.keypoint_conf_threshold,
        )

    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        include_keypoints: bool = False,
        save_visualization: bool | None = None,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        image_bgr, image_id, detections = self.detect_image(image_path)
        image_h, image_w = image_bgr.shape[:2]

        annotations: list[dict[str, Any]] = []
        used_detections: list[PersonDetection] = []

        for det in detections:
            ann = self.analyze_detection(
                image_bgr=image_bgr,
                image_id=image_id,
                detection=det,
                annotation_id=len(annotations) + 1,
                include_keypoints=include_keypoints,
                all_detections=detections,
            )
            annotations.append(ann)
            used_detections.append(det)

        image_record = make_image_record(
            image_id=image_id,
            file_name=image_path.name,
            width=image_w,
            height=image_h,
        )
        payload = make_dataset_payload(images=[image_record], annotations=annotations, version="v3")

        should_save_vis = self.config.save_visualization if save_visualization is None else save_visualization
        if output_dir is not None:
            out_dir = ensure_dir(output_dir)
            save_json(out_dir / f"{image_path.stem}.json", payload)
            if should_save_vis:
                visual = draw_annotations(
                    image_bgr,
                    used_detections,
                    annotations,
                    keypoint_threshold=self.config.attributes.keypoint_conf_threshold,
                )
                write_image(out_dir / "annotated_images" / f"{image_path.stem}_annotated.jpg", visual)
            if self.config.include_debug and self.config.attributes.save_classifier_crop_debug:
                self._save_classifier_debug_crops(image_bgr, annotations, out_dir / "classifier_crops", image_path.stem)
        return payload

    def process_folder(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        recursive: bool = False,
        include_keypoints: bool = False,
    ) -> dict[str, Any]:
        image_paths = list_images(input_dir, recursive=recursive)
        out_dir = ensure_dir(output_dir)
        all_images: list[dict[str, Any]] = []
        all_annotations: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        global_ann_id = 1

        for image_path in image_paths:
            try:
                per_image = self.process_image(
                    image_path,
                    output_dir=out_dir,
                    include_keypoints=include_keypoints,
                    save_visualization=self.config.save_visualization,
                )
                all_images.extend(per_image.get("images", []))
                for ann in per_image.get("annotations", []):
                    ann["id"] = global_ann_id
                    global_ann_id += 1
                    all_annotations.append(ann)
            except Exception as exc:
                errors.append({"image_id": image_path.name, "reason": str(exc)})

        payload = make_dataset_payload(all_images, all_annotations, errors=errors, version="v3")
        save_json(out_dir / "annotations.json", payload)
        return payload

    def process_selected_detections(
        self,
        image_path: str | Path,
        selected_indices: list[int],
        output_dir: str | Path | None = None,
        include_keypoints: bool = False,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        image_bgr, image_id, detections = self.detect_image(image_path)
        image_h, image_w = image_bgr.shape[:2]
        annotations: list[dict[str, Any]] = []
        kept_detections: list[PersonDetection] = []
        for idx in selected_indices:
            if idx < 0 or idx >= len(detections):
                continue
            ann = self.analyze_detection(
                image_bgr=image_bgr,
                image_id=image_id,
                detection=detections[idx],
                annotation_id=len(annotations) + 1,
                include_keypoints=include_keypoints,
                all_detections=detections,
            )
            annotations.append(ann)
            kept_detections.append(detections[idx])
        payload = make_dataset_payload(
            images=[make_image_record(image_id, image_path.name, image_w, image_h)],
            annotations=annotations,
            version="v3_selected",
        )
        if output_dir is not None:
            out_dir = ensure_dir(output_dir)
            save_json(out_dir / f"{image_path.stem}_selected.json", payload)
            visual = draw_annotations(image_bgr, kept_detections, annotations, self.config.attributes.keypoint_conf_threshold)
            write_image(out_dir / "annotated_images" / f"{image_path.stem}_selected_annotated.jpg", visual)
            if self.config.include_debug and self.config.attributes.save_classifier_crop_debug:
                self._save_classifier_debug_crops(image_bgr, annotations, out_dir / "classifier_crops", f"{image_path.stem}_selected")
        return payload

    def _classify_if_valid(
        self,
        classifier_crop: np.ndarray | None,
        detection: PersonDetection,
        quality: str,
    ) -> ClassificationResult:
        if classifier_crop is None or classifier_crop.size == 0:
            return ClassificationResult(age_group="unknown", confidence=0.0, class_id=None, backend=self.config.classifier.backend)
        crop_h, crop_w = classifier_crop.shape[:2]
        if crop_w < self.config.attributes.min_classify_crop_width or crop_h < self.config.attributes.min_classify_crop_height:
            return ClassificationResult(age_group="unknown", confidence=0.0, class_id=None, backend=self.config.classifier.backend)
        if quality == "low" and not self.config.attributes.low_quality_still_classify:
            return ClassificationResult(age_group="unknown", confidence=0.0, class_id=None, backend=self.config.classifier.backend)
        return self.classifier.predict(classifier_crop, include_probabilities=self.config.include_debug)

    def _compose_attributes(
        self,
        classification: ClassificationResult,
        attr: Any,
        crop_result: Any,
    ) -> dict[str, Any]:
        crop_info = {
            "mode": crop_result.source,
            "quality": crop_result.quality,
            "bbox": list(crop_result.bbox_xyxy) if crop_result.bbox_xyxy is not None else None,
            "crowded": bool(crop_result.crowded),
            "reason": crop_result.reason,
        }
        attributes: dict[str, Any] = {
            "gender": classification.gender,
            "visibility": attr.visibility,
            "age_group": classification.age_group,
            "occlusion": attr.occlusion,
            "quality": attr.quality,
            "occluded": bool(attr.occluded),
            "rotation": round(float(attr.rotation), self.config.round_digits),
            "classifier_crop": crop_info,
        }
        # Expose confidences in the primary output schema
        attributes["age_confidence"] = round(float(classification.confidence), self.config.round_digits)
        attributes["gender_confidence"] = round(float(classification.gender_confidence), self.config.round_digits)
        if self.config.include_debug:
            attributes["debug"] = {
                "age_class_id": classification.class_id,
                "age_confidence": round(float(classification.confidence), 6),
                "age_probabilities": classification.probabilities,
                "gender_class_id": classification.gender_class_id,
                "gender_confidence": round(float(classification.gender_confidence), 6),
                "gender_probabilities": classification.gender_probabilities,
                "classifier_backend": classification.backend,
                "classifier_layout": classification.layout,
                "classifier_crop_candidates": crop_result.candidates,
                **(attr.debug or {}),
            }
        return attributes

    def _save_classifier_debug_crops(
        self,
        image_bgr: np.ndarray,
        annotations: list[dict[str, Any]],
        output_dir: Path,
        image_stem: str,
    ) -> None:
        out_dir = ensure_dir(output_dir)
        for ann in annotations:
            attrs = ann.get("attributes", {})
            crop_info = attrs.get("classifier_crop", {})
            bbox = crop_info.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            crop = crop_bgr(image_bgr, tuple(int(v) for v in bbox))
            if crop is None or crop.size == 0:
                continue
            source = str(crop_info.get("mode", "crop")).replace("/", "_")
            write_image(out_dir / f"{image_stem}_person{ann.get('id', 'x')}_{source}.jpg", crop)
