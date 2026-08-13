from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from person_detector_classifier.config import AttributeConfig, ClassifierConfig
from person_detector_classifier.src.classifier import AttributeClassifier
from person_detector_classifier.src.attributes import extract_attributes
from person_detector_classifier.src.detector import PersonDetection
from person_detector_classifier.src.preprocessing import build_classifier_crop, crop_bgr, expand_bbox_xyxy, preprocess_bgr_crop_for_mobilenet
from person_detector_classifier.src.schema import make_annotation, make_dataset_payload, make_image_record
from person_detector_classifier.src.utils import ensure_dir, save_json, write_image
from person_detector_classifier.src.visualizer import draw_annotations


def main() -> None:
    out_dir = ensure_dir("outputs/smoke_test")
    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (230, 80), (400, 430), (180, 180, 180), -1)
    cv2.circle(image, (315, 55), 45, (180, 180, 180), -1)

    keypoints = np.zeros((17, 3), dtype=np.float32)
    pts = {
        0: (315, 45), 5: (260, 130), 6: (370, 130), 7: (245, 230), 8: (385, 230),
        9: (235, 315), 10: (395, 315), 11: (275, 315), 12: (355, 315),
        13: (270, 410), 14: (360, 410), 15: (265, 455), 16: (365, 455),
    }
    for idx, (x, y) in pts.items():
        keypoints[idx] = [x, y, 0.9]

    detection = PersonDetection(
        bbox_xyxy=(220.0, 20.0, 410.0, 470.0),
        confidence=0.92,
        class_id=0,
        keypoints=keypoints,
    )

    padded = expand_bbox_xyxy(detection.bbox_xyxy, image_width=640, image_height=480, padding_ratio=0.08)
    person_crop = crop_bgr(image, padded)
    crop_result = build_classifier_crop(
        image_bgr=image,
        bbox_xyxy=detection.bbox_xyxy,
        keypoints=detection.keypoints,
        all_bboxes_xyxy=[detection.bbox_xyxy],
    )
    classifier_crop = crop_result.crop_bgr if crop_result.crop_bgr is not None else person_crop
    tensor = preprocess_bgr_crop_for_mobilenet(classifier_crop)
    assert tuple(tensor.shape) == (1, 3, 256, 256), tuple(tensor.shape)

    attr = extract_attributes(
        detection,
        image_width=640,
        image_height=480,
        crop_bgr=person_crop,
        config=AttributeConfig(),
        include_debug=True,
    )
    # Try to run the classifier if available, otherwise keep unknown placeholders
    age_group = "unknown"
    age_conf = 0.0
    gender = "unknown"
    gender_conf = 0.0
    try:
        cfg = ClassifierConfig()
        clf = AttributeClassifier(cfg)
        res = clf.predict(classifier_crop, include_probabilities=True)
        age_group = res.age_group
        age_conf = round(float(res.confidence), 2)
        gender = res.gender
        gender_conf = round(float(res.gender_confidence), 2)
    except Exception:
        # classifier not available or failed; keep unknowns
        pass

    attributes = {
        "gender": gender,
        "visibility": attr.visibility,
        "age_group": age_group,
        "occlusion": attr.occlusion,
        "quality": attr.quality,
        "occluded": attr.occluded,
        "rotation": round(attr.rotation, 2),
        "classifier_crop": {
            "mode": crop_result.source,
            "quality": crop_result.quality,
            "bbox": list(crop_result.bbox_xyxy) if crop_result.bbox_xyxy is not None else None,
            "crowded": crop_result.crowded,
            "reason": crop_result.reason,
        },
        "age_confidence": age_conf,
        "gender_confidence": gender_conf,
        "debug": attr.debug,
    }
    ann = make_annotation(1, "smoke_test.jpg", detection, attributes, include_keypoints=True)
    payload = make_dataset_payload(
        images=[make_image_record("smoke_test.jpg", "smoke_test.jpg", 640, 480)],
        annotations=[ann],
        version="v3_smoke_test",
    )
    save_json(out_dir / "annotations.json", payload)
    visual = draw_annotations(image, [detection], [ann])
    write_image(out_dir / "smoke_test_annotated.jpg", visual)

    print("[PDC] Smoke test passed.")
    print(f"[PDC] Wrote: {Path(out_dir / 'annotations.json').resolve()}")
    print(json.dumps(payload["annotations"][0]["attributes"], indent=2))


if __name__ == "__main__":
    main()
