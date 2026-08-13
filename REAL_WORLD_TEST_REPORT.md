# PDC v3 Real-World Rectification Test Report

## Test source

The included `test_assets/images_for_rectifying/` folder was used as the local substitute for online real-world images.

## Backend caveat

This environment did not have cached `yolo11n-pose.pt` and should not be treated as a true YOLO11-pose benchmark. The v3 rectification logic was validated with the offline OpenCV fallback detector plus the ONNX MobileNetV4 classifier path. In real Colab/Windows testing with internet or local YOLO weights, use the YOLO backend.

## Validation commands

Syntax and smoke test:

```bash
python -m compileall -q .
python quick_smoke_test.py
```

Subset real-world crop/inference validation:

```bash
python run_folder.py \
  --input-dir test_assets/images_for_rectifying \
  --output-dir outputs/realworld_v3 \
  --detector-backend opencv_hog \
  --backend onnx \
  --debug \
  --save-classifier-crops
```

## Observed v2.1 flaw

When the selected person crop contained a full body, more than one person, or a tiny face, MobileNet often produced low confidence and the output became `unknown`. The screenshot case also showed a bad `90.0°` rotation for an upright person.

## v3 rectifications

- MobileNet receives a face/head crop first instead of a full-body crop.
- In crowded selections, upper/full-body fallback is blocked, preventing multi-person contamination.
- TTA probability averaging stabilizes classifier output.
- The actual classifier crop is shown in the local and Colab examiners.
- Rotation estimation is conservative and now rejects vertical shoulder/hip artifacts.
- `attributes.classifier_crop` records crop source, bbox, quality, and crowded-scene status.

## Example expected debug fields

```json
"classifier_crop": {
  "mode": "face_detector",
  "quality": "high",
  "bbox": [219, 164, 731, 715],
  "crowded": false,
  "reason": "ok"
}
```

## Remaining limitation

Gender will only be produced if the provided checkpoint has a real gender output. The package supports common gender-capable MobileNet layouts, but it will not invent gender from an age-only checkpoint, because that would be less engineering and more horoscope software.
