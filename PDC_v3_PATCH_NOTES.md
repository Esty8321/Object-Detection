# PDC v3 Patch Notes

## Purpose

PDC v3 fixes the failure pattern observed in multi-person/real-world images without retraining the MobileNetV4 classifier.

## Main fixes

1. MobileNet no longer receives the full image or an ambiguous multi-person crop.
2. The classifier crop is now chosen in this priority order:
   - OpenCV face detector inside selected person bbox
   - pose keypoint face crop
   - shoulder-derived head crop
   - upper-body crop
   - full person crop only when the selected person is not crowded
3. If another person overlaps the selected bbox, v3 forces face/head-only classification.
4. TTA probability averaging is enabled by default.
5. Unknown gating is softer by default and configurable.
6. Rotation no longer reports 90 degrees for upright portrait-like people unless trustworthy keypoints support it.
7. Local and Colab examiners show the actual classifier crop used by MobileNet.
8. JSON contains `attributes.classifier_crop` for traceability.

## Recommended demo command

```bash
python pdc_examiner.py --detector-backend yolo --backend torch --debug
```

If YOLO weights cannot be downloaded:

```bash
python pdc_examiner.py --detector-backend auto --backend torch --debug
```

## Debug command

```bash
python run_image.py --image path/to/image.jpg --detector-backend auto --backend torch --debug --save-classifier-crops
```

## Important limitation

If the MobileNet checkpoint has no gender output head, PDC cannot manufacture gender from thin air. v3 supports gender-capable checkpoints automatically, but the checkpoint itself must expose gender logits or an ONNX gender output.
