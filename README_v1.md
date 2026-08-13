# PDC_v1 - Person Detector and Classifier

Production-oriented two-stage pipeline for multi-person image annotation:

1. **YOLO11n-pose** detects persons, bounding boxes, and 17 COCO keypoints.
2. **MobileNetV4 Conv Small** classifies each valid person crop into one of six age groups.
3. Heuristic modules derive visibility, occlusion, quality, and rotation.
4. Output is COCO-like JSON with your required `attributes` object.

> Important: the included MobileNetV4 checkpoint is a 6-class **age-group** classifier only. It does not predict gender. Therefore `gender` is intentionally returned as `"unknown"` unless you later plug in a trained gender model. This is not a bug. This is the pipeline refusing to lie for applause.

---

## Folder structure

```text
PDC_v1/
├── person_detector_classifier/
│   ├── config.py
│   ├── models/
│   │   ├── mobilenetv4_utkface.pth
│   │   ├── mobilenetv4_utkface.onnx
│   │   ├── mobilenetv4_utkface.onnx.data
│   │   ├── export_original.py
│   │   ├── infer_original.py
│   │   └── train_export_original.py
│   └── src/
│       ├── attributes.py
│       ├── classifier.py
│       ├── detector.py
│       ├── pipeline.py
│       ├── preprocessing.py
│       ├── schema.py
│       ├── utils.py
│       └── visualizer.py
├── run_image.py
├── run_folder.py
├── quick_smoke_test.py
├── requirements.txt
├── requirements_colab.txt
└── colab_demo.ipynb
```

---

## Colab quick start

Upload `PDC_v1.zip` to Colab, then run:

```bash
!unzip -q PDC_v1.zip
%cd PDC_v1
!pip install -q -r requirements_colab.txt
```

Upload one or more test images into `/content/PDC_v1/sample_inputs/`, then run:

```bash
!python run_folder.py \
  --input-dir sample_inputs \
  --output-dir outputs \
  --backend torch \
  --device cuda:0 \
  --debug
```

Outputs:

```text
outputs/
├── annotations.json
├── image_name.json
└── annotated_images/
    └── image_name_annotated.jpg
```

---

## Local PC quick start

Create and activate a virtual environment.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run:

```bash
python quick_smoke_test.py
python run_image.py --image path/to/image.jpg --output-dir outputs --backend torch
```

For GPU:

```bash
python run_image.py --image path/to/image.jpg --output-dir outputs --backend torch --device cuda:0
```

---

## Example output

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": "image_001.jpg",
      "area": 120746.3488,
      "bbox": [249.57, 5.82, 318.08, 379.61],
      "iscrowd": 0,
      "attributes": {
        "gender": "unknown",
        "visibility": "upper_body",
        "age_group": "adult",
        "occlusion": "none",
        "quality": "medium",
        "occluded": false,
        "rotation": 0.0
      }
    }
  ]
}
```

---

## Attribute logic

### `age_group`

Predicted by MobileNetV4:

| Class | Label |
|---:|---|
| 0 | `child` |
| 1 | `teen` |
| 2 | `young_adult` |
| 3 | `adult` |
| 4 | `middle_age` |
| 5 | `senior` |

If confidence is below `--cls-conf`, the result becomes `unknown`.

### `visibility`

Derived from YOLO keypoints:

- `full_body`
- `upper_body`
- `lower_body`
- `partial`
- `unknown`

### `occlusion` and `occluded`

Derived from missing keypoint ratio and whether bbox touches the image boundary:

- `none`
- `partial`
- `heavy`
- `unknown`

### `quality`

Derived from crop size, blur score, detection confidence, and visible keypoints:

- `high`
- `medium`
- `low`

Low-quality crops are not classified for age. They return `age_group: "unknown"`.

### `rotation`

Estimated from shoulder line. If shoulders are not visible, hips are used. If neither is available, rotation is `0.0`.

---

## Useful commands

### Single image

```bash
python run_image.py \
  --image sample_inputs/test.jpg \
  --output-dir outputs \
  --backend torch \
  --debug
```

### Folder

```bash
python run_folder.py \
  --input-dir sample_inputs \
  --output-dir outputs \
  --backend torch \
  --recursive \
  --debug
```

### ONNX backend

```bash
python run_image.py \
  --image sample_inputs/test.jpg \
  --output-dir outputs_onnx \
  --backend onnx
```

The ONNX backend needs `mobilenetv4_utkface.onnx` and `mobilenetv4_utkface.onnx.data` to stay in the same folder.

---

## Known limitation

The current classifier does **age-group only**. To support real gender prediction later, add a `GenderClassifier` module and update `_compose_attributes()` inside `src/pipeline.py`.
