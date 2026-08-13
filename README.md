# PDC v3 — Person Detector & Classifier

Detect people in an image, then predict **age group** + **gender** for each selected person.

Works on **Windows / macOS / Linux**. CPU is enough.

---

## 1) Requirements

- Python **3.10+** (3.11 or 3.12 recommended)
- Internet (first install; YOLO may download weights if the local file is missing)
- These model files must exist:

```text
person_detector_classifier/models/
├── yolo11n-pose.pt
├── mobilenetv4_utkface_age_gender_best.pth   # preferred
├── mobilenetv4_utkface_age_gender.pth
├── mobilenetv4_utkface_age_gender.onnx
└── mobilenetv4_utkface_age_gender.onnx.data
```

If any of these are missing after `git clone`, ask the project owner for the model files and put them in that folder.

---

## 2) Setup (do this once)

### macOS / Linux

```bash
cd /path/to/SyedHabib

# create venv (use whatever python3 you have)
python3 -m venv .venv

# activate
source .venv/bin/activate

# install deps
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd C:\path\to\SyedHabib

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Always activate `.venv` before running any command.
> Prompt should look like: `(.venv) ...`

---

## 3) Verify install (2 minutes)

```bash
python quick_smoke_test.py
```

Expected:

```text
[PDC] Smoke test passed.
```

If this fails, stop and fix install first. Do not run images yet.

---

## 4) Run on one image (recommended)

Defaults already point to the correct YOLO + age/gender checkpoints, so this is enough:

```bash
python run_image.py \
  --image test_assets/images_for_rectifying/woman_portrait_00054.jpg \
  --output-dir outputs/demo \
  --detector-backend auto \
  --backend torch \
  --debug
```

### Your own image

```bash
python run_image.py \
  --image path/to/your_image.jpg \
  --output-dir outputs/demo \
  --detector-backend auto \
  --backend torch \
  --debug
```

### Outputs

```text
outputs/demo/
├── annotations.json
├── <image_name>.json
└── annotated_images/
    └── <image_name>_annotated.jpg
```

---

## 5) Interactive examiner (GUI)

### Desktop GUI

```bash
python pdc_examiner.py --detector-backend auto --backend torch --debug
```

### With a specific image

```bash
python pdc_examiner.py \
  --image path/to/your_image.jpg \
  --detector-backend auto \
  --backend torch \
  --debug
```

Selected result is saved under `outputs/examiner/`.

---

## 6) Run a whole folder

```bash
python run_folder.py \
  --input-dir test_assets/images_for_rectifying \
  --output-dir outputs/folder \
  --detector-backend auto \
  --backend torch \
  --debug
```

---

## 7) Google Colab

```bash
!pip install -q -r requirements_colab.txt
!python quick_smoke_test.py
!python pdc_examiner.py --colab --detector-backend auto --backend torch --debug
```

---

## 8) Common errors & fixes

| Error | Fix |
|---|---|
| `python: command not found` | Use `python3` instead of `python` |
| `pip: command not found` | Use `python -m pip install -r requirements.txt` |
| `python3.11: command not found` | Use `python3 -m venv .venv` (any 3.10+) |
| YOLO / ultralytics weight missing | Confirm `person_detector_classifier/models/yolo11n-pose.pt` exists, or use `--detector-backend auto` |
| `FileNotFoundError` for `.pth` / `.onnx` | Put model files into `person_detector_classifier/models/` |
| Gender always `unknown` | Use age+gender checkpoint (`mobilenetv4_utkface_age_gender_best.pth`), not age-only `mobilenetv4_utkface.pth` |
| `annotations: []` | Image too small / people too far / crowded. Try a clearer, larger photo |
| GUI does not open | Use CLI `run_image.py` instead |

---

## 9) What each model does

| File | Role |
|---|---|
| `yolo11n-pose.pt` | Detect persons + keypoints |
| `mobilenetv4_utkface_age_gender_best.pth` | Age + gender classifier (default / recommended) |
| `mobilenetv4_utkface.pth` | Age-only (gender will be unknown) |

---

## 10) Optional advanced flags

```bash
--no-tta                         # disable TTA averaging
--strict-unknown                 # hide low-confidence outputs
--cls-conf 0.30                  # age confidence threshold when strict unknown is enabled
--gender-conf 0.30               # gender confidence threshold when strict unknown is enabled
--classifier-crop-mode auto_face # best default for face-trained classifiers
--save-classifier-crops          # with --debug, writes actual MobileNet crops
--pth-model path/to/model.pth    # override classifier checkpoint
--yolo-model path/to/yolo.pt     # override detector weights
```

---

## 11) Minimal success checklist

1. `source .venv/bin/activate` (or Windows activate)
2. `pip install -r requirements.txt`
3. Confirm model files exist in `person_detector_classifier/models/`
4. `python quick_smoke_test.py` → passed
5. Run `run_image.py` with the command in section 4
6. Open annotated image + JSON in `outputs/demo/`

If step 4 passes and step 5 fails, the issue is image path / model path — not install.

---

## Project layout

```text
├── pdc_examiner.py
├── run_image.py
├── run_folder.py
├── quick_smoke_test.py
├── requirements.txt
├── person_detector_classifier/
│   ├── config.py
│   ├── models/          # YOLO + MobileNet checkpoints (required)
│   └── src/
├── test_assets/
└── tools/
```
