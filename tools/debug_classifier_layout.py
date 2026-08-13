from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np

from person_detector_classifier.config import ClassifierConfig
from person_detector_classifier.src.classifier import AttributeClassifier


def make_dummy_image() -> np.ndarray:
    # Create a neutral gray 256x256 BGR image
    return np.full((256, 256, 3), 127, dtype=np.uint8)


def try_backend(backend: str) -> None:
    print(f"\n--- Testing backend: {backend} ---")
    cfg = ClassifierConfig()
    cfg = ClassifierConfig(**{**cfg.__dict__, "backend": backend})
    try:
        clf = AttributeClassifier(cfg)
    except Exception as exc:
        print("Failed to construct classifier:")
        traceback.print_exc()
        return

    print("backend:", clf.backend)
    print("layout:", clf.layout)
    print("output_dim:", getattr(clf, "output_dim", None))
    print("output_names:", getattr(clf, "output_names", None))

    img = make_dummy_image()
    try:
        res = clf.predict(img, include_probabilities=True)
        print("Classification result:")
        print(" age_group:", res.age_group)
        print(" age_confidence:", res.confidence)
        print(" gender:", res.gender)
        print(" gender_confidence:", res.gender_confidence)
        print(" age_probs:", res.probabilities)
        print(" gender_probs:", res.gender_probabilities)
        print(" layout reported by result:", res.layout)
    except Exception:
        print("Predict failed:")
        traceback.print_exc()


def main() -> None:
    # Try both backends
    for b in ("torch", "onnx"):
        try_backend(b)

    # Also try any explicit models shipped at repository root models/ (common test files)
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "models" / "mobilenetv4_utkface_age_gender_best.pth",
        root / "models" / "mobilenetv4_utkface_age_gender.onnx",
    ]
    if any(p.exists() for p in candidates):
        print("\n--- Testing explicit workspace models (if present) ---")
        pth = str(candidates[0])
        onnx = str(candidates[1])
        print(f"Testing PTH: {pth}")
        cfg = ClassifierConfig()
        cfg = ClassifierConfig(**{**cfg.__dict__, "backend": "torch", "pth_path": Path(pth), "onnx_path": Path(onnx)})
        try:
            clf = AttributeClassifier(cfg)
            print("torch layout:", clf.layout, "output_dim:", clf.output_dim)
            res = clf.predict(make_dummy_image(), include_probabilities=True)
            print(" result gender:", res.gender, res.gender_confidence, "age_conf:", res.confidence)
        except Exception:
            traceback.print_exc()
        print(f"Testing ONNX: {onnx}")
        cfg = ClassifierConfig()
        cfg = ClassifierConfig(**{**cfg.__dict__, "backend": "onnx", "pth_path": Path(pth), "onnx_path": Path(onnx)})
        try:
            clf = AttributeClassifier(cfg)
            print("onnx layout:", clf.layout, "output_dim:", clf.output_dim, "output_names:", getattr(clf, 'output_names', None))
            res = clf.predict(make_dummy_image(), include_probabilities=True)
            print(" result gender:", res.gender, res.gender_confidence, "age_conf:", res.confidence)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
