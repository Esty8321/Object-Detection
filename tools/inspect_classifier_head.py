from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MobileNetV4 checkpoint output head dimensions.")
    parser.add_argument("--pth", default="person_detector_classifier/models/mobilenetv4_utkface.pth")
    args = parser.parse_args()
    path = Path(args.pth)
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(str(path), map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    print(f"Checkpoint: {path.resolve()}")
    for key, value in sd.items():
        if key in {"classifier.weight", "classifier.bias", "head.fc.weight", "head.fc.bias", "head.weight", "head.bias", "fc.weight", "fc.bias"}:
            print(f"{key}: {tuple(value.shape)}")
    weight = None
    for key in ("classifier.weight", "head.fc.weight", "head.weight", "fc.weight"):
        if key in sd:
            weight = sd[key]
            break
    if weight is None:
        print("Could not find a standard final classification weight.")
        return
    out_dim = int(weight.shape[0])
    if out_dim == 6:
        print("Layout detected: age6 only. Gender cannot be predicted from this checkpoint.")
    elif out_dim == 8:
        print("Layout detected: age6_gender2. logits[0:6] age, logits[6:8] gender.")
    elif out_dim == 2:
        print("Layout detected: gender2 only. Age cannot be predicted from this checkpoint.")
    else:
        print(f"Unknown output layout: {out_dim} logits. Set ClassifierConfig(layout=...) manually.")


if __name__ == "__main__":
    main()
