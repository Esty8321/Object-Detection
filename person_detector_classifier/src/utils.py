from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VALID_IMAGE_EXTENSIONS


def list_images(path: str | Path, recursive: bool = False) -> List[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if root.is_file():
        if not is_image_file(root):
            raise ValueError(f"Input file is not a supported image: {root}")
        return [root]
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in root.glob(pattern) if p.is_file() and is_image_file(p)])


def read_image_bgr(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # Convert BGRA to BGR on a white background so transparent PNGs do not become black blobs.
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        image = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
    elif image.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape {image.shape} for {path}")
    return image


def write_image(path: str | Path, image_bgr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    ext = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ValueError(f"Could not encode image for output path: {path}")
    encoded.tofile(str(path))


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def stable_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_float_list(values: Iterable[float], digits: int = 2) -> list[float]:
    return [round(float(v), digits) for v in values]
