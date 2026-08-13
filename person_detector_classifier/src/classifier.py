from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from person_detector_classifier.config import ClassifierConfig
from person_detector_classifier.src.preprocessing import make_tta_crop_variants, preprocess_bgr_crop_for_mobilenet


AGE_GROUPS: Dict[int, str] = {
    0: "child",
    1: "teen",
    2: "young_adult",
    3: "adult",
    4: "middle_age",
    5: "senior",
}

AGE_GROUPS_VERBOSE: Dict[int, str] = {
    0: "0-10 Years (Child)",
    1: "11-20 Years (Teen)",
    2: "21-30 Years (Youth)",
    3: "31-40 Years (Adult)",
    4: "41-60 Years (Middle Age)",
    5: "60+ Years (Senior)",
}

GENDER_LABELS: Dict[int, str] = {
    0: "male",
    1: "female",
}


@dataclass
class ClassificationResult:
    age_group: str = "unknown"
    confidence: float = 0.0
    class_id: Optional[int] = None
    probabilities: list[float] | None = None
    gender: str = "unknown"
    gender_confidence: float = 0.0
    gender_class_id: Optional[int] = None
    gender_probabilities: list[float] | None = None
    backend: str = "torch"
    layout: str = "unknown"


class AttributeClassifier:
    """MobileNetV4 attribute classifier with torch and optional ONNX backends.

    v2 used the bundled checkpoint as an age-only classifier. v2.1 keeps that behavior but also
    auto-detects future age+gender heads:
      - 6 logits: age only
      - 8 logits: first 6 age logits, next 2 gender logits
      - 2 logits: gender only
      - ONNX two outputs: output0 age logits, output1 gender logits
    """

    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig()
        self.backend = self.config.backend.lower().strip()
        if self.backend not in {"torch", "onnx"}:
            raise ValueError("Classifier backend must be either 'torch' or 'onnx'")
        self.device = self._resolve_device(self.config.device)
        self.model = None
        self.session = None
        self.input_name = None
        self.output_names: list[str] = []
        self.output_dim: int | None = None
        self.layout = self.config.layout
        if self.backend == "torch":
            self._load_torch_model()
        else:
            self._load_onnx_model()

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> dict:
        if isinstance(checkpoint, dict):
            return checkpoint.get("model_state_dict", checkpoint)
        return checkpoint  # type: ignore[return-value]

    @staticmethod
    def _infer_output_dim_from_state_dict(state_dict: dict, fallback: int) -> int:
        for key in (
            "classifier.weight",
            "head.fc.weight",
            "head.weight",
            "fc.weight",
        ):
            value = state_dict.get(key)
            if hasattr(value, "shape") and len(value.shape) >= 2:
                return int(value.shape[0])
        # Last-resort scan for a likely final Linear layer.
        candidates = []
        for key, value in state_dict.items():
            if key.endswith(".weight") and hasattr(value, "shape") and len(value.shape) == 2:
                candidates.append((key, int(value.shape[0]), int(value.shape[1])))
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            return candidates[0][1]
        return int(fallback)

    def _load_torch_model(self) -> None:
        try:
            import timm  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency-path guard
            raise RuntimeError(
                "timm is required for the PyTorch MobileNetV4 backend. Install it with: pip install timm"
            ) from exc

        pth_path = Path(self.config.pth_path)
        if not pth_path.exists():
            # Try top-level project models/ as a convenience fallback
            try:
                from person_detector_classifier.config import PROJECT_ROOT
                alt = PROJECT_ROOT / "models" / pth_path.name
                if alt.exists():
                    pth_path = alt
                else:
                    raise FileNotFoundError(f"MobileNetV4 checkpoint not found: {pth_path}")
            except Exception:
                raise FileNotFoundError(f"MobileNetV4 checkpoint not found: {pth_path}")

        checkpoint = torch.load(str(pth_path), map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        # Prefer explicit metadata when available
        meta_layout = None
        try:
            if isinstance(checkpoint, dict):
                meta_layout = checkpoint.get("output_layout") or checkpoint.get("meta", {}).get("output_layout")
                # some exporters may store metadata under model_metadata or metadata
                if not meta_layout:
                    meta_layout = checkpoint.get("model_metadata", {}).get("output_layout") or checkpoint.get("metadata", {}).get("output_layout")
        except Exception:
            meta_layout = None

        output_dim = self._infer_output_dim_from_state_dict(state_dict, self.config.num_classes)
        self.output_dim = output_dim

        class _Wrapped(torch.nn.Module):
            def __init__(self, backbone_name, out_dim, input_size):
                super().__init__()
                self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
                self.backbone.eval()
                with torch.no_grad():
                    feat = self.backbone(torch.zeros(1, 3, input_size, input_size))
                if feat.ndim != 2:
                    feat = torch.flatten(feat, 1)
                feature_dim = int(feat.shape[1])
                self.dropout = torch.nn.Dropout(0.2)
                self.head = torch.nn.Linear(feature_dim, out_dim)

            def forward(self, x):
                x = self.backbone(x)
                if x.ndim != 2:
                    x = torch.flatten(x, 1)
                x = self.dropout(x)
                return self.head(x)

        model = _Wrapped(self.config.timm_model_name, output_dim, self.config.input_size[0])
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self.model = model
        # Resolve layout from metadata if present and allowed, otherwise infer from output dim
        if meta_layout and (self.config.layout or "auto").lower().strip() == "auto":
            try:
                self.layout = str(meta_layout)
            except Exception:
                self.layout = self._resolve_layout(output_dim=output_dim, num_outputs=1)
        else:
            self.layout = self._resolve_layout(output_dim=output_dim, num_outputs=1)

    def _load_onnx_model(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency-path guard
            raise RuntimeError(
                "onnxruntime is required for the ONNX backend. Install it with: pip install onnxruntime-gpu "
                "or pip install onnxruntime"
            ) from exc

        onnx_path = Path(self.config.onnx_path)
        if not onnx_path.exists():
            # Fallback to top-level project models/ directory
            try:
                from person_detector_classifier.config import PROJECT_ROOT
                alt = PROJECT_ROOT / "models" / onnx_path.name
                if alt.exists():
                    onnx_path = alt
                else:
                    raise FileNotFoundError(f"MobileNetV4 ONNX model not found: {onnx_path}")
            except Exception:
                raise FileNotFoundError(f"MobileNetV4 ONNX model not found: {onnx_path}")

        providers = ["CPUExecutionProvider"]
        if self.config.device and "cuda" in self.config.device.lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        first_shape = self.session.get_outputs()[0].shape
        if first_shape and len(first_shape) >= 2 and isinstance(first_shape[-1], int):
            self.output_dim = int(first_shape[-1])
        # Check ONNX model metadata for explicit layout hint
        try:
            model_meta = self.session.get_modelmeta()
            if hasattr(model_meta, "custom_metadata_map") and isinstance(model_meta.custom_metadata_map, dict):
                meta_layout = model_meta.custom_metadata_map.get("output_layout")
            else:
                meta_layout = None
        except Exception:
            meta_layout = None

        if meta_layout and (self.config.layout or "auto").lower().strip() == "auto":
            self.layout = str(meta_layout)
        else:
            self.layout = self._resolve_layout(output_dim=self.output_dim, num_outputs=len(self.output_names))

    def _resolve_layout(self, output_dim: int | None, num_outputs: int) -> str:
        requested = (self.config.layout or "auto").lower().strip()
        if requested != "auto":
            return requested
        if num_outputs >= 2:
            return "age6_gender2_multi_output"
        if output_dim == 8:
            return "age6_gender2"
        if output_dim == 6:
            return "age6"
        if output_dim == 2:
            return "gender2"
        return "unknown"

    def predict(self, crop_bgr: np.ndarray, include_probabilities: bool = False) -> ClassificationResult:
        if not self.config.use_tta:
            return self._predict_single(crop_bgr, include_probabilities)

        variants = make_tta_crop_variants(
            crop_bgr,
            center_crop_ratio=self.config.tta_center_crop_ratio,
            expand_canvas_ratio=self.config.tta_expand_canvas_ratio,
            use_hflip=self.config.tta_use_hflip,
        )
        if not variants:
            return ClassificationResult(backend=self.backend, layout=self.layout)

        single_results = [self._predict_single(v, include_probabilities=True) for v in variants]
        return self._average_results(single_results, include_probabilities=include_probabilities)

    def _predict_single(self, crop_bgr: np.ndarray, include_probabilities: bool = False) -> ClassificationResult:
        tensor = preprocess_bgr_crop_for_mobilenet(
            crop_bgr,
            input_size=self.config.input_size,
            mean=self.config.normalize_mean,
            std=self.config.normalize_std,
        )
        if self.backend == "torch":
            return self._predict_torch(tensor, include_probabilities)
        return self._predict_onnx(tensor, include_probabilities)

    def _average_results(self, results: list[ClassificationResult], include_probabilities: bool = False) -> ClassificationResult:
        if not results:
            return ClassificationResult(backend=self.backend, layout=self.layout)

        age_blocks = [np.asarray(r.probabilities, dtype=np.float64) for r in results if r.probabilities is not None and len(r.probabilities) > 0]
        gender_blocks = [np.asarray(r.gender_probabilities, dtype=np.float64) for r in results if r.gender_probabilities is not None and len(r.gender_probabilities) >= 2]

        age_group = "unknown"
        age_conf = 0.0
        age_id: Optional[int] = None
        age_probs_out = None
        gender = "unknown"
        gender_conf = 0.0
        gender_id: Optional[int] = None
        gender_probs_out = None

        if age_blocks:
            avg_age = np.mean(np.stack(age_blocks, axis=0), axis=0)
            age_group, age_conf, age_id, age_probs_out = self._make_age_result(avg_age, include_probabilities, already_probs=True)
            if self.config.min_age_margin > 0 and avg_age.size >= 2:
                top2 = np.sort(avg_age)[-2:]
                if float(top2[-1] - top2[-2]) < self.config.min_age_margin:
                    age_group = "unknown"

        if gender_blocks:
            avg_gender = np.mean(np.stack(gender_blocks, axis=0), axis=0)
            gender, gender_conf, gender_id, gender_probs_out = self._make_gender_result(avg_gender, include_probabilities, already_probs=True)
            if self.config.min_gender_margin > 0 and avg_gender.size >= 2:
                top2 = np.sort(avg_gender)[-2:]
                if float(top2[-1] - top2[-2]) < self.config.min_gender_margin:
                    gender = "unknown"

        return ClassificationResult(
            age_group=age_group,
            confidence=age_conf,
            class_id=age_id,
            probabilities=age_probs_out,
            gender=gender,
            gender_confidence=gender_conf,
            gender_class_id=gender_id,
            gender_probabilities=gender_probs_out,
            backend=self.backend,
            layout=f"{self.layout}+tta{len(results)}",
        )

    def _make_age_result(
        self,
        logits_or_probs: np.ndarray,
        include_probabilities: bool,
        already_probs: bool = False,
    ) -> tuple[str, float, Optional[int], list[float] | None]:
        arr = np.asarray(logits_or_probs, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return "unknown", 0.0, None, None
        if not already_probs:
            probs = F.softmax(torch.from_numpy(arr).float().reshape(1, -1), dim=1)[0].numpy().astype(np.float64)
        else:
            probs = arr / max(float(arr.sum()), 1e-12)
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        label = AGE_GROUPS.get(class_id, "unknown")
        if self.config.unknown_on_low_confidence and confidence < self.config.min_confidence:
            label = "unknown"
        return label, confidence, class_id, probs.round(6).tolist() if include_probabilities else None

    def _make_gender_result(
        self,
        logits_or_probs: np.ndarray,
        include_probabilities: bool,
        already_probs: bool = False,
    ) -> tuple[str, float, Optional[int], list[float] | None]:
        arr = np.asarray(logits_or_probs, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            return "unknown", 0.0, None, None
        if not already_probs:
            probs = F.softmax(torch.from_numpy(arr[:2]).float().reshape(1, -1), dim=1)[0].numpy().astype(np.float64)
        else:
            probs = arr[:2] / max(float(arr[:2].sum()), 1e-12)
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        label = GENDER_LABELS.get(class_id, "unknown")
        if self.config.unknown_on_low_confidence and confidence < self.config.min_gender_confidence:
            label = "unknown"
        return label, confidence, class_id, probs.round(6).tolist() if include_probabilities else None

    def _make_result_from_outputs(self, outputs: list[np.ndarray], include_probabilities: bool) -> ClassificationResult:
        layout = self.layout
        age_group = "unknown"
        age_conf = 0.0
        age_id: Optional[int] = None
        age_probs = None
        gender = "unknown"
        gender_conf = 0.0
        gender_id: Optional[int] = None
        gender_probs = None

        if layout == "age6_gender2_multi_output" and len(outputs) >= 2:
            age_group, age_conf, age_id, age_probs = self._make_age_result(outputs[0], include_probabilities)
            gender, gender_conf, gender_id, gender_probs = self._make_gender_result(outputs[1], include_probabilities)
        else:
            flat = np.asarray(outputs[0], dtype=np.float64).reshape(-1) if outputs else np.array([], dtype=np.float64)
            if layout == "age6_gender2" or flat.size == 8:
                age_group, age_conf, age_id, age_probs = self._make_age_result(flat[:6], include_probabilities)
                gender, gender_conf, gender_id, gender_probs = self._make_gender_result(flat[6:8], include_probabilities)
            elif layout == "gender2" or flat.size == 2:
                gender, gender_conf, gender_id, gender_probs = self._make_gender_result(flat[:2], include_probabilities)
            elif layout == "age6" or flat.size == 6:
                age_group, age_conf, age_id, age_probs = self._make_age_result(flat[:6], include_probabilities)
            else:
                # Unknown layout: expose nothing rather than mislabeling logits.
                pass

        return ClassificationResult(
            age_group=age_group,
            confidence=age_conf,
            class_id=age_id,
            probabilities=age_probs,
            gender=gender,
            gender_confidence=gender_conf,
            gender_class_id=gender_id,
            gender_probabilities=gender_probs,
            backend=self.backend,
            layout=layout,
        )

    def _predict_torch(self, tensor: torch.Tensor, include_probabilities: bool) -> ClassificationResult:
        if self.model is None:
            raise RuntimeError("Torch model has not been loaded")
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            if isinstance(logits, dict):
                outputs = []
                for key in ("age", "age_logits", "output", "gender", "gender_logits"):
                    if key in logits:
                        val = logits[key]
                        outputs.append(val.detach().cpu().numpy())
                if not outputs:
                    outputs = [next(iter(logits.values())).detach().cpu().numpy()]
            elif isinstance(logits, (tuple, list)):
                outputs = [x.detach().cpu().numpy() for x in logits]
            else:
                outputs = [logits.detach().cpu().numpy()]
        return self._make_result_from_outputs(outputs, include_probabilities)

    def _predict_onnx(self, tensor: torch.Tensor, include_probabilities: bool) -> ClassificationResult:
        if self.session is None or self.input_name is None:
            raise RuntimeError("ONNX session has not been loaded")
        input_np = tensor.numpy().astype(np.float32)
        outputs = self.session.run(self.output_names or None, {self.input_name: input_np})
        outputs_np = [np.asarray(x) for x in outputs]
        return self._make_result_from_outputs(outputs_np, include_probabilities)


# Backward-compatible name used by v1/v2 imports.
AgeClassifier = AttributeClassifier
