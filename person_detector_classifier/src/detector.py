from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol

import cv2
import numpy as np

from person_detector_classifier.config import DetectorConfig


@dataclass
class PersonDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    keypoints: np.ndarray | None = None  # expected shape: (17, 3), x/y/conf
    source: str = "yolo"

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.bbox_xyxy
        return max(0.0, x2 - x1)

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.bbox_xyxy
        return max(0.0, y2 - y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.bbox_xyxy
        return x1 <= x <= x2 and y1 <= y <= y2


class DetectorProtocol(Protocol):
    def predict(self, image_bgr: np.ndarray) -> List[PersonDetection]:
        ...


class YOLOPoseDetector:
    """Thin wrapper around Ultralytics YOLO11 pose inference."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency-path guard
            raise RuntimeError(
                "Ultralytics is required for YOLO11 pose detection. Install it with: pip install ultralytics"
            ) from exc
        try:
            model_path = self.config.model_path
            # If user provided a bare filename and it wasn't found, try top-level models/ fallback
            try:
                p = Path(model_path)
                if not p.exists():
                    from person_detector_classifier.config import PROJECT_ROOT
                    alt = PROJECT_ROOT / "models" / p.name
                    if alt.exists():
                        model_path = str(alt)
            except Exception:
                pass
            self.model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(
                "Could not load YOLO pose model. Put yolo11n-pose.pt beside your script or allow the "
                "first Ultralytics download on a machine with internet. Original error: " + str(exc)
            ) from exc

    def predict(self, image_bgr: np.ndarray) -> List[PersonDetection]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to YOLOPoseDetector.predict")

        predict_kwargs: dict[str, Any] = {
            "source": image_bgr,
            "imgsz": self.config.image_size,
            "conf": self.config.conf_threshold,
            "iou": self.config.iou_threshold,
            "max_det": self.config.max_det,
            "verbose": False,
        }
        if self.config.device:
            predict_kwargs["device"] = self.config.device
        if self.config.use_person_class_filter:
            predict_kwargs["classes"] = [self.config.person_class_id]

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
        cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy), dtype=np.float32)

        keypoints_np: np.ndarray | None = None
        kps_obj = getattr(result, "keypoints", None)
        if kps_obj is not None:
            # Ultralytics exposes .data as Nx17x3 when confidences are available.
            if getattr(kps_obj, "data", None) is not None:
                keypoints_np = kps_obj.data.detach().cpu().numpy()
            elif getattr(kps_obj, "xy", None) is not None:
                xy = kps_obj.xy.detach().cpu().numpy()
                c = np.ones((*xy.shape[:2], 1), dtype=np.float32)
                keypoints_np = np.concatenate([xy, c], axis=2)

        detections: list[PersonDetection] = []
        for idx, box in enumerate(xyxy):
            class_id = int(cls[idx]) if idx < len(cls) else 0
            if self.config.use_person_class_filter and class_id != self.config.person_class_id:
                continue
            kp = None
            if keypoints_np is not None and idx < len(keypoints_np):
                kp = keypoints_np[idx].astype(np.float32)
            detections.append(
                PersonDetection(
                    bbox_xyxy=tuple(float(v) for v in box),
                    confidence=float(conf[idx]) if idx < len(conf) else 1.0,
                    class_id=class_id,
                    keypoints=kp,
                    source="yolo",
                )
            )

        # Stable left-to-right order keeps JSON ids and click behavior predictable.
        detections.sort(key=lambda d: (d.bbox_xyxy[0], d.bbox_xyxy[1], -d.confidence))
        return detections[: self.config.max_det]


class OpenCVHOGPersonDetector:
    """Offline heuristic person detector fallback.

    It first tries OpenCV HOG full-body detection. If HOG returns nothing, it uses Haar face
    detections to infer approximate person boxes. It does not return pose keypoints. This is not a
    replacement for YOLO11-pose; it is a no-internet safety rail for testing UI, cropping, schema,
    and classifier behavior on real images.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig(backend="opencv_hog")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    def predict(self, image_bgr: np.ndarray) -> List[PersonDetection]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to OpenCVHOGPersonDetector.predict")
        detections = self._predict_hog(image_bgr)
        if detections:
            return detections
        return self._predict_face_boxes(image_bgr)

    def _predict_hog(self, image_bgr: np.ndarray) -> List[PersonDetection]:
        h, w = image_bgr.shape[:2]
        scale = 1.0
        max_side = max(h, w)
        # HOG is CPU-heavy on giant images. Resize for detection, map boxes back.
        if max_side > 720:
            scale = 720.0 / max_side
            detect_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            detect_img = image_bgr

        rects, weights = self.hog.detectMultiScale(
            detect_img,
            winStride=(16, 16),
            padding=(8, 8),
            scale=1.08,
            hitThreshold=self.config.hog_hit_threshold,
        )
        if len(rects) == 0:
            return []

        boxes_xyxy = []
        confidences = []
        inv = 1.0 / scale
        for (x, y, bw, bh), weight in zip(rects, weights):
            x1 = float(x * inv)
            y1 = float(y * inv)
            x2 = float((x + bw) * inv)
            y2 = float((y + bh) * inv)
            boxes_xyxy.append([x1, y1, x2, y2])
            # HOG weights are not calibrated probabilities. Map them to a useful display range.
            conf = float(1.0 / (1.0 + np.exp(-float(weight)))) if np.isfinite(weight) else 0.50
            confidences.append(conf)
        return self._nms_to_detections(boxes_xyxy, confidences, source="opencv_hog")

    def _predict_face_boxes(self, image_bgr: np.ndarray) -> List[PersonDetection]:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        max_side = max(h, w)
        scale = 1.0
        if max_side > 1100:
            scale = 1100.0 / max_side
            small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = gray

        min_dim = min(small.shape[:2])
        min_face = max(24, int(min_dim * 0.035))
        faces = []
        for cascade in (self.face, self.profile):
            if cascade.empty():
                continue
            found = cascade.detectMultiScale(
                small,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(min_face, min_face),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            faces.extend(found.tolist() if hasattr(found, "tolist") else list(found))

        if not faces:
            return []
        inv = 1.0 / scale
        boxes_xyxy = []
        confidences = []
        for x, y, fw, fh in faces:
            x *= inv; y *= inv; fw *= inv; fh *= inv
            # Infer an upper/full person box from face geometry. Clamp later.
            cx = x + fw / 2.0
            top = y - 0.75 * fh
            box_w = max(fw * 3.2, 45.0)
            box_h = max(fh * 6.0, 90.0)
            x1 = max(0.0, cx - box_w / 2.0)
            y1 = max(0.0, top)
            x2 = min(float(w - 1), cx + box_w / 2.0)
            y2 = min(float(h - 1), y1 + box_h)
            if (x2 - x1) < 25 or (y2 - y1) < 40:
                continue
            boxes_xyxy.append([x1, y1, x2, y2])
            # Larger faces tend to be more reliable detections.
            area_ratio = min(1.0, (fw * fh) / max(1.0, w * h) * 40.0)
            confidences.append(float(0.45 + 0.35 * area_ratio))
        return self._nms_to_detections(boxes_xyxy, confidences, source="opencv_haar_person_proxy")

    def _nms_to_detections(
        self,
        boxes_xyxy: list[list[float]],
        confidences: list[float],
        source: str,
    ) -> List[PersonDetection]:
        if not boxes_xyxy:
            return []
        boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_xyxy]
        keep = cv2.dnn.NMSBoxes(boxes_xywh, confidences, score_threshold=0.01, nms_threshold=self.config.hog_nms_threshold)
        if keep is None or len(keep) == 0:
            return []
        keep_indices = np.array(keep).reshape(-1).tolist()
        detections = [
            PersonDetection(
                bbox_xyxy=tuple(float(v) for v in boxes_xyxy[i]),
                confidence=float(confidences[i]),
                class_id=0,
                keypoints=None,
                source=source,
            )
            for i in keep_indices
        ]
        detections.sort(key=lambda d: (d.bbox_xyxy[0], d.bbox_xyxy[1], -d.confidence))
        return detections[: self.config.max_det]


class AutoPersonDetector:
    """Try YOLO first, fall back to HOG only if explicitly requested through backend='auto'."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig(backend="auto")
        self.backend_name = "yolo"
        try:
            self.detector: DetectorProtocol = YOLOPoseDetector(self.config)
        except Exception as exc:
            self.backend_name = "opencv_hog"
            self.yolo_error = str(exc)
            self.detector = OpenCVHOGPersonDetector(self.config)

    def predict(self, image_bgr: np.ndarray) -> List[PersonDetection]:
        return self.detector.predict(image_bgr)


def build_detector(config: DetectorConfig | None = None) -> DetectorProtocol:
    cfg = config or DetectorConfig()
    backend = cfg.backend.lower().strip()
    if backend == "yolo":
        return YOLOPoseDetector(cfg)
    if backend == "opencv_hog":
        return OpenCVHOGPersonDetector(cfg)
    if backend == "auto":
        return AutoPersonDetector(cfg)
    raise ValueError("Detector backend must be one of: yolo, opencv_hog, auto")
