from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from person_detector_classifier.config import PipelineConfig
from person_detector_classifier.src.detector import PersonDetection
from person_detector_classifier.src.pipeline import PersonDetectorClassifierPipeline
from person_detector_classifier.src.preprocessing import crop_bgr, expand_bbox_xyxy
from person_detector_classifier.src.schema import make_dataset_payload, make_image_record
from person_detector_classifier.src.utils import ensure_dir, save_json, write_image
from person_detector_classifier.src.visualizer import draw_annotations, draw_detections_preview


def _bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _resize_for_canvas(image_bgr: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image dimensions")
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = image_bgr.copy()
    return resized, scale


class LocalPDCExaminer:
    """Tkinter examiner for Windows/Linux/macOS desktop sessions.

    Detection happens once after image load. Classification and heuristic JSON are computed only after a
    person is clicked. This keeps the main frame responsive and avoids classifying irrelevant people.
    """

    def __init__(self, config: PipelineConfig | None = None, output_dir: str | Path = "outputs/examiner"):
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
            from PIL import Image, ImageTk
        except Exception as exc:  # pragma: no cover - GUI dependency path
            raise RuntimeError(
                "Local examiner needs tkinter and Pillow. On Colab, use launch_colab_examiner instead."
            ) from exc

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.Image = Image
        self.ImageTk = ImageTk
        self.output_dir = ensure_dir(output_dir)
        self.pipeline = PersonDetectorClassifierPipeline(config)

        self.root = tk.Tk()
        self.root.title("PDC v3 Interactive Examiner")
        self.root.geometry("1160x780")
        self.root.minsize(900, 620)
        self.root.configure(bg="#0f1724")

        self.image_path: Path | None = None
        self.image_bgr: np.ndarray | None = None
        self.preview_bgr: np.ndarray | None = None
        self.display_scale = 1.0
        self.image_id = ""
        self.detections: list[PersonDetection] = []
        self.annotations_by_index: dict[int, dict[str, Any]] = {}

        self._build_layout()

    def _build_layout(self) -> None:
        tk = self.tk
        top = tk.Frame(self.root, bg="#0f1724")
        top.pack(fill="x", padx=16, pady=(14, 8))

        title = tk.Label(
            top,
            text="PDC v3 Interactive Examiner",
            fg="#f8fafc",
            bg="#0f1724",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(side="left")

        btn_open = tk.Button(
            top,
            text="Browse Image",
            command=self.open_image_dialog,
            bg="#38bdf8",
            fg="#07111f",
            activebackground="#7dd3fc",
            relief="flat",
            padx=18,
            pady=8,
            font=("Segoe UI", 10, "bold"),
        )
        btn_open.pack(side="right", padx=(8, 0))

        btn_save = tk.Button(
            top,
            text="Save Selected JSON",
            command=self.save_selected_json,
            bg="#22c55e",
            fg="#06170c",
            activebackground="#86efac",
            relief="flat",
            padx=18,
            pady=8,
            font=("Segoe UI", 10, "bold"),
        )
        btn_save.pack(side="right", padx=(8, 0))

        body = tk.Frame(self.root, bg="#0f1724")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.canvas = tk.Canvas(body, bg="#111827", highlightthickness=1, highlightbackground="#263244")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        side = tk.Frame(body, bg="#111827", width=280)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        self.status = tk.Label(
            side,
            text="Load an image. Detections will appear here.",
            fg="#cbd5e1",
            bg="#111827",
            wraplength=240,
            justify="left",
            font=("Segoe UI", 10),
        )
        self.status.pack(fill="x", padx=16, pady=16)

        self.listbox = tk.Listbox(
            side,
            bg="#0b1220",
            fg="#e5e7eb",
            selectbackground="#38bdf8",
            selectforeground="#06121f",
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 10),
        )
        self.listbox.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.listbox.bind("<Double-Button-1>", self.on_listbox_double_click)

        hint = tk.Label(
            side,
            text="Click a bounding box or double-click a person id. The result window opens after MobileNet + heuristics run.",
            fg="#94a3b8",
            bg="#111827",
            wraplength=240,
            justify="left",
            font=("Segoe UI", 9),
        )
        hint.pack(fill="x", padx=16, pady=(0, 16))

    def open_image_dialog(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Choose an image for PDC examination",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.load_image(path)

    def load_image(self, image_path: str | Path) -> None:
        self.canvas.delete("all")
        self.annotations_by_index.clear()
        self.listbox.delete(0, self.tk.END)
        self.status.config(text="Running detector once. The window is alive, unlike many rushed demos...")
        self.root.update_idletasks()

        try:
            self.image_path = Path(image_path)
            self.image_bgr, self.image_id, self.detections = self.pipeline.detect_image(self.image_path)
            self.preview_bgr = draw_detections_preview(
                self.image_bgr,
                self.detections,
                keypoint_threshold=self.pipeline.config.attributes.keypoint_conf_threshold,
            )
            self.render_preview()
            if self.detections:
                self.status.config(text=f"Detected {len(self.detections)} person(s). Click a person to examine.")
                for i, det in enumerate(self.detections, start=1):
                    self.listbox.insert(self.tk.END, f"#{i:<2} conf={det.confidence:.2f} src={det.source}")
            else:
                self.status.config(text="No persons detected. Try a clearer image or lower detection confidence.")
        except Exception as exc:
            self.messagebox.showerror("PDC examiner error", str(exc))
            self.status.config(text="Image loading/detection failed. Check model path and dependencies.")

    def render_preview(self) -> None:
        if self.preview_bgr is None:
            return
        canvas_w = max(100, self.canvas.winfo_width() or 820)
        canvas_h = max(100, self.canvas.winfo_height() or 620)
        display_bgr, self.display_scale = _resize_for_canvas(self.preview_bgr, canvas_w - 20, canvas_h - 20)
        rgb = _bgr_to_rgb(display_bgr)
        pil = self.Image.fromarray(rgb)
        self.photo = self.ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(10, 10, image=self.photo, anchor="nw")

    def _canvas_to_image_xy(self, canvas_x: int, canvas_y: int) -> tuple[float, float]:
        return (canvas_x - 10) / self.display_scale, (canvas_y - 10) / self.display_scale

    def _pick_detection(self, x: float, y: float) -> int | None:
        hits = [(i, d.area, d.confidence) for i, d in enumerate(self.detections) if d.contains(x, y)]
        if not hits:
            return None
        # If boxes overlap, choose the smallest hit area, then highest confidence.
        hits.sort(key=lambda item: (item[1], -item[2]))
        return hits[0][0]

    def on_canvas_click(self, event: Any) -> None:
        if self.image_bgr is None:
            return
        x, y = self._canvas_to_image_xy(event.x, event.y)
        idx = self._pick_detection(x, y)
        if idx is None:
            self.status.config(text="Clicked empty area. Aim for a bbox, cruel as precision may be.")
            return
        self.examine_index(idx)

    def on_listbox_double_click(self, _event: Any) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        self.examine_index(int(selection[0]))

    def examine_index(self, idx: int) -> None:
        if self.image_bgr is None or idx < 0 or idx >= len(self.detections):
            return
        self.status.config(text=f"Examining person #{idx + 1}: MobileNet + heuristics running...")
        self.root.update_idletasks()
        annotation = self.pipeline.analyze_detection(
            image_bgr=self.image_bgr,
            image_id=self.image_id,
            detection=self.detections[idx],
            annotation_id=idx + 1,
            include_keypoints=True,
            all_detections=self.detections,
        )
        self.annotations_by_index[idx] = annotation
        self.status.config(text=f"Person #{idx + 1} examined. You may choose another person.")
        self._show_result_window(idx, annotation)

    def _show_result_window(self, idx: int, annotation: dict[str, Any]) -> None:
        if self.image_bgr is None:
            return
        tk = self.tk
        win = tk.Toplevel(self.root)
        win.title(f"PDC Result - Person #{idx + 1}")
        win.geometry("430x470")
        win.resizable(False, False)
        win.configure(bg="#0f1724")
        win.grab_set()

        header = tk.Label(
            win,
            text=f"Person #{idx + 1} Analysis",
            fg="#f8fafc",
            bg="#0f1724",
            font=("Segoe UI", 17, "bold"),
        )
        header.pack(pady=(18, 4))

        det = self.detections[idx]
        attrs = annotation.get("attributes", {})
        crop_info = attrs.get("classifier_crop", {}) if isinstance(attrs, dict) else {}
        crop_box = crop_info.get("bbox") if isinstance(crop_info, dict) else None
        crop = None
        if crop_box and len(crop_box) == 4:
            crop = crop_bgr(self.image_bgr, tuple(int(v) for v in crop_box))
        if crop is None:
            padded = expand_bbox_xyxy(det.bbox_xyxy, self.image_bgr.shape[1], self.image_bgr.shape[0], 0.08)
            crop = crop_bgr(self.image_bgr, padded)
        if crop is not None:
            thumb, _ = _resize_for_canvas(crop, 160, 160)
            rgb = _bgr_to_rgb(thumb)
            pil = self.Image.fromarray(rgb)
            self.result_photo = self.ImageTk.PhotoImage(pil)
            img_label = tk.Label(win, image=self.result_photo, bg="#0f1724")
            img_label.pack(pady=(8, 4))
            cap = tk.Label(
                win,
                text=f"Classifier crop: {crop_info.get('mode', 'fallback')} · {crop_info.get('quality', 'unknown')}",
                fg="#94a3b8",
                bg="#0f1724",
                font=("Segoe UI", 8),
            )
            cap.pack(pady=(0, 8))

        rows = [
            ("Age Group", attrs.get("age_group", "unknown")),
            ("Age Confidence", attrs.get("age_confidence", 0.0)),
            ("Gender", attrs.get("gender", "unknown")),
            ("Gender Confidence", attrs.get("gender_confidence", 0.0)),
            ("Visibility", attrs.get("visibility", "unknown")),
            ("Occlusion", attrs.get("occlusion", "unknown")),
            ("Quality", attrs.get("quality", "unknown")),
            ("Rotation", f"{attrs.get('rotation', 0.0)}°"),
            ("Cls Crop", crop_info.get("mode", "unknown") if isinstance(crop_info, dict) else "unknown"),
            ("BBox", json.dumps(annotation.get("bbox", []))),
        ]
        panel = tk.Frame(win, bg="#111827")
        panel.pack(fill="x", padx=22, pady=(0, 14))
        for label, value in rows:
            row = tk.Frame(panel, bg="#111827")
            row.pack(fill="x", padx=14, pady=5)
            tk.Label(row, text=label, fg="#94a3b8", bg="#111827", font=("Segoe UI", 9, "bold"), width=13, anchor="w").pack(side="left")
            tk.Label(row, text=str(value), fg="#e5e7eb", bg="#111827", font=("Segoe UI", 9), anchor="w").pack(side="left", fill="x", expand=True)

        btn = tk.Button(
            win,
            text="Back to People",
            command=win.destroy,
            bg="#38bdf8",
            fg="#06121f",
            activebackground="#7dd3fc",
            relief="flat",
            padx=20,
            pady=8,
            font=("Segoe UI", 10, "bold"),
        )
        btn.pack(pady=(0, 16))

    def save_selected_json(self) -> None:
        if self.image_bgr is None or self.image_path is None:
            self.messagebox.showinfo("PDC", "No image loaded yet.")
            return
        anns = [self.annotations_by_index[i] for i in sorted(self.annotations_by_index)]
        image_h, image_w = self.image_bgr.shape[:2]
        payload = make_dataset_payload(
            images=[make_image_record(self.image_id, self.image_path.name, image_w, image_h)],
            annotations=anns,
            version="v3_examiner_selected",
        )
        out_json = self.output_dir / f"{self.image_path.stem}_selected.json"
        save_json(out_json, payload)
        if anns:
            kept = [self.detections[i] for i in sorted(self.annotations_by_index)]
            visual = draw_annotations(self.image_bgr, kept, anns, self.pipeline.config.attributes.keypoint_conf_threshold)
        else:
            visual = self.preview_bgr if self.preview_bgr is not None else self.image_bgr
        out_img = self.output_dir / "annotated_images" / f"{self.image_path.stem}_selected.jpg"
        write_image(out_img, visual)
        self.messagebox.showinfo("PDC saved", f"Saved:\n{out_json}\n{out_img}")

    def run(self) -> None:
        self.root.mainloop()


def launch_local_examiner(config: PipelineConfig | None = None, output_dir: str | Path = "outputs/examiner") -> None:
    app = LocalPDCExaminer(config=config, output_dir=output_dir)
    app.run()
