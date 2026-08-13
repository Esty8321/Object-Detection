from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from person_detector_classifier.config import PipelineConfig
from person_detector_classifier.src.pipeline import PersonDetectorClassifierPipeline
from person_detector_classifier.src.schema import make_dataset_payload, make_image_record
from person_detector_classifier.src.utils import ensure_dir, save_json, write_image
from person_detector_classifier.src.visualizer import draw_annotations, draw_detections_preview


def _is_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def launch_colab_examiner(
    image_path: str | Path | None = None,
    config: PipelineConfig | None = None,
    output_dir: str | Path = "outputs/examiner_colab",
) -> dict[str, Any] | None:
    """Colab-friendly examiner.

    Colab cannot open persistent desktop windows or capture raw image clicks reliably without custom JS.
    This function gives the same workflow through upload + preview + person buttons.
    """
    try:
        from IPython.display import clear_output, display
        import ipywidgets as widgets
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Colab examiner requires IPython, ipywidgets, and Pillow.") from exc

    out_dir = ensure_dir(output_dir)
    pipeline = PersonDetectorClassifierPipeline(config)
    state: dict[str, Any] = {"annotations": {}, "image_bgr": None, "detections": [], "image_id": "", "image_path": None}
    output = widgets.Output()

    def show_bgr(image_bgr: np.ndarray, width: int = 900) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        display(Image.fromarray(rgb).resize((width, int(rgb.shape[0] * width / rgb.shape[1]))))

    def load(path: str | Path) -> None:
        image_bgr, image_id, detections = pipeline.detect_image(path)
        state.update({"image_bgr": image_bgr, "detections": detections, "image_id": image_id, "image_path": Path(path), "annotations": {}})
        preview = draw_detections_preview(image_bgr, detections, pipeline.config.attributes.keypoint_conf_threshold)
        with output:
            clear_output(wait=True)
            print(f"Loaded: {Path(path).name}")
            print(f"Detected persons: {len(detections)}")
            show_bgr(preview)
            if not detections:
                print("No person detected. Try a clearer image or lower detector confidence.")
                return
            buttons = []
            for idx, det in enumerate(detections):
                btn = widgets.Button(
                    description=f"Examine #{idx + 1}",
                    tooltip=f"conf={det.confidence:.2f}, source={det.source}",
                    button_style="info",
                    layout=widgets.Layout(width="150px"),
                )
                btn.on_click(lambda _b, i=idx: examine(i))
                buttons.append(btn)
            save_btn = widgets.Button(description="Save Selected JSON", button_style="success", layout=widgets.Layout(width="180px"))
            save_btn.on_click(lambda _b: save_selected())
            display(widgets.HBox(buttons + [save_btn]))

    def examine(idx: int) -> None:
        image_bgr = state["image_bgr"]
        if image_bgr is None:
            return
        det = state["detections"][idx]
        ann = pipeline.analyze_detection(image_bgr, state["image_id"], det, annotation_id=idx + 1, include_keypoints=True, all_detections=state["detections"])
        state["annotations"][idx] = ann
        attrs = ann.get("attributes", {})
        crop_info = attrs.get("classifier_crop", {}) if isinstance(attrs, dict) else {}
        crop_box = crop_info.get("bbox") if isinstance(crop_info, dict) else None
        crop = None
        if crop_box and len(crop_box) == 4:
            x1, y1, x2, y2 = [int(v) for v in crop_box]
            crop = image_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop is None or crop.size == 0:
            crop = image_bgr[int(max(0, det.bbox_xyxy[1])):int(max(0, det.bbox_xyxy[3])), int(max(0, det.bbox_xyxy[0])):int(max(0, det.bbox_xyxy[2]))]

        with output:
            print("\n" + "=" * 72)
            print(f"Person #{idx + 1} result")
            print(f"Age Group : {attrs.get('age_group', 'unknown')}")
            print(f"Gender    : {attrs.get('gender', 'unknown')}")
            print(f"Visibility: {attrs.get('visibility', 'unknown')}")
            print(f"Occlusion : {attrs.get('occlusion', 'unknown')}")
            print(f"Quality   : {attrs.get('quality', 'unknown')}")
            print(f"Rotation  : {attrs.get('rotation', 0.0)}°")
            print(f"Cls Crop  : {crop_info.get('mode', 'unknown') if isinstance(crop_info, dict) else 'unknown'} / {crop_info.get('quality', 'unknown') if isinstance(crop_info, dict) else 'unknown'}")
            if crop is not None and crop.size:
                show_bgr(crop, width=260)

    def save_selected() -> None:
        image_bgr = state["image_bgr"]
        image_path = state["image_path"]
        if image_bgr is None or image_path is None:
            return
        anns = [state["annotations"][i] for i in sorted(state["annotations"])]
        h, w = image_bgr.shape[:2]
        payload = make_dataset_payload(
            images=[make_image_record(state["image_id"], image_path.name, w, h)],
            annotations=anns,
            version="v3_examiner_colab_selected",
        )
        out_json = out_dir / f"{image_path.stem}_selected.json"
        save_json(out_json, payload)
        if anns:
            kept = [state["detections"][i] for i in sorted(state["annotations"])]
            visual = draw_annotations(image_bgr, kept, anns, pipeline.config.attributes.keypoint_conf_threshold)
        else:
            visual = draw_detections_preview(image_bgr, state["detections"], pipeline.config.attributes.keypoint_conf_threshold)
        out_img = out_dir / "annotated_images" / f"{image_path.stem}_selected.jpg"
        write_image(out_img, visual)
        with output:
            print(f"\nSaved JSON: {out_json}")
            print(f"Saved image: {out_img}")

    if image_path is not None:
        load(image_path)
        display(output)
        return state

    uploader = widgets.FileUpload(accept="image/*", multiple=False)

    def on_upload(change: Any) -> None:
        if not uploader.value:
            return
        item = next(iter(uploader.value.values())) if isinstance(uploader.value, dict) else uploader.value[0]
        name = item.get("metadata", {}).get("name", item.get("name", "uploaded_image.jpg"))
        content = item["content"]
        path = out_dir / name
        path.write_bytes(content)
        load(path)

    uploader.observe(on_upload, names="value")
    display(widgets.VBox([widgets.HTML("<b>PDC v3 Colab Examiner</b><br>Upload an image, then examine any detected person."), uploader, output]))
    return state
