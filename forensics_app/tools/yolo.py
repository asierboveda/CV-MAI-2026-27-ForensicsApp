"""Set 9 — YOLO object detection (COCO-pretrained or our blood-cell models)."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

SET9_DIR = Path(__file__).resolve().parent.parent / "models" / "set9"


@lru_cache(maxsize=3)
def _load_yolo(path: Path):
    from ultralytics import YOLO  # heavy import, only when the tool runs

    return YOLO(str(path))


def gt_label_path(image_path: Path | None) -> Path | None:
    """YOLO layout: .../images/x.jpg -> .../labels/x.txt (None if absent)."""
    if image_path is None:
        return None
    label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return label if label.exists() else None


def read_yolo_labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    """Normalised ``cls xc yc w h`` lines -> absolute ``(cls, x1, y1, x2, y2)``."""
    boxes = []
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        cls, xc, yc, w, h = (float(v) for v in line.split())
        boxes.append((int(cls), (xc - w / 2) * width, (yc - h / 2) * height,
                      (xc + w / 2) * width, (yc + h / 2) * height))
    return boxes


def detect(image: Image.Image, model_path: Path, conf: float, image_path: Path | None = None):
    """Run YOLO; return the annotated image, prediction counts per class and the
    number of GT boxes (None without a label file). GT class ids belong to the
    dataset, not to the model, so they are not named with ``model.names``."""
    model = _load_yolo(model_path)
    result = model.predict(image.convert("RGB"), conf=conf, verbose=False)[0]
    output = Image.fromarray(result.plot(line_width=1)[..., ::-1])  # plot() returns BGR
    predicted = Counter(model.names[int(c)] for c in result.boxes.cls.tolist())

    label_path = gt_label_path(image_path)
    if label_path is None:
        return output, predicted, None
    draw = ImageDraw.Draw(output)
    gt = read_yolo_labels(label_path, *output.size)
    for _, *box in gt:
        draw.rectangle(box, outline="black", width=2)
    return output, predicted, len(gt)


class YOLODetectionTool(ForensicsTool):
    tool_id = "yolo_detection"
    title = "YOLO detection"
    category = "Set 9"
    description = "Detect objects with a pretrained YOLO (COCO) or our blood-cell model; shows GT if labels exist."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(
            title="Select YOLO weights", parent=parent, initialdir=SET9_DIR,
            filetypes=[("YOLO weights", "*.pt")],
        )
        if not filename:
            return None
        conf = simpledialog.askfloat("YOLO", "Confidence threshold (0–1):", parent=parent,
                                     initialvalue=0.25, minvalue=0.0, maxvalue=1.0)
        if conf is None:
            return None

        output, predicted, gt = detect(document.current, Path(filename), conf, document.path)
        details = {"Model": Path(filename).name, "Confidence threshold": conf,
                   "Detections": sum(predicted.values()),
                   **{f"Pred {name}": count for name, count in sorted(predicted.items())}}
        message = f"{sum(predicted.values())} detection(s) with {Path(filename).name}."
        if gt is not None:
            details["GT boxes"] = gt
            message += " Black boxes: ground truth."
        return ToolResult(image=output, message=message, details=details)
