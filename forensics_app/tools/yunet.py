"""Face detection using the YuNet model from the Set 6 notebook."""

from pathlib import Path
import tkinter as tk
from tkinter import simpledialog

import cv2
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class YuNetTool(ForensicsTool):
    tool_id = "yunet"
    title = "YuNet"
    category = "Set 6"
    description = "Detect faces using YuNet."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        threshold = simpledialog.askfloat(
            "YuNet", "Minimum confidence threshold (0–1):", parent=parent,
            initialvalue=0.6, minvalue=0.0, maxvalue=1.0,
        )
        if threshold is None:
            return None
        image = cv2.cvtColor(np.array(document.current.convert("RGB")), cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        model_path = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
        detector = cv2.FaceDetectorYN.create(str(model_path), "", (width, height), threshold, 0.3, 5000)
        _, faces = detector.detect(image)

        for face in faces if faces is not None else ():
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"Face: {face[14]:.2f}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for px, py in face[4:14].reshape(5, 2).astype(int):
                cv2.circle(image, (px, py), 4, (255, 0, 0), -1)

        count = 0 if faces is None else len(faces)
        return ToolResult(
            image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            message=f"Detected {count} face(s) using YuNet.",
            details={"Detector": "YuNet", "Confidence threshold": threshold, "Faces": count},
        )
