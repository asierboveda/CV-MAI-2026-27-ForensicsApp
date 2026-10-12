import math
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def compute_canny_edges(image: Image.Image, sigma: float) -> np.ndarray:
    gray = np.array(image.convert("L"), dtype=np.float64) / 255.0
    return feature.canny(gray, sigma=sigma)


class CannyTool(ForensicsTool):
    tool_id = "canny"
    title = "Canny edge detection"
    category = "Set 3"
    description = "Detect edges with a configurable Gaussian sigma."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        while True:
            sigma = simpledialog.askfloat(
                self.title, "Sigma (pixels):", parent=parent, initialvalue=1.0, minvalue=0.0
            )
            if sigma is None:
                return None
            if math.isfinite(sigma) and sigma >= 0:
                break
            messagebox.showerror("Invalid sigma", "Sigma must be finite and non-negative.", parent=parent)

        edges = compute_canny_edges(document.current, sigma)
        output = Image.fromarray(edges.astype(np.uint8) * 255)
        return ToolResult(
            image=output,
            message=f"Detected Canny edges (sigma {sigma:g}).",
            details={"Operation": "Canny", "Sigma": sigma, "Output mode": output.mode},
        )
