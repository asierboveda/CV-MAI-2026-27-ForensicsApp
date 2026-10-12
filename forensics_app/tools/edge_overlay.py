import math
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .canny import compute_canny_edges


class EdgeOverlayTool(ForensicsTool):
    tool_id = "edge_overlay"
    title = "Edge overlay"
    category = "Set 3"
    description = "Superimpose Canny edges in red on the original image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.original is not None
        while True:
            sigma = simpledialog.askfloat(
                self.title, "Sigma (pixels):", parent=parent, initialvalue=1.0, minvalue=0.0
            )
            if sigma is None:
                return None
            if math.isfinite(sigma) and sigma >= 0:
                break
            messagebox.showerror("Invalid sigma", "Sigma must be finite and non-negative.", parent=parent)

        edges = compute_canny_edges(document.original, sigma)
        pixels = np.array(document.original.convert("RGB"))
        pixels[edges] = (255, 0, 0)
        output = Image.fromarray(pixels)
        return ToolResult(
            image=output,
            message=f"Overlaid Canny edges on the original image (sigma {sigma:g}).",
            details={"Operation": "Edge overlay", "Source": "Original", "Sigma": sigma},
        )
