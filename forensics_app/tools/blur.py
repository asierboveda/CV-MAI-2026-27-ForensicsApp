import tkinter as tk
from tkinter import simpledialog

from PIL import ImageFilter

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

class BlurTool(ForensicsTool):
    toot_id = "gaussian blur"
    title = "Gaussian Blur"
    category = "Filtering"
    description = "Applies a Gaussian blur filter to the image."

    def run(self, parent: tk.Misc, image_document: ImageDocument) -> ToolResult:
        radius = simpledialog.askfloat(
            "Gaussian Blur", 
            "Enter blur radius:", 
            parent=parent, 
            minvalue=0.1, 
            maxvalue=100.0
        )
        if radius is None:
            return None

        output = image_document.current.filter(ImageFilter.GaussianBlur(radius))

        return ToolResult(
            image=output,
            message=f"Applied Gaussian blur with radius {radius}.",
            details={"Operation": "Gaussian Blur", "Radius": radius}
        )