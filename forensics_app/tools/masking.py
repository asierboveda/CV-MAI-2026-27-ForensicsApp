"""Composite a black-background foreground or texture over the current image."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageChops

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def compose_masked(base, foreground, threshold, texture=None):
    """Select non-black foreground pixels and copy them onto an aligned base.

    The binary mask is white inside the selected region and black elsewhere.
    An optional texture is resized to the whole image and clipped by that mask.
    """
    if base.size != foreground.size:
        raise ValueError("Base and foreground must have the same size and be aligned.")
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be between 0 and 255.")
    red, green, blue = foreground.convert("RGB").split()
    brightness = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    mask = brightness.point(lambda value: 255 if value > threshold else 0)
    if "A" in foreground.getbands() or "transparency" in foreground.info:
        alpha_mask = foreground.convert("RGBA").getchannel("A").point(
            lambda value: 255 if value > 0 else 0
        )
        mask = ImageChops.multiply(mask, alpha_mask)

    mode = "RGBA" if "A" in base.getbands() or "transparency" in base.info else "RGB"
    fill = foreground if texture is None else texture.resize(base.size, Image.Resampling.LANCZOS)
    return Image.composite(fill.convert(mode), base.convert(mode), mask), mask


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking"
    category = "Set 2"
    description = "Overlay a foreground on black, optionally filled with a texture."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(
            title="Choose aligned foreground on black (e.g. coat.png)",
            parent=parent,
        )
        if not filename:
            return None
        with Image.open(filename) as image:
            foreground = image.copy()
        if foreground.size != document.current.size:
            raise ValueError("Base and foreground must have the same size and be aligned.")

        threshold = simpledialog.askinteger(
            "Masking", "Black-background threshold (0–255):",
            parent=parent, initialvalue=10, minvalue=0, maxvalue=255,
        )
        if threshold is None:
            return None

        texture = None
        texture_path = None
        if messagebox.askyesno(
            "Masking", "Fill the selected region with a texture?\n"
            "Choose No to keep the foreground's original colors.", parent=parent,
        ):
            texture_path = filedialog.askopenfilename(title="Choose texture", parent=parent)
            if not texture_path:
                return None
            with Image.open(texture_path) as image:
                texture = image.copy()

        output, mask = compose_masked(document.current, foreground, threshold, texture)
        selected = mask.histogram()[255]
        if selected == 0:
            raise ValueError("The mask is empty. Try a lower threshold or another foreground.")
        return ToolResult(
            image=output,
            message="Applied masking. Use Reset before trying another texture.",
            details={
                "Operation": "Masking",
                "Foreground": Path(filename).name,
                "Threshold": threshold,
                "Selected pixels": selected,
                "Coverage": f"{100 * selected / (mask.width * mask.height):.1f}%",
                "Texture": Path(texture_path).name if texture_path else "None",
                "Texture resized to": f"{output.width} × {output.height}" if texture else "N/A",
            },
        )
