"""HOG extraction and visualization."""

import tkinter as tk
from tkinter import messagebox, simpledialog
from skimage.feature import hog
from skimage.exposure import rescale_intensity
import numpy as np
from PIL import Image
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class HOGTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG extraction and visualization"
    category = "Set 4"
    description = "Extract and visualize Histogram of Oriented Gradients."

    def run(self, parent: tk.Misc, document: ImageDocument,) -> ToolResult | None:
        assert document.current is not None

        image = document.current.copy()
        orientations = simpledialog.askinteger(
            title= "Orientations",
            prompt="Number of intervals ",
            initialvalue=9,
            minvalue=2,
            parent=parent,
        )
        block_h = simpledialog.askinteger(
                    title= "Block high",
                    prompt="Block high ",
                    initialvalue=3,
                    minvalue=1,
                    parent=parent,
                )
        block_w = simpledialog.askinteger(
                            title= "Block width",
                            prompt="Block witdhs ",
                            initialvalue=3,
                            minvalue=1,
                            parent=parent,
                        )
        pix_cel_h = simpledialog.askinteger(
                    title= "Pixels per cell high",
                    prompt="Pixels per cell high ",
                    initialvalue=3,
                    minvalue=1,
                    parent=parent,
                )
        pix_cel_w = simpledialog.askinteger(
                            title= "Pixels per cell width",
                            prompt="Pixels per cell width ",
                            initialvalue=3,
                            minvalue=1,
                            parent=parent,
                        )
                
        for i in [orientations,block_w, block_h, pix_cel_w, pix_cel_h]:
            if i is None:
                return None
        required_height = block_h * pix_cel_h
        required_width = block_w * pix_cel_w
        if image.height < required_height or image.width < required_width:
            messagebox.showerror(
                "Invalid HOG parameters",
                f"These settings require an image of at least "
                f"{required_width} x {required_height} pixels. "
                "Choose smaller cells or blocks.",
                parent=parent,
            )
            return None
        image_rgb = np.asarray(image.convert("RGB"))
        feature, hog_image = hog(image_rgb, orientations = orientations, pixels_per_cell =(pix_cel_h,pix_cel_w), cells_per_block=(block_h,block_w), visualize = True, channel_axis = 2)

                
        pixels = rescale_intensity(
            hog_image,
            in_range="image",
            out_range=(0, 255),
        )

        output = Image.fromarray(pixels.astype(np.uint8))
        return ToolResult(
            image=output,
            message="HOG calculated.",
            details={
                "Orientations": orientations,
                "Pixels per cell": (pix_cel_h, pix_cel_w),
                "Cells per block": (block_h, block_w),
                "Features": feature.size,
    },
        )
