"""Normalized cross-correlation with a part of the image"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageChops, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

import numpy as np
from skimage.feature import match_template

class Norm_cross_corr(ForensicsTool):
    tool_id = "normalizedcrosscorrelation"
    title = "Normalized Cross-correlation"
    category = "Set 4"
    description = ""

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        selection_window = tk.Toplevel(parent)
        selection_window.title("Select the template: ")

        canvas = tk.Canvas(
            selection_window,
            width=document.current.width,
            height=document.current.height
        )
        canvas.pack()

        preview = ImageTk.PhotoImage(document.current, master=selection_window)
        canvas.create_image(0, 0, anchor="nw", image=preview)
        canvas.image = preview
        start_x = 0
        start_y = 0
        rectangle_id = None
        template = None

        def on_press(event):
            nonlocal start_x, start_y, rectangle_id

            start_x = event.x
            start_y = event.y

            if rectangle_id is not None:
                canvas.delete(rectangle_id)

            rectangle_id = canvas.create_rectangle(
                start_x, start_y, start_x, start_y,
                outline="red", width=2,
            )

        def on_drag(event):
            if rectangle_id is not None:
                canvas.coords(
                    rectangle_id,
                    start_x, start_y,
                    event.x, event.y,
                )
        def on_release(event):
            nonlocal template

            end_x = max(0, min(event.x, document.current.width))
            end_y = max(0, min(event.y, document.current.height))

            left = min(start_x, end_x)
            top = min(start_y, end_y)
            right = max(start_x, end_x)
            bottom = max(start_y, end_y)

            if left == right or top == bottom:
                return

            template = document.current.crop((left, top, right, bottom))
            selection_window.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        parent.wait_window(selection_window)

        if template is None:
            return None
        threshold = simpledialog.askfloat(
                    "Optional Threshold 0:1" ,
                    "Cancel=show correlation map",
                    minvalue=0.0,
                    maxvalue=1.0,
                    parent=parent,
                )
        angle = simpledialog.askfloat(
            title="Template angle",
            prompt="Cancel= without rotation",
            initialvalue=0.0,
            minvalue=-180.0,
            maxvalue=180.0,
            parent=parent,
        )
        if angle is None:
            angle = 0
        template = template.rotate(angle)
        image_gray = np.asarray(document.current.convert("L"))
        template_gray = np.asarray(template.convert("L"))

        correlation = match_template(
            image_gray,
            template_gray,
            pad_input=True,
        )
        pixels = (np.clip(correlation, -1, 1) + 1) * 127.5 # we want values between 1:255
        correlation_image = Image.fromarray(pixels.astype(np.uint8))


        

        if threshold is None:
            return ToolResult(
                image=correlation_image,
                message="Normalized cross-correlation calculated.",
            )

        pixels = np.asarray(
            document.current.convert("RGB"),
            dtype=np.float32,
        ).copy()

        pixels *= 0.3


        matches = correlation >= threshold
        pixels[matches] = [255, 0, 0]

        return ToolResult(
            image=Image.fromarray(pixels.astype(np.uint8)),
            message=f"Cross-correlation with threshold {threshold}.",
        )
