import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image
from skimage import morphology

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class MorphologyDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, width: int, height: int, title: str = "Structuring Elements") -> None:
        self.width = width
        self.height = height
        self.max_dim = max(width, height)
        super().__init__(parent, title)

    def _get_param_info(self, shape: str) -> tuple[str, int, int]:
        min_dim = min(self.width, self.height)
        if shape in {"Disk", "Diamond"}:
            return "Radius", min_dim // 2, 25
        if shape == "Square":
            return "Width", min_dim, 50
        if shape == "Star":
            return "Size (a)", min_dim // 4, 15
        return "Size", self.max_dim, 25

    def body(self, master: tk.Frame) -> tk.Widget:
        ttk.Label(master, text="Shape:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.shape_combo = ttk.Combobox(
            master,
            values=["Disk", "Diamond", "Square", "Star"],
            state="readonly",
        )
        self.shape_combo.set("Disk")
        self.shape_combo.grid(row=0, column=1, padx=6, pady=6)
        self.shape_combo.bind("<<ComboboxSelected>>", self._on_shape_change)

        param_name, limit, default = self._get_param_info("Disk")
        self.param_label = ttk.Label(master, text=f"{param_name} (1–{limit}):")
        self.param_label.grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.size_entry = ttk.Entry(master)
        self.size_entry.insert(0, str(min(default, limit)))
        self.size_entry.grid(row=1, column=1, padx=6, pady=6)

        ttk.Label(master, text="Background:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        self.bg_combo = ttk.Combobox(
            master,
            values=["Black background", "Current image (grayscale)"],
            state="readonly",
        )
        self.bg_combo.set("Black background")
        self.bg_combo.grid(row=2, column=1, padx=6, pady=6)

        return self.shape_combo

    def _on_shape_change(self, _event=None) -> None:
        shape = self.shape_combo.get()
        param_name, limit, default = self._get_param_info(shape)
        self.param_label.configure(text=f"{param_name} (1–{limit}):")
        self.size_entry.delete(0, tk.END)
        self.size_entry.insert(0, str(min(default, limit)))

    def validate(self) -> bool:
        shape = self.shape_combo.get()
        param_name, limit, _ = self._get_param_info(shape)
        try:
            size = int(self.size_entry.get())
            if not (1 <= size <= limit):
                messagebox.showerror(
                    "Invalid Parameter",
                    f"{param_name} must be between 1 and {limit} for {shape}.",
                    parent=self,
                )
                return False
            return True
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer.", parent=self)
            return False

    def apply(self) -> None:
        self.result = (self.shape_combo.get(), int(self.size_entry.get()), self.bg_combo.get())


class MorphologyTool(ForensicsTool):
    tool_id = "morphology_elements"
    title = "Morphology Elements"
    category = "Set 3"
    description = "Generate 2D structuring element shapes (disk, diamond, square, star)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        width, height = document.current.size

        dialog = MorphologyDialog(parent, width, height)
        if dialog.result is None:
            return None

        shape, size, bg_type = dialog.result

        if shape == "Disk":
            selem = morphology.disk(size)
            param_name = "Radius"
        elif shape == "Diamond":
            selem = morphology.diamond(size)
            param_name = "Radius"
        elif shape == "Square":
            selem = morphology.square(size)
            param_name = "Width"
        elif shape == "Star":
            selem = morphology.star(size)
            param_name = "Size (a)"
        else:
            return None

        if bg_type == "Black background":
            canvas = np.zeros((height, width), dtype=np.uint8)
        else:
            canvas = np.array(document.current.convert("L"), dtype=np.uint8)

        sh, sw = selem.shape[:2]
        y0 = (height - sh) // 2
        x0 = (width - sw) // 2

        y1 = max(0, y0)
        y2 = min(height, y0 + sh)
        x1 = max(0, x0)
        x2 = min(width, x0 + sw)

        sy1 = max(0, -y0)
        sy2 = sy1 + (y2 - y1)
        sx1 = max(0, -x0)
        sx2 = sx1 + (x2 - x1)

        canvas[y1:y2, x1:x2] = np.where(selem[sy1:sy2, sx1:sx2] > 0, 255, canvas[y1:y2, x1:x2])

        output = Image.fromarray(canvas)
        return ToolResult(
            image=output,
            message=f"Created {shape.lower()} structuring element ({param_name.lower()}={size}) on {bg_type.lower()}.",
            details={
                "Operation": "Morphology Element",
                "Shape": shape,
                param_name: size,
                "Element shape": f"{selem.shape[0]} × {selem.shape[1]}",
                "Background": bg_type,
            },
        )
