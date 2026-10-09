import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ConvolutionDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, width: int, height: int, title: str = "Convolution") -> None:
        self.width = width
        self.height = height
        self.kernel_size = 15
        self.direction = "2D"
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self.max_label = ttk.Label(master, text=f"Kernel size (1–{min(self.width, self.height)}):")
        self.max_label.grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.size_entry = ttk.Entry(master)
        self.size_entry.insert(0, str(min(15, self.width, self.height)))
        self.size_entry.grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(master, text="Direction:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.dir_combo = ttk.Combobox(
            master,
            values=["2D", "Horizontal (1D)", "Vertical (1D)"],
            state="readonly",
        )
        self.dir_combo.set("2D")
        self.dir_combo.grid(row=1, column=1, padx=6, pady=6)
        self.dir_combo.bind("<<ComboboxSelected>>", self._on_dir_change)
        return self.size_entry

    def _get_max_for_dir(self, direction: str) -> int:
        if direction == "Horizontal (1D)":
            return self.width
        if direction == "Vertical (1D)":
            return self.height
        return min(self.width, self.height)

    def _on_dir_change(self, _event=None) -> None:
        limit = self._get_max_for_dir(self.dir_combo.get())
        self.max_label.configure(text=f"Kernel size (1–{limit}):")

    def validate(self) -> bool:
        try:
            val = int(self.size_entry.get())
            direction = self.dir_combo.get()
            limit = self._get_max_for_dir(direction)
            if not (1 <= val <= limit):
                messagebox.showerror(
                    "Invalid Kernel Size",
                    f"Kernel size must be between 1 and {limit} for {direction} convolution.",
                    parent=self,
                )
                return False
            return True
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter a valid integer kernel size.",
                parent=self,
            )
            return False

    def apply(self) -> None:
        self.kernel_size = int(self.size_entry.get())
        self.direction = self.dir_combo.get()
        self.result = (self.kernel_size, self.direction)


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution (Arbitrary Kernel)"
    category = "Set 3"
    description = "Convolve grayscale image with arbitrary 1D or 2D kernel."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        width, height = document.current.size

        dialog = ConvolutionDialog(parent, width, height)
        if dialog.result is None:
            return None

        size, direction = dialog.result
        gray = np.array(document.current.convert("L"), dtype=np.float64)

        if direction == "Horizontal (1D)":
            mask = np.ones((1, size), dtype=np.float64) / size
            conv = ndimage.uniform_filter1d(gray, size=size, axis=1) if size > 64 else ndimage.convolve(gray, mask)
        elif direction == "Vertical (1D)":
            mask = np.ones((size, 1), dtype=np.float64) / size
            conv = ndimage.uniform_filter1d(gray, size=size, axis=0) if size > 64 else ndimage.convolve(gray, mask)
        else:
            mask = np.ones((size, size), dtype=np.float64) / (size * size)
            conv = ndimage.uniform_filter(gray, size=size) if size > 64 else ndimage.convolve(gray, mask)

        output = Image.fromarray(np.clip(conv, 0, 255).astype(np.uint8))

        return ToolResult(
            image=output,
            message=f"Applied {direction} convolution with kernel size {size}.",
            details={
                "Operation": "Convolution",
                "Direction": direction,
                "Kernel size": size,
                "Kernel shape": f"{mask.shape[0]} × {mask.shape[1]}",
                "Input size": f"{document.current.width} × {document.current.height}",
                "Output mode": output.mode,
            },
        )
