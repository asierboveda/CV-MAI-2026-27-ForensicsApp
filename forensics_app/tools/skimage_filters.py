import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class SkimageFilterDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, title: str = "Skimage Filters") -> None:
        self.filter_name = "Gaussian"
        self.param_value = 2.0
        self.axis_value = "Both (Magnitude)"
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        ttk.Label(master, text="Filter:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.filter_combo = ttk.Combobox(
            master,
            values=[
                "Gaussian",
                "Sobel",
                "Prewitt",
                "Scharr",
                "Otsu Threshold",
            ],
            state="readonly",
        )
        self.filter_combo.set("Gaussian")
        self.filter_combo.grid(row=0, column=1, padx=6, pady=6)
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        self.param_label = ttk.Label(master, text="Sigma:")
        self.param_label.grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.param_entry = ttk.Entry(master)
        self.param_entry.insert(0, "2.0")
        self.param_entry.grid(row=1, column=1, padx=6, pady=6)

        self.axis_label = ttk.Label(master, text="Axis:")
        self.axis_combo = ttk.Combobox(
            master,
            values=["Both (Magnitude)", "Horizontal", "Vertical"],
            state="readonly",
        )
        self.axis_combo.set("Both (Magnitude)")

        return self.filter_combo

    def _on_filter_change(self, _event=None) -> None:
        name = self.filter_combo.get()
        if name == "Gaussian":
            self.axis_label.grid_remove()
            self.axis_combo.grid_remove()
            self.param_label.grid(row=1, column=0, sticky="w", padx=6, pady=6)
            self.param_entry.grid(row=1, column=1, padx=6, pady=6)
        elif name in {"Sobel", "Prewitt", "Scharr"}:
            self.param_label.grid_remove()
            self.param_entry.grid_remove()
            self.axis_label.grid(row=1, column=0, sticky="w", padx=6, pady=6)
            self.axis_combo.grid(row=1, column=1, padx=6, pady=6)
        else:
            self.param_label.grid_remove()
            self.param_entry.grid_remove()
            self.axis_label.grid_remove()
            self.axis_combo.grid_remove()

    def validate(self) -> bool:
        name = self.filter_combo.get()
        if name == "Gaussian":
            try:
                val = float(self.param_entry.get())
                if val <= 0:
                    messagebox.showerror("Invalid Parameter", "Sigma must be greater than 0.", parent=self)
                    return False
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid numeric value for Sigma.", parent=self)
                return False
        return True

    def apply(self) -> None:
        name = self.filter_combo.get()
        param = float(self.param_entry.get()) if name == "Gaussian" else 0.0
        axis = self.axis_combo.get() if name in {"Sobel", "Prewitt", "Scharr"} else None
        self.result = (name, param, axis)


class SkimageFiltersTool(ForensicsTool):
    tool_id = "skimage_filters"
    title = "Skimage Filters"
    category = "Set 3"
    description = "Apply filters from skimage.filters (Gaussian, Sobel, Prewitt, Scharr, Otsu Threshold)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        dialog = SkimageFilterDialog(parent)
        if dialog.result is None:
            return None

        name, param, axis_choice = dialog.result
        is_color = document.current.mode not in {"1", "L", "I", "F"}
        details = {"Operation": "Skimage Filter", "Filter": name}

        if name == "Gaussian":
            arr = np.array(document.current.convert("RGB") if is_color else document.current.convert("L"), dtype=np.float64)
            channel_axis = -1 if is_color else None
            res = filters.gaussian(arr, sigma=param, channel_axis=channel_axis, preserve_range=True)
            out_arr = np.clip(res, 0, 255).astype(np.uint8)
            details["Sigma"] = param
        else:
            gray = np.array(document.current.convert("L"), dtype=np.float64) / 255.0
            axis_param = None
            if axis_choice == "Horizontal":
                axis_param = 0
            elif axis_choice == "Vertical":
                axis_param = 1

            if name == "Sobel":
                res = filters.sobel(gray, axis=axis_param)
                out_arr = np.clip(np.abs(res) * 255, 0, 255).astype(np.uint8)
                details["Axis"] = axis_choice or "Both (Magnitude)"
            elif name == "Prewitt":
                res = filters.prewitt(gray, axis=axis_param)
                out_arr = np.clip(np.abs(res) * 255, 0, 255).astype(np.uint8)
                details["Axis"] = axis_choice or "Both (Magnitude)"
            elif name == "Scharr":
                res = filters.scharr(gray, axis=axis_param)
                out_arr = np.clip(np.abs(res) * 255, 0, 255).astype(np.uint8)
                details["Axis"] = axis_choice or "Both (Magnitude)"
            elif name == "Otsu Threshold":
                gray_u8 = np.array(document.current.convert("L"), dtype=np.uint8)
                threshold = filters.threshold_otsu(gray_u8)
                out_arr = ((gray_u8 > threshold) * 255).astype(np.uint8)
                details["Threshold"] = threshold
            else:
                return None

        output = Image.fromarray(out_arr)
        return ToolResult(
            image=output,
            message=f"Applied {name} filter from skimage.filters.",
            details=details,
        )
