"""Corner detection using skimage.feature (FAST, Harris) with configurable peak threshold."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def detect_corners(
    image: Image.Image,
    method: str = "FAST",
    peak_threshold_rel: float = 0.1,
    min_distance: int = 5,
    fast_threshold: float = 0.15,
    fast_n: int = 9,
    harris_k: float = 0.05,
    harris_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """Detect corners using FAST or Harris response and corner_peaks.

    Returns (corner_points, response_map, visual_image).
    """
    arr = np.array(image.convert("RGB"))
    img_gray = color.rgb2gray(arr)

    method_upper = method.upper()
    if method_upper == "FAST":
        response = feature.corner_fast(img_gray, n=fast_n, threshold=fast_threshold)
    elif method_upper == "HARRIS":
        response = feature.corner_harris(img_gray, k=harris_k, sigma=harris_sigma)
    else:
        raise ValueError(f"Unknown corner detection method: {method}")

    corner_points = feature.corner_peaks(
        response,
        min_distance=min_distance,
        threshold_rel=peak_threshold_rel,
    )

    # Visualize: grayscale background with red open circles (facecolors='none', edgecolors='r')
    gray_u8 = (img_gray * 255).astype(np.uint8)
    visual = Image.fromarray(gray_u8).convert("RGB")
    draw = ImageDraw.Draw(visual)
    radius = max(3, int(min(visual.size) * 0.006))

    for row, col in corner_points:
        draw.ellipse(
            [(col - radius, row - radius), (col + radius, row + radius)],
            outline="red",
            width=2,
        )

    return corner_points, response, visual


class CornerDetectionDialog(simpledialog.Dialog):
    """Dialog for configuring FAST and Harris corner detection."""

    def __init__(self, parent: tk.Misc, title: str = "Corner Detection") -> None:
        self.method = "FAST"
        self.peak_threshold_rel = 0.1
        self.min_distance = 5
        self.fast_threshold = 0.15
        self.fast_n = 9
        self.harris_k = 0.05
        self.harris_sigma = 1.0
        self.result: dict | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        ttk.Label(master, text="Method:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.method_combo = ttk.Combobox(
            master,
            values=["FAST", "Harris"],
            state="readonly",
        )
        self.method_combo.set("FAST")
        self.method_combo.grid(row=0, column=1, padx=6, pady=4)
        self.method_combo.bind("<<ComboboxSelected>>", self._on_method_change)

        ttk.Label(master, text="Peak Threshold (rel):").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.peak_thresh_entry = ttk.Entry(master)
        self.peak_thresh_entry.insert(0, "0.1")
        self.peak_thresh_entry.grid(row=1, column=1, padx=6, pady=4)

        ttk.Label(master, text="Min Peak Distance (px):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.min_dist_entry = ttk.Entry(master)
        self.min_dist_entry.insert(0, "5")
        self.min_dist_entry.grid(row=2, column=1, padx=6, pady=4)

        # FAST specific widgets
        self.fast_thresh_label = ttk.Label(master, text="FAST Threshold:")
        self.fast_thresh_label.grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.fast_thresh_entry = ttk.Entry(master)
        self.fast_thresh_entry.insert(0, "0.15")
        self.fast_thresh_entry.grid(row=3, column=1, padx=6, pady=4)

        self.fast_n_label = ttk.Label(master, text="FAST n (circle points):")
        self.fast_n_label.grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self.fast_n_entry = ttk.Entry(master)
        self.fast_n_entry.insert(0, "9")
        self.fast_n_entry.grid(row=4, column=1, padx=6, pady=4)

        # Harris specific widgets (initially hidden)
        self.harris_k_label = ttk.Label(master, text="Harris k:")
        self.harris_k_entry = ttk.Entry(master)
        self.harris_k_entry.insert(0, "0.05")

        self.harris_sigma_label = ttk.Label(master, text="Harris Sigma:")
        self.harris_sigma_entry = ttk.Entry(master)
        self.harris_sigma_entry.insert(0, "1.0")

        return self.method_combo

    def _on_method_change(self, _event=None) -> None:
        method = self.method_combo.get()
        if method == "FAST":
            self.harris_k_label.grid_remove()
            self.harris_k_entry.grid_remove()
            self.harris_sigma_label.grid_remove()
            self.harris_sigma_entry.grid_remove()

            self.fast_thresh_label.grid(row=3, column=0, sticky="w", padx=6, pady=4)
            self.fast_thresh_entry.grid(row=3, column=1, padx=6, pady=4)
            self.fast_n_label.grid(row=4, column=0, sticky="w", padx=6, pady=4)
            self.fast_n_entry.grid(row=4, column=1, padx=6, pady=4)
        else:
            self.fast_thresh_label.grid_remove()
            self.fast_thresh_entry.grid_remove()
            self.fast_n_label.grid_remove()
            self.fast_n_entry.grid_remove()

            self.harris_k_label.grid(row=3, column=0, sticky="w", padx=6, pady=4)
            self.harris_k_entry.grid(row=3, column=1, padx=6, pady=4)
            self.harris_sigma_label.grid(row=4, column=0, sticky="w", padx=6, pady=4)
            self.harris_sigma_entry.grid(row=4, column=1, padx=6, pady=4)

    def validate(self) -> bool:
        try:
            p_rel = float(self.peak_thresh_entry.get())
            if not 0.0 <= p_rel <= 1.0:
                messagebox.showerror("Invalid Input", "Peak threshold (rel) must be between 0.0 and 1.0.", parent=self)
                return False
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric value for Peak Threshold.", parent=self)
            return False

        try:
            m_dist = int(self.min_dist_entry.get())
            if m_dist < 1:
                messagebox.showerror("Invalid Input", "Min distance must be at least 1.", parent=self)
                return False
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for Min Distance.", parent=self)
            return False

        method = self.method_combo.get()
        if method == "FAST":
            try:
                f_thresh = float(self.fast_thresh_entry.get())
                if f_thresh <= 0:
                    messagebox.showerror("Invalid Input", "FAST threshold must be > 0.", parent=self)
                    return False
                f_n = int(self.fast_n_entry.get())
                if not (1 <= f_n <= 16):
                    messagebox.showerror("Invalid Input", "FAST n must be between 1 and 16.", parent=self)
                    return False
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for FAST parameters.", parent=self)
                return False
        else:
            try:
                h_k = float(self.harris_k_entry.get())
                if not (0.0 < h_k < 1.0):
                    messagebox.showerror("Invalid Input", "Harris k must be between 0.0 and 1.0.", parent=self)
                    return False
                h_sigma = float(self.harris_sigma_entry.get())
                if h_sigma <= 0:
                    messagebox.showerror("Invalid Input", "Harris sigma must be > 0.", parent=self)
                    return False
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for Harris parameters.", parent=self)
                return False

        return True

    def apply(self) -> None:
        method = self.method_combo.get()
        self.result = {
            "method": method,
            "peak_threshold_rel": float(self.peak_thresh_entry.get()),
            "min_distance": int(self.min_dist_entry.get()),
            "fast_threshold": float(self.fast_thresh_entry.get()) if method == "FAST" else 0.15,
            "fast_n": int(self.fast_n_entry.get()) if method == "FAST" else 12,
            "harris_k": float(self.harris_k_entry.get()) if method == "Harris" else 0.05,
            "harris_sigma": float(self.harris_sigma_entry.get()) if method == "Harris" else 1.0,
        }


class CornerDetectionTool(ForensicsTool):
    tool_id = "corner_detection"
    title = "Corner detection"
    category = "Set 5"
    description = "Detect corners with FAST or Harris and configurable peak threshold."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        dialog = CornerDetectionDialog(parent)
        if dialog.result is None:
            return None

        cfg = dialog.result
        method = cfg["method"]
        corner_points, _, visual = detect_corners(
            document.current,
            method=method,
            peak_threshold_rel=cfg["peak_threshold_rel"],
            min_distance=cfg["min_distance"],
            fast_threshold=cfg["fast_threshold"],
            fast_n=cfg["fast_n"],
            harris_k=cfg["harris_k"],
            harris_sigma=cfg["harris_sigma"],
        )

        details = {
            "Operation": "Corner detection",
            "Method": method,
            "Corners detected": len(corner_points),
            "Peak threshold (rel)": cfg["peak_threshold_rel"],
            "Min peak distance": cfg["min_distance"],
        }
        if method == "FAST":
            details["FAST threshold"] = cfg["fast_threshold"]
            details["FAST n"] = cfg["fast_n"]
        else:
            details["Harris k"] = cfg["harris_k"]
            details["Harris sigma"] = cfg["harris_sigma"]

        return ToolResult(
            image=visual,
            message=f"Detected {len(corner_points)} corners using {method}.",
            details=details,
        )
