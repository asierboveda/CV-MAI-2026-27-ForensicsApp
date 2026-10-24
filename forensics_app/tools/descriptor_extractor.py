"""Descriptor extractor (ORB, SIFT) and keypoint visualization for Set 5.2."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, feature
from skimage.feature import SIFT

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def extract_descriptors(
    image: Image.Image,
    method: str = "ORB",
    n_keypoints: int = 200,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """Extract keypoints and descriptors using ORB or SIFT, and visualize keypoints.

    Returns (keypoints, descriptors, visual_image).
    """
    arr = np.array(image.convert("RGB"))
    img_gray = color.rgb2gray(arr)

    method_upper = method.upper()
    if method_upper == "ORB":
        descriptor_extractor = feature.ORB(n_keypoints=n_keypoints)
        try:
            descriptor_extractor.detect_and_extract(img_gray)
            keypoints = descriptor_extractor.keypoints
            descriptors = descriptor_extractor.descriptors
        except RuntimeError:
            # Handle images where ORB finds no corner features without crashing
            keypoints = np.empty((0, 2), dtype=float)
            descriptors = np.empty((0, 256), dtype=bool)
    elif method_upper == "SIFT":
        sift_ref = SIFT(n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min)
        sift_ref.detect_and_extract(img_gray)
        keypoints = sift_ref.keypoints
        descriptors = sift_ref.descriptors
    else:
        raise ValueError(f"Unknown descriptor extractor method: {method}")

    # Visualize keypoints with red circles on grayscale image
    gray_u8 = (img_gray * 255).astype(np.uint8)
    visual = Image.fromarray(gray_u8).convert("RGB")
    draw = ImageDraw.Draw(visual)
    radius = max(3, int(min(visual.size) * 0.006))

    for row, col in keypoints:
        draw.ellipse(
            [(col - radius, row - radius), (col + radius, row + radius)],
            outline="red",
            width=2,
        )

    return keypoints, descriptors, visual


class DescriptorExtractorDialog(simpledialog.Dialog):
    """Dialog for configuring ORB and SIFT descriptor extractors."""

    def __init__(self, parent: tk.Misc, title: str = "Descriptor Extractor") -> None:
        self.method = "ORB"
        self.n_keypoints = 200
        self.n_octaves = 8
        self.n_scales = 3
        self.sigma_min = 1.6
        self.result: dict | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        ttk.Label(master, text="Method:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.method_combo = ttk.Combobox(
            master,
            values=["ORB", "SIFT"],
            state="readonly",
        )
        self.method_combo.set("ORB")
        self.method_combo.grid(row=0, column=1, padx=6, pady=4)
        self.method_combo.bind("<<ComboboxSelected>>", self._on_method_change)

        # ORB parameters
        self.n_keypoints_label = ttk.Label(master, text="Max Keypoints (n_keypoints):")
        self.n_keypoints_label.grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.n_keypoints_entry = ttk.Entry(master)
        self.n_keypoints_entry.insert(0, "200")
        self.n_keypoints_entry.grid(row=1, column=1, padx=6, pady=4)

        # SIFT parameters (initially hidden)
        self.n_octaves_label = ttk.Label(master, text="Octaves (n_octaves):")
        self.n_octaves_entry = ttk.Entry(master)
        self.n_octaves_entry.insert(0, "8")

        self.n_scales_label = ttk.Label(master, text="Scales per octave (n_scales):")
        self.n_scales_entry = ttk.Entry(master)
        self.n_scales_entry.insert(0, "3")

        self.sigma_min_label = ttk.Label(master, text="Initial sigma (sigma_min):")
        self.sigma_min_entry = ttk.Entry(master)
        self.sigma_min_entry.insert(0, "1.6")

        return self.method_combo

    def _on_method_change(self, _event=None) -> None:
        method = self.method_combo.get()
        if method == "ORB":
            self.n_octaves_label.grid_remove()
            self.n_octaves_entry.grid_remove()
            self.n_scales_label.grid_remove()
            self.n_scales_entry.grid_remove()
            self.sigma_min_label.grid_remove()
            self.sigma_min_entry.grid_remove()

            self.n_keypoints_label.grid(row=1, column=0, sticky="w", padx=6, pady=4)
            self.n_keypoints_entry.grid(row=1, column=1, padx=6, pady=4)
        else:
            self.n_keypoints_label.grid_remove()
            self.n_keypoints_entry.grid_remove()

            self.n_octaves_label.grid(row=1, column=0, sticky="w", padx=6, pady=4)
            self.n_octaves_entry.grid(row=1, column=1, padx=6, pady=4)
            self.n_scales_label.grid(row=2, column=0, sticky="w", padx=6, pady=4)
            self.n_scales_entry.grid(row=2, column=1, padx=6, pady=4)
            self.sigma_min_label.grid(row=3, column=0, sticky="w", padx=6, pady=4)
            self.sigma_min_entry.grid(row=3, column=1, padx=6, pady=4)

    def validate(self) -> bool:
        method = self.method_combo.get()
        if method == "ORB":
            try:
                kp = int(self.n_keypoints_entry.get())
                if kp < 1:
                    messagebox.showerror("Invalid Input", "n_keypoints must be >= 1.", parent=self)
                    return False
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid integer for n_keypoints.", parent=self)
                return False
        else:
            try:
                octaves = int(self.n_octaves_entry.get())
                if octaves < 1:
                    messagebox.showerror("Invalid Input", "n_octaves must be >= 1.", parent=self)
                    return False
                scales = int(self.n_scales_entry.get())
                if scales < 1:
                    messagebox.showerror("Invalid Input", "n_scales must be >= 1.", parent=self)
                    return False
                sigma = float(self.sigma_min_entry.get())
                if sigma <= 0:
                    messagebox.showerror("Invalid Input", "sigma_min must be > 0.", parent=self)
                    return False
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values for SIFT parameters.", parent=self)
                return False

        return True

    def apply(self) -> None:
        method = self.method_combo.get()
        self.result = {
            "method": method,
            "n_keypoints": int(self.n_keypoints_entry.get()) if method == "ORB" else 200,
            "n_octaves": int(self.n_octaves_entry.get()) if method == "SIFT" else 8,
            "n_scales": int(self.n_scales_entry.get()) if method == "SIFT" else 3,
            "sigma_min": float(self.sigma_min_entry.get()) if method == "SIFT" else 1.6,
        }


class DescriptorExtractorTool(ForensicsTool):
    tool_id = "descriptor_extractor"
    title = "Descriptor extractor"
    category = "Set 5"
    description = "Extract and visualize keypoints using ORB or SIFT."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        dialog = DescriptorExtractorDialog(parent)
        if dialog.result is None:
            return None

        cfg = dialog.result
        method = cfg["method"]
        keypoints, descriptors, visual = extract_descriptors(
            document.current,
            method=method,
            n_keypoints=cfg["n_keypoints"],
            n_octaves=cfg["n_octaves"],
            n_scales=cfg["n_scales"],
            sigma_min=cfg["sigma_min"],
        )

        details = {
            "Operation": "Descriptor extractor",
            "Method": method,
            "Keypoints detected": len(keypoints),
            "Descriptor shape": str(descriptors.shape if descriptors is not None else (0,)),
        }
        if method == "ORB":
            details["Max keypoints (n_keypoints)"] = cfg["n_keypoints"]
        else:
            details["Octaves (n_octaves)"] = cfg["n_octaves"]
            details["Scales per octave (n_scales)"] = cfg["n_scales"]
            details["Initial sigma (sigma_min)"] = cfg["sigma_min"]

        return ToolResult(
            image=visual,
            message=f"Extracted {len(keypoints)} keypoints using {method}.",
            details=details,
        )
