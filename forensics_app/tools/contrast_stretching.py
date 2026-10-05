"""Contrast stretching and exposure adjustment tools based on Set 2.5."""

from __future__ import annotations

import tkinter as tk
from typing import Any

import numpy as np
from PIL import Image
import skimage.color as skcolor
import skimage.exposure as exposure
from skimage.util import img_as_float, img_as_ubyte

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def simulate_low_contrast(image_np: np.ndarray) -> np.ndarray:
    """Artificially create a low-contrast image using scikit-image:
    
    low_contrast = exposure.rescale_intensity(
        image,
        in_range=(0, 1),
        out_range=(0.3, 0.7)
    )
    """
    img_float = img_as_float(image_np)
    low_contrast = exposure.rescale_intensity(
        img_float,
        in_range=(0.0, 1.0),
        out_range=(0.3, 0.7),
    )
    return img_as_ubyte(np.clip(low_contrast, 0.0, 1.0))


def apply_percentile_contrast_stretching(image_np: np.ndarray) -> np.ndarray:
    """1. Contrast stretching using 2nd and 98th percentiles:
    
    p2, p98 = np.percentile(low_contrast, (2, 98))
    stretched = exposure.rescale_intensity(
        low_contrast,
        in_range=(p2, p98),
        out_range=(0, 1)
    )
    """
    img_float = img_as_float(image_np)
    p2, p98 = np.percentile(img_float, (2.0, 98.0))
    stretched = exposure.rescale_intensity(
        img_float,
        in_range=(p2, p98),
        out_range=(0.0, 1.0),
    )
    return img_as_ubyte(np.clip(stretched, 0.0, 1.0))


def apply_histogram_equalization(image_np: np.ndarray) -> np.ndarray:
    """2. Histogram equalization using scikit-image:
    
    equalized = exposure.equalize_hist(low_contrast)
    For color images, performs equalization on the luminance (L) channel in LAB space.
    """
    img_float = img_as_float(image_np)
    if img_float.ndim == 2:
        eq = exposure.equalize_hist(img_float)
        return img_as_ubyte(np.clip(eq, 0.0, 1.0))

    # Color image: LAB color space luminance equalization
    lab = skcolor.rgb2lab(img_float[:, :, :3])
    L = lab[:, :, 0] / 100.0  # L* is [0, 100]
    L_eq = exposure.equalize_hist(L)
    lab_eq = lab.copy()
    lab_eq[:, :, 0] = L_eq * 100.0
    lum_eq = np.clip(skcolor.lab2rgb(lab_eq), 0.0, 1.0)
    return img_as_ubyte(lum_eq)


def apply_adaptive_equalization(image_np: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """3. Adaptive histogram equalization (CLAHE):
    
    adaptive = exposure.equalize_adapthist(
        low_contrast,
        clip_limit=0.03
    )
    """
    img_float = img_as_float(image_np)
    if img_float.ndim == 3 and img_float.shape[2] > 3:
        img_float = img_float[:, :, :3]
    adaptive = exposure.equalize_adapthist(img_float, clip_limit=clip_limit)
    return img_as_ubyte(np.clip(adaptive, 0.0, 1.0))


class LowContrastSimTool(ForensicsTool):
    tool_id = "low_contrast_sim"
    title = "Simulate low contrast"
    category = "Contrast stretching"
    description = "Rescale intensity to range (0.3, 0.7) to create an artificial low-contrast image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        curr_img = document.current
        img_np = np.array(curr_img)
        low_np = simulate_low_contrast(img_np)
        out_img = Image.fromarray(low_np)

        details: dict[str, Any] = {
            "Technique": "Artificially create low-contrast image",
            "Input Range": "(0.0, 1.0)",
            "Output Range": "(0.3, 0.7)",
            "Input Std (Contrast)": f"{img_np.std():.2f}",
            "Output Std (Contrast)": f"{low_np.std():.2f}",
        }

        return ToolResult(
            image=out_img,
            message="Created artificial low-contrast image (range 0.3 – 0.7).",
            details=details,
        )


class ContrastStretchingPercentileTool(ForensicsTool):
    tool_id = "contrast_stretching_percentile"
    title = "Contrast stretching (2%–98%)"
    category = "Contrast stretching"
    description = "Stretch contrast between 2nd and 98th percentiles using exposure.rescale_intensity."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        curr_img = document.current
        img_np = np.array(curr_img)
        p2, p98 = np.percentile(img_np, (2.0, 98.0))
        stretched_np = apply_percentile_contrast_stretching(img_np)
        out_img = Image.fromarray(stretched_np)

        gain = float(stretched_np.std()) / (float(img_np.std()) + 1e-6)

        details: dict[str, Any] = {
            "Technique": "1. Percentile Contrast Stretching",
            "p2 (2nd percentile)": f"{p2:.1f}",
            "p98 (98th percentile)": f"{p98:.1f}",
            "Original Dynamic Range": f"{img_np.min()} – {img_np.max()}",
            "Stretched Dynamic Range": f"{stretched_np.min()} – {stretched_np.max()}",
            "Contrast Gain": f"{gain:.2f}×",
        }

        return ToolResult(
            image=out_img,
            message=f"Applied 2%–98% percentile contrast stretching (Gain: {gain:.2f}×).",
            details=details,
        )


class HistogramEqualizationTool(ForensicsTool):
    tool_id = "histogram_equalization"
    title = "Histogram equalization"
    category = "Contrast stretching"
    description = "Global histogram equalization using exposure.equalize_hist (LAB color preservation)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        curr_img = document.current
        img_np = np.array(curr_img)
        eq_np = apply_histogram_equalization(img_np)
        out_img = Image.fromarray(eq_np)

        gain = float(eq_np.std()) / (float(img_np.std()) + 1e-6)

        details: dict[str, Any] = {
            "Technique": "2. Global Histogram Equalization",
            "Color Space": "LAB (Luminance Equalization)" if img_np.ndim == 3 else "Grayscale",
            "Input Contrast (Std)": f"{img_np.std():.2f}",
            "Output Contrast (Std)": f"{eq_np.std():.2f}",
            "Contrast Gain": f"{gain:.2f}×",
        }

        return ToolResult(
            image=out_img,
            message="Applied global histogram equalization.",
            details=details,
        )


class AdaptiveEqualizationTool(ForensicsTool):
    tool_id = "adaptive_equalization"
    title = "Adaptive equalization (CLAHE)"
    category = "Contrast stretching"
    description = "Adaptive histogram equalization (CLAHE) with clip_limit=0.03."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        curr_img = document.current
        img_np = np.array(curr_img)
        clahe_np = apply_adaptive_equalization(img_np, clip_limit=0.03)
        out_img = Image.fromarray(clahe_np)

        gain = float(clahe_np.std()) / (float(img_np.std()) + 1e-6)

        details: dict[str, Any] = {
            "Technique": "3. Adaptive Histogram Equalization (CLAHE)",
            "Clip Limit": "0.03",
            "Input Contrast (Std)": f"{img_np.std():.2f}",
            "Output Contrast (Std)": f"{clahe_np.std():.2f}",
            "Contrast Gain": f"{gain:.2f}×",
        }

        return ToolResult(
            image=out_img,
            message="Applied CLAHE adaptive histogram equalization (clip_limit=0.03).",
            details=details,
        )


# Backward compatibility alias
def stretch_contrast(
    image_np: np.ndarray,
    method: str = "percentile",
    p_low: float = 2.0,
    p_high: float = 98.0,
    clip_limit: float = 0.03,
) -> np.ndarray:
    if method == "percentile":
        return apply_percentile_contrast_stretching(image_np)
    elif method == "equalize":
        return apply_histogram_equalization(image_np)
    elif method == "clahe":
        return apply_adaptive_equalization(image_np, clip_limit=clip_limit)
    elif method == "low_contrast":
        return simulate_low_contrast(image_np)
    else:
        return apply_percentile_contrast_stretching(image_np)

