"""Histogram visualization tool using skimage.exposure.histogram with side panel display."""

from __future__ import annotations

import io
import tkinter as tk
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage.exposure as exposure
import skimage.measure

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def compute_channel_histogram(channel_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 256-bin histogram for a channel using skimage.exposure.histogram."""
    return exposure.histogram(channel_np, nbins=256)


def plot_skimage_histogram(
    image_np: np.ndarray,
    show_r: bool = True,
    show_g: bool = True,
    show_b: bool = True,
    title: str = "Intensity Histogram",
) -> Image.Image:
    """Plot intensity histogram using skimage.exposure.histogram.
    
    If color image (RGB), plots Red, Green, Blue curves according to show_r, show_g, show_b.
    If grayscale image, plots single intensity curve.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=100)
    fig.patch.set_facecolor("#ffffff")

    is_color = (image_np.ndim == 3 and image_np.shape[2] >= 3)

    if is_color:
        channels_info = [
            (0, "#d62728", "Red", show_r),
            (1, "#2ca02c", "Green", show_g),
            (2, "#1f77b4", "Blue", show_b),
        ]
        plotted_any = False
        for ch_idx, col, lbl, enabled in channels_info:
            if enabled:
                ch_data = image_np[:, :, ch_idx]
                hist, centers = exposure.histogram(ch_data, nbins=256)
                ax.plot(centers, hist, color=col, label=lbl, linewidth=1.5)
                plotted_any = True

        if plotted_any:
            ax.legend(frameon=True, fontsize=9, loc="upper right")
        else:
            ax.text(0.5, 0.5, "No channels selected", transform=ax.transAxes, ha="center", va="center", color="#777")
    else:
        gray = image_np if image_np.ndim == 2 else image_np[:, :, 0]
        hist, centers = exposure.histogram(gray, nbins=256)
        ax.plot(centers, hist, color="#222222", label="Intensity", linewidth=1.5)
        ax.legend(frameon=True, fontsize=9, loc="upper right")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Intensity (0 – 255)", fontsize=9)
    ax.set_ylabel("Pixel Count", fontsize=9)
    ax.set_xlim([0, 255])
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.tick_params(labelsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram visualization"
    category = "Histogram & Analysis"
    description = "Compute and plot intensity distributions in side panel using scikit-image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        curr_img = document.current
        is_color = curr_img.mode not in ("L", "1")
        if is_color:
            img_np = np.array(curr_img.convert("RGB"))
        else:
            img_np = np.array(curr_img.convert("L"))

        stem = document.path.stem if document.path else "Image"
        title_str = f"{stem} ({'RGB' if is_color else 'Grayscale'} Histogram)"

        # Generate plot using scikit-image exposure.histogram
        out_img = plot_skimage_histogram(img_np, show_r=True, show_g=True, show_b=True, title=title_str)

        # Statistics
        if is_color:
            r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
            gray = np.array(curr_img.convert("L"))
            entropy_val = skimage.measure.shannon_entropy(gray)
            details: dict[str, Any] = {
                "Image Type": "Color (RGB)",
                "Mean Intensity (R/G/B)": f"{r.mean():.1f} / {g.mean():.1f} / {b.mean():.1f}",
                "Std Deviation (R/G/B)": f"{r.std():.1f} / {g.std():.1f} / {b.std():.1f}",
                "Shannon Entropy": f"{entropy_val:.3f} bits",
            }
        else:
            entropy_val = skimage.measure.shannon_entropy(img_np)
            details = {
                "Image Type": "Grayscale",
                "Mean Intensity": f"{img_np.mean():.1f}",
                "Std Deviation": f"{img_np.std():.1f}",
                "Intensity Min / Max": f"{img_np.min()} / {img_np.max()}",
                "Shannon Entropy": f"{entropy_val:.3f} bits",
            }

        msg = f"Generated {details['Image Type']} histogram in side panel using skimage."
        return ToolResult(image=None, graph=out_img, message=msg, details=details)

