"""Channel split feature: split RGB into individual Red, Green, and Blue channels."""

from __future__ import annotations

import io
import tkinter as tk
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def split_rgb_channels(image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an RGB image array into Red, Green, and Blue 2D grayscale arrays."""
    if image_np.ndim == 2:
        # Grayscale image: all channels are identical
        return image_np.copy(), image_np.copy(), image_np.copy()
    if image_np.shape[2] >= 3:
        img1 = image_np[:, :, 0]
        img2 = image_np[:, :, 1]
        img3 = image_np[:, :, 2]
        return img1, img2, img3
    raise ValueError(f"Unsupported image shape: {image_np.shape}")


def combine_rgb_channels(
    img1: np.ndarray, img2: np.ndarray, img3: np.ndarray
) -> np.ndarray:
    """Combine three separate grayscale channel arrays into an RGB image array."""
    h, w = img1.shape[:2]
    img_cl = np.zeros((h, w, 3), dtype=np.uint8)
    img_cl[:, :, 0] = img1
    img_cl[:, :, 1] = img2
    img_cl[:, :, 2] = img3
    return img_cl


def create_3channel_figure(
    img1: np.ndarray, img2: np.ndarray, img3: np.ndarray, title: str = "Channel Split"
) -> Image.Image:
    """Generate a 3-panel grayscale visualization matching the coursework code:
    
    f, axarr = plt.subplots(ncols=3, nrows=1, figsize=(10, 10))
    axarr[0].imshow(img1, cmap='gray')
    axarr[1].imshow(img2, cmap='gray')
    axarr[2].imshow(img3, cmap='gray')
    """
    fig, axarr = plt.subplots(ncols=3, nrows=1, figsize=(12, 4.5), dpi=100)
    fig.patch.set_facecolor("#ffffff")

    axarr[0].imshow(img1, cmap="gray")
    axarr[0].set_title("Red channel", fontsize=11, fontweight="bold")
    axarr[0].axis("off")

    axarr[1].imshow(img2, cmap="gray")
    axarr[1].set_title("Green channel", fontsize=11, fontweight="bold")
    axarr[1].axis("off")

    axarr[2].imshow(img3, cmap="gray")
    axarr[2].set_title("Blue channel", fontsize=11, fontweight="bold")
    axarr[2].axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def create_channel_comparison_grid(
    image_np: np.ndarray, title: str = "Image"
) -> Image.Image:
    """Generate a 4-panel figure: Original, Red channel, Green channel, Blue channel."""
    r, g, b = split_rgb_channels(image_np)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=100)
    fig.patch.set_facecolor("#ffffff")

    if image_np.ndim == 2:
        axes[0].imshow(image_np, cmap="gray")
    else:
        axes[0].imshow(image_np[:, :, :3])
    axes[0].set_title(f"{title} image", fontsize=11, fontweight="bold")

    axes[1].imshow(r, cmap="gray")
    axes[1].set_title("Red channel", fontsize=11, fontweight="bold")

    axes[2].imshow(g, cmap="gray")
    axes[2].set_title("Green channel", fontsize=11, fontweight="bold")

    axes[3].imshow(b, cmap="gray")
    axes[3].set_title("Blue channel", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def extract_single_channel(
    image_np: np.ndarray, channel_idx: int, colorized: bool = False
) -> np.ndarray:
    """Extract a single channel (0=Red, 1=Green, 2=Blue) either as grayscale or colorized."""
    if image_np.ndim == 2:
        return image_np.copy()

    single_ch = image_np[:, :, channel_idx]
    if not colorized:
        return single_ch

    out = np.zeros_like(image_np[:, :, :3])
    out[:, :, channel_idx] = single_ch
    return out


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel split"
    category = "Color & Channels"
    description = "Split the current image into 3 separate grayscale channels (Red, Green, Blue)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # Convert working image to RGB numpy array
        curr_rgb = document.current.convert("RGB")
        img_np = np.array(curr_rgb)
        img1, img2, img3 = split_rgb_channels(img_np)

        stem = document.path.stem if document.path else "Image"
        graph_img = create_3channel_figure(img1, img2, img3, title=stem)

        details: dict[str, Any] = {
            "Red mean / std": f"{img1.mean():.1f} / {img1.std():.1f}",
            "Green mean / std": f"{img2.mean():.1f} / {img2.std():.1f}",
            "Blue mean / std": f"{img3.mean():.1f} / {img3.std():.1f}",
            "R min / max": f"{img1.min()} / {img1.max()}",
            "G min / max": f"{img2.min()} / {img2.max()}",
            "B min / max": f"{img3.min()} / {img3.max()}",
        }

        return ToolResult(
            image=None,
            graph=graph_img,
            message="Split image into 3 channels (Red, Green, Blue). Displayed in side panel.",
            details=details,
        )
