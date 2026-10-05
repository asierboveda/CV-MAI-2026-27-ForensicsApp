"""Channel swap feature: swap color channels with dynamic output channel exclusion."""

from __future__ import annotations

import io
import tkinter as tk
from tkinter import ttk
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


CHANNEL_MAP = [("Red", 0), ("Green", 1), ("Blue", 2)]


def swap_channels(image_np: np.ndarray, ch_a: int, ch_b: int) -> np.ndarray:
    """Swap two color channels in an RGB image array using .copy() to preserve values."""
    if image_np.ndim == 2 or image_np.shape[2] < 3:
        # If grayscale or no 3rd channel, return as-is
        return image_np.copy()

    swapped = image_np.copy()
    tmp = swapped[:, :, ch_a].copy()
    swapped[:, :, ch_a] = swapped[:, :, ch_b]
    swapped[:, :, ch_b] = tmp
    return swapped


def create_swap_comparison(
    original_np: np.ndarray,
    swapped_np: np.ndarray,
    title_orig: str = "Original image",
    title_trans: str = "Transformed image",
) -> Image.Image:
    """Generate a side-by-side comparison figure matching the coursework style."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor("#ffffff")

    axes[0].imshow(original_np[:, :, :3] if original_np.ndim == 3 else original_np, cmap="gray" if original_np.ndim == 2 else None)
    axes[0].set_title(title_orig, fontsize=12, fontweight="bold")

    axes[1].imshow(swapped_np[:, :, :3] if swapped_np.ndim == 3 else swapped_np, cmap="gray" if swapped_np.ndim == 2 else None)
    axes[1].set_title(title_trans, fontsize=12, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Channel swap"
    category = "Color & Channels"
    description = "Swap two distinct color channels in the current image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        curr_rgb = document.current.convert("RGB")
        img_np = np.array(curr_rgb)

        dialog_res = self._prompt_swap_dialog(parent)
        if dialog_res is None:
            return None

        ch_origin, ch_dest = dialog_res

        ch_names_short = ["Red", "Green", "Blue"]
        orig_name = ch_names_short[ch_origin]
        dest_name = ch_names_short[ch_dest]

        swapped_np = swap_channels(img_np, ch_origin, ch_dest)

        details: dict[str, Any] = {
            "Origin Channel": orig_name,
            "Target Channel": dest_name,
            "Swap Operation": f"{orig_name} ↔ {dest_name}",
            "Image Dimensions": f"{img_np.shape[1]} × {img_np.shape[0]}",
        }

        out_img = Image.fromarray(swapped_np)
        msg = f"Swapped {orig_name} and {dest_name} channels."
        details["View"] = "Transformed Image"
        return ToolResult(image=out_img, message=msg, details=details)

    def _prompt_swap_dialog(
        self, parent: tk.Misc
    ) -> tuple[int, int] | None:
        dialog = tk.Toplevel(parent)
        dialog.title("Channel Swap")
        dialog.geometry("360x190")
        dialog.resizable(False, False)
        dialog.transient(parent)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Select Channels to Swap:",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        all_channels = [("Red (0)", 0), ("Green (1)", 1), ("Blue (2)", 2)]

        # Origin channel dropdown
        origin_frame = ttk.Frame(frame)
        origin_frame.pack(fill="x", pady=4)
        ttk.Label(origin_frame, text="Origin Channel:", width=16).pack(side="left")
        origin_cb = ttk.Combobox(
            origin_frame, values=[name for name, _ in all_channels], state="readonly", width=18
        )
        origin_cb.current(0)  # Red default
        origin_cb.pack(side="left", fill="x", expand=True)

        # Output/Target channel dropdown (dynamically excludes origin)
        dest_frame = ttk.Frame(frame)
        dest_frame.pack(fill="x", pady=4)
        ttk.Label(dest_frame, text="Output Channel:", width=16).pack(side="left")
        dest_cb = ttk.Combobox(
            dest_frame, state="readonly", width=18
        )
        dest_cb.pack(side="left", fill="x", expand=True)

        # Map current dest combobox indices to actual channel IDs
        current_dest_ids: list[int] = []

        def update_dest_options(_event: tk.Event | None = None) -> None:
            nonlocal current_dest_ids
            origin_idx = origin_cb.current()
            if origin_idx < 0:
                origin_idx = 0
            
            # Available destinations: everything except selected origin
            available = [(name, cid) for name, cid in all_channels if cid != origin_idx]
            current_dest_ids = [cid for _, cid in available]
            dest_cb.configure(values=[name for name, _ in available])
            dest_cb.current(0)

        origin_cb.bind("<<ComboboxSelected>>", update_dest_options)
        update_dest_options()

        result: list[tuple[int, int] | None] = [None]

        def on_ok() -> None:
            ch_a = origin_cb.current()
            dest_select_idx = dest_cb.current()
            if dest_select_idx < 0 or dest_select_idx >= len(current_dest_ids):
                return
            ch_b = current_dest_ids[dest_select_idx]
            result[0] = (ch_a, ch_b)
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right")
        ttk.Button(btn_frame, text="Swap", command=on_ok).pack(side="right", padx=(0, 8))

        dialog.wait_window()
        return result[0]

