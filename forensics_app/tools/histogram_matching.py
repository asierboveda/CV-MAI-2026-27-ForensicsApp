"""Histogram matching tool using scikit-image exposure.match_histograms."""

from __future__ import annotations

import io
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import skimage.exposure as exposure

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def match_image_histograms(
    source_np: np.ndarray, reference_np: np.ndarray
) -> np.ndarray:
    """Match the histogram of the source image to the reference image using scikit-image:
    
    matched_image = exposure.match_histograms(
        source_gray,
        reference_gray
    )
    """
    # If source is RGB and reference is RGB/gray
    matched = exposure.match_histograms(
        source_np,
        reference_np,
        channel_axis=-1 if source_np.ndim == 3 else None,
    )
    return matched.astype(source_np.dtype)


def create_matching_comparison_figure(
    source_np: np.ndarray,
    reference_np: np.ndarray,
    matched_np: np.ndarray,
) -> Image.Image:
    """Generate a 3-column analysis figure showing Source, Reference, and Matched Images and Histograms."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=100)
    fig.patch.set_facecolor("#ffffff")

    # Images (Row 0)
    axes[0, 0].imshow(source_np if source_np.ndim == 3 else source_np, cmap="gray" if source_np.ndim == 2 else None)
    axes[0, 0].set_title("Source Image", fontsize=11, fontweight="bold")

    axes[0, 1].imshow(reference_np if reference_np.ndim == 3 else reference_np, cmap="gray" if reference_np.ndim == 2 else None)
    axes[0, 1].set_title("Reference Image", fontsize=11, fontweight="bold")

    axes[0, 2].imshow(matched_np if matched_np.ndim == 3 else matched_np, cmap="gray" if matched_np.ndim == 2 else None)
    axes[0, 2].set_title("Matched Image", fontsize=11, fontweight="bold")

    for ax in axes[0]:
        ax.axis("off")

    # Histograms (Row 1)
    h_s, _ = np.histogram(source_np.ravel(), bins=256, range=(0, 256))
    axes[1, 0].plot(h_s, color="#1f77b4", linewidth=1.4)
    axes[1, 0].set_title("Source Histogram", fontsize=10)
    axes[1, 0].set_xlim([0, 255])
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    h_r, _ = np.histogram(reference_np.ravel(), bins=256, range=(0, 256))
    axes[1, 1].plot(h_r, color="#2ca02c", linewidth=1.4)
    axes[1, 1].set_title("Reference Histogram", fontsize=10)
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    h_m, _ = np.histogram(matched_np.ravel(), bins=256, range=(0, 256))
    axes[1, 2].plot(h_m, color="#d62728", linewidth=1.4)
    axes[1, 2].set_title("Matched Histogram", fontsize=10)
    axes[1, 2].set_xlim([0, 255])
    axes[1, 2].grid(True, linestyle="--", alpha=0.4)

    for ax in axes[1]:
        ax.tick_params(labelsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


class HistogramMatchingTool(ForensicsTool):
    tool_id = "histogram_matching"
    title = "Histogram matching"
    category = "Histogram & Analysis"
    description = "Match the current image histogram to a reference image distribution."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        curr_img = document.current
        source_np = np.array(curr_img)

        config = self._prompt_dialog(parent)
        if config is None:
            return None

        ref_img, show_comparison, ref_name = config
        
        # Ensure compatible dimensions / channels
        if curr_img.mode in ("L", "1"):
            ref_pil = ref_img.convert("L")
        else:
            ref_pil = ref_img.convert("RGB")
        ref_np = np.array(ref_pil)

        matched_np = match_image_histograms(source_np, ref_np)
        out_img = Image.fromarray(matched_np)

        details: dict[str, Any] = {
            "Reference Source": ref_name,
            "Source Dimensions": f"{source_np.shape[1]} × {source_np.shape[0]}",
            "Reference Dimensions": f"{ref_np.shape[1]} × {ref_np.shape[0]}",
            "Source Mean / Std": f"{source_np.mean():.1f} / {source_np.std():.1f}",
            "Matched Mean / Std": f"{matched_np.mean():.1f} / {matched_np.std():.1f}",
        }

        if show_comparison:
            comp_fig = create_matching_comparison_figure(source_np, ref_np, matched_np)
            msg = f"Matched histogram with {ref_name}. Comparison displayed in side panel."
            return ToolResult(image=out_img, graph=comp_fig, message=msg, details=details)

        msg = f"Matched image histogram to {ref_name}."
        return ToolResult(image=out_img, message=msg, details=details)

    def _prompt_dialog(
        self, parent: tk.Misc
    ) -> tuple[Image.Image, bool, str] | None:
        dialog = tk.Toplevel(parent)
        dialog.title("Histogram Matching")
        dialog.geometry("460x320")
        dialog.resizable(False, False)
        dialog.transient(parent)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Select Reference Image for Histogram Matching",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        sample_dir = Path("forensics_app/images/set2-images")

        ref_img_holder: list[Image.Image | None] = [None]
        ref_name_holder: list[str] = ["No reference image loaded"]
        photo_holder: list[ImageTk.PhotoImage | None] = [None]

        # Preview card
        ref_lf = ttk.LabelFrame(frame, text="Reference Image Target", padding=10)
        ref_lf.pack(fill="x", pady=6)

        preview_lbl = tk.Label(ref_lf, width=70, height=70, bg="#20242b", relief="groove")
        preview_lbl.pack(side="left", padx=(0, 12))

        info_frame = ttk.Frame(ref_lf)
        info_frame.pack(side="left", fill="both", expand=True)

        name_lbl = ttk.Label(info_frame, text="No reference loaded", font=("TkDefaultFont", 9, "bold"))
        name_lbl.pack(anchor="w", pady=(2, 4))
        dim_lbl = ttk.Label(info_frame, text="", foreground="#666666")
        dim_lbl.pack(anchor="w", pady=(0, 4))

        def update_preview(img: Image.Image, name: str) -> None:
            ref_img_holder[0] = img
            ref_name_holder[0] = name
            name_lbl.configure(text=name)
            dim_lbl.configure(text=f"{img.width} × {img.height} px")
            thumb = img.copy()
            thumb.thumbnail((68, 68), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            photo_holder[0] = photo
            preview_lbl.configure(image=photo, width=68, height=68)

        def pick_ref() -> None:
            fn = filedialog.askopenfilename(
                title="Select Reference Image",
                initialdir=str(sample_dir) if sample_dir.exists() else ".",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.webp"), ("All files", "*.*")],
                parent=dialog,
            )
            if fn:
                try:
                    loaded = Image.open(fn)
                    update_preview(loaded, Path(fn).name)
                except Exception as err:
                    tk.messagebox.showerror("Invalid Image", str(err), parent=dialog)

        ttk.Button(info_frame, text="+ Add / Change Reference Image", command=pick_ref).pack(anchor="w")

        # Load a default sample if exists
        for sample_candidate in ["sillas.jpg", "aquatermi_lowcontrast.jpg", "coat.png"]:
            cand_path = sample_dir / sample_candidate
            if cand_path.exists():
                try:
                    update_preview(Image.open(cand_path), cand_path.name)
                    break
                except Exception:
                    pass

        comp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Generate side-by-side comparison in side panel",
            variable=comp_var,
        ).pack(anchor="w", pady=(12, 4))

        result: list[tuple[Image.Image, bool, str] | None] = [None]

        def on_ok() -> None:
            if ref_img_holder[0] is None:
                tk.messagebox.showwarning("Reference Required", "Please select a reference image first.", parent=dialog)
                return
            result[0] = (ref_img_holder[0], comp_var.get(), ref_name_holder[0])
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right")
        ttk.Button(btn_frame, text="Match Histogram", command=on_ok).pack(side="right", padx=(0, 8))

        dialog.wait_window()
        return result[0]
