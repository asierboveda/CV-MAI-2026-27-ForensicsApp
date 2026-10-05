"""Masking and Texture Transfer feature with visual image previews."""

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

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def apply_binary_threshold_mask(
    image_np: np.ndarray, threshold: int = 135
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply binary threshold masking matching the coursework code:
    
    # Mask: keep brighter pixels
    mask = image > 135
    # Apply mask
    masked1 = image * mask
    # Apply mask 2
    masked2 = np.zeros_like(image)
    masked2[~mask] = image[~mask]
    """
    if image_np.ndim == 3:
        # For RGB, compute mask across channels or luminance
        gray = np.mean(image_np[:, :, :3], axis=-1)
        mask = gray > threshold
        mask_3d = mask[:, :, np.newaxis]

        masked1 = (image_np[:, :, :3] * mask_3d).astype(np.uint8)
        masked2 = np.zeros_like(image_np[:, :, :3])
        masked2[~mask] = image_np[:, :, :3][~mask]
        return mask, masked1, masked2
    else:
        mask = image_np > threshold
        masked1 = (image_np * mask).astype(np.uint8)
        masked2 = np.zeros_like(image_np)
        masked2[~mask] = image_np[~mask]
        return mask, masked1, masked2


def extract_binary_mask(mask_img: Image.Image, threshold: int = 135) -> np.ndarray:
    """Extract a 2D boolean mask from an image for pixels with intensity > threshold (default 135)."""
    mask_np = np.array(mask_img)
    if mask_np.ndim == 3 and mask_np.shape[2] == 4:
        # RGBA: check alpha first if transparent
        alpha = mask_np[:, :, 3]
        if np.any(alpha < 250):
            return alpha > threshold
    # RGB / Grayscale: identify pixels with any intensity > threshold
    rgb = mask_np[:, :, :3] if mask_np.ndim == 3 else mask_np
    if rgb.ndim == 3:
        return np.any(rgb > threshold, axis=-1)
    return rgb > threshold


def composite_texture_or_overlay(
    base_np: np.ndarray,
    mask_np_rgb: np.ndarray,
    texture_np: np.ndarray | None = None,
    mode: str = "tiled",
    preserve_shading: bool = True,
    threshold: int = 135,
) -> np.ndarray:
    """Composite a mask or texture onto a base image for pixels > threshold (default 135).
    
    Modes:
    - "overlay": Use the mask's original RGB pixels (e.g. yellow coat).
    - "tiled": Tile the texture repeating across the image dimensions.
    - "resized": Resize the texture to fit the image dimensions.
    """
    h, w = base_np.shape[:2]
    binary_mask = extract_binary_mask(Image.fromarray(mask_np_rgb), threshold=threshold)

    # Resize binary mask if dimensions differ from base
    if binary_mask.shape[:2] != (h, w):
        mask_pil = Image.fromarray(binary_mask.astype(np.uint8) * 255).resize(
            (w, h), Image.Resampling.NEAREST
        )
        binary_mask = np.array(mask_pil) > 127

    output = base_np.copy()

    if texture_np is None or mode == "overlay":
        # Direct mask color overlay
        mask_resized = np.array(
            Image.fromarray(mask_np_rgb[:, :, :3]).resize((w, h), Image.Resampling.BILINEAR)
        )
        output[binary_mask] = mask_resized[binary_mask]
        return output

    # Handle texture
    th, tw = texture_np.shape[:2]
    if mode == "tiled":
        reps_y = int(np.ceil(h / th))
        reps_x = int(np.ceil(w / tw))
        pattern = np.tile(texture_np[:, :, :3], (reps_y, reps_x, 1))[:h, :w]
    else:  # "resized"
        pattern = np.array(
            Image.fromarray(texture_np[:, :, :3]).resize((w, h), Image.Resampling.BILINEAR)
        )

    if preserve_shading and np.any(binary_mask):
        mask_resized = np.array(
            Image.fromarray(mask_np_rgb[:, :, :3]).resize((w, h), Image.Resampling.BILINEAR)
        )
        shading = mask_resized.mean(axis=-1, keepdims=True) / 255.0
        norm_factor = shading[binary_mask].mean() + 1e-6
        shading_norm = np.clip(shading / norm_factor, 0.2, 1.8)
        pattern = np.clip(pattern.astype(np.float32) * shading_norm, 0, 255).astype(np.uint8)

    output[binary_mask] = pattern[binary_mask]
    return output


def create_masking_comparison_strip(
    coat_img: Image.Image,
    model_img: Image.Image,
    with_coat_img: Image.Image,
    with_texture_img: Image.Image,
) -> Image.Image:
    """Generate the 4-panel demonstration strip matching course slides."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=100)
    fig.patch.set_facecolor("#ffffff")

    axes[0].imshow(coat_img)
    axes[0].set_title("Coat", fontsize=11, fontweight="bold")

    axes[1].imshow(model_img)
    axes[1].set_title("Model", fontsize=11, fontweight="bold")

    axes[2].imshow(with_coat_img)
    axes[2].set_title("Model with Coat", fontsize=11, fontweight="bold")

    axes[3].imshow(with_texture_img)
    axes[3].set_title("Model with Textured Coat", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def create_coins_mask_comparison(
    image: np.ndarray, mask: np.ndarray, masked1: np.ndarray, masked2: np.ndarray
) -> Image.Image:
    """Generate the 4-panel binary masking demonstration matching the course coins example."""
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.8), dpi=100)
    fig.patch.set_facecolor("#ffffff")

    ax[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax[0].set_title("Original image", fontsize=10, fontweight="bold")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Binary mask", fontsize=10, fontweight="bold")

    ax[2].imshow(masked1, cmap="gray" if masked1.ndim == 2 else None)
    ax[2].set_title("Masked image 1 (>135)", fontsize=10, fontweight="bold")

    ax[3].imshow(masked2, cmap="gray" if masked2.ndim == 2 else None)
    ax[3].set_title("Masked image 2 (≤135)", fontsize=10, fontweight="bold")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking & Texture Transfer"
    category = "Masking & Compositing"
    description = "Apply image masks, threshold filtering, or transfer textures onto masked regions."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        base_img = document.current.convert("RGB")
        base_np = np.array(base_img)

        # Dialog with visual thumbnails and Add buttons
        config = self._prompt_masking_dialog(parent)
        if config is None:
            return None

        mask_pil, texture_pil, texture_mode, preserve_shading, threshold_val, mask_name, tex_name = config

        mask_np = np.array(mask_pil)
        texture_np = np.array(texture_pil) if texture_pil is not None else None

        # Perform compositing with threshold (identifying pixels > threshold)
        applied_np = composite_texture_or_overlay(
            base_np=base_np,
            mask_np_rgb=mask_np,
            texture_np=texture_np,
            mode=texture_mode,
            preserve_shading=preserve_shading,
            threshold=threshold_val,
        )

        binary_mask = extract_binary_mask(mask_pil, threshold=threshold_val)
        mask_pixels = int(np.sum(binary_mask))
        mask_pct = (mask_pixels / binary_mask.size) * 100

        details: dict[str, Any] = {
            "Mask source": mask_name,
            "Texture source": tex_name if texture_pil else "None (Direct Overlay)",
            "Intensity Threshold": f"> {threshold_val}",
            "Masked area (px)": f"{mask_pixels:,} px",
            "Mask coverage": f"{mask_pct:.2f}%",
            "Texture mode": texture_mode.capitalize(),
            "Shading blend": "Enabled" if preserve_shading else "Disabled",
        }

        out_img = Image.fromarray(applied_np)
        msg = f"Applied mask and texture compositing (Mask pixels > {threshold_val})."
        details["Output Type"] = "Composited Image"
        return ToolResult(image=out_img, message=msg, details=details)

    def _prompt_masking_dialog(
        self, parent: tk.Misc
    ) -> tuple[Image.Image, Image.Image | None, str, bool, int, str, str] | None:
        dialog = tk.Toplevel(parent)
        dialog.title("Masking & Texture Settings")
        dialog.geometry("520x460")
        dialog.minsize(480, 420)
        dialog.transient(parent)
        dialog.grab_set()

        container = ttk.Frame(dialog, padding=16)
        container.pack(fill="both", expand=True)

        # Bottom action buttons frame packed FIRST with side="bottom" so they are ALWAYS visible
        btn_frame = ttk.Frame(container)
        btn_frame.pack(side="bottom", fill="x", pady=(12, 0))

        content_frame = ttk.Frame(container)
        content_frame.pack(side="top", fill="both", expand=True)

        ttk.Label(
            content_frame,
            text="Masking & Texture Settings",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        sample_dir = Path("forensics_app/images/set2-images")

        current_mask_img: list[Image.Image | None] = [None]
        current_mask_name: list[str] = ["No mask loaded"]
        current_tex_img: list[Image.Image | None] = [None]
        current_tex_name: list[str] = ["None (Direct Coat Color)"]

        photo_refs: dict[str, ImageTk.PhotoImage | None] = {"mask": None, "tex": None}

        # --- Section 1: Mask / Overlay Preview Card ---
        mask_lf = ttk.LabelFrame(content_frame, text="Mask / Coat Overlay Image", padding=10)
        mask_lf.pack(fill="x", pady=(0, 8))

        mask_preview_lbl = tk.Label(mask_lf, width=70, height=70, bg="#20242b", relief="groove")
        mask_preview_lbl.pack(side="left", padx=(0, 12))

        mask_info_frame = ttk.Frame(mask_lf)
        mask_info_frame.pack(side="left", fill="both", expand=True)

        mask_name_lbl = ttk.Label(mask_info_frame, text="coat.png", font=("TkDefaultFont", 9, "bold"))
        mask_name_lbl.pack(anchor="w", pady=(0, 2))
        mask_dim_lbl = ttk.Label(mask_info_frame, text="", foreground="#666666")
        mask_dim_lbl.pack(anchor="w", pady=(0, 4))

        def update_mask_preview(img: Image.Image, name: str) -> None:
            current_mask_img[0] = img
            current_mask_name[0] = name
            mask_name_lbl.configure(text=name)
            mask_dim_lbl.configure(text=f"Dimensions: {img.width} × {img.height} px")
            thumb = img.copy()
            thumb.thumbnail((68, 68), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            photo_refs["mask"] = photo
            mask_preview_lbl.configure(image=photo, width=68, height=68)

        def add_mask_action() -> None:
            fn = filedialog.askopenfilename(
                title="Select Mask / Coat Image",
                initialdir=str(sample_dir) if sample_dir.exists() else ".",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.webp"), ("All files", "*.*")],
                parent=dialog,
            )
            if fn:
                try:
                    loaded = Image.open(fn).convert("RGB")
                    update_mask_preview(loaded, Path(fn).name)
                except Exception as err:
                    tk.messagebox.showerror("Invalid Image", str(err), parent=dialog)

        ttk.Button(mask_info_frame, text="+ Add / Change Coat", command=add_mask_action).pack(anchor="w")

        # Load default coat.png if exists
        default_coat_path = sample_dir / "coat.png"
        if default_coat_path.exists():
            try:
                update_mask_preview(Image.open(default_coat_path).convert("RGB"), "coat.png")
            except Exception:
                pass

        # --- Section 2: Texture Image Preview Card ---
        tex_lf = ttk.LabelFrame(content_frame, text="Texture Image (Optional)", padding=10)
        tex_lf.pack(fill="x", pady=(0, 8))

        tex_preview_lbl = tk.Label(tex_lf, width=70, height=70, bg="#20242b", relief="groove")
        tex_preview_lbl.pack(side="left", padx=(0, 12))

        tex_info_frame = ttk.Frame(tex_lf)
        tex_info_frame.pack(side="left", fill="both", expand=True)

        tex_name_lbl = ttk.Label(tex_info_frame, text="texture.png", font=("TkDefaultFont", 9, "bold"))
        tex_name_lbl.pack(anchor="w", pady=(0, 2))
        tex_dim_lbl = ttk.Label(tex_info_frame, text="", foreground="#666666")
        tex_dim_lbl.pack(anchor="w", pady=(0, 4))

        tex_btn_row = ttk.Frame(tex_info_frame)
        tex_btn_row.pack(anchor="w")

        # Create a blank placeholder for empty texture
        blank_tex_img = Image.new("RGB", (68, 68), (32, 36, 43))
        blank_tex_photo = ImageTk.PhotoImage(blank_tex_img)
        photo_refs["blank"] = blank_tex_photo

        def update_tex_preview(img: Image.Image | None, name: str) -> None:
            current_tex_img[0] = img
            current_tex_name[0] = name
            tex_name_lbl.configure(text=name)
            if img is not None:
                tex_dim_lbl.configure(text=f"Dimensions: {img.width} × {img.height} px")
                thumb = img.copy()
                thumb.thumbnail((68, 68), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                photo_refs["tex"] = photo
                tex_preview_lbl.configure(image=photo, width=68, height=68)
            else:
                tex_dim_lbl.configure(text="Using original coat color (No texture)")
                photo_refs["tex"] = None
                tex_preview_lbl.configure(image=blank_tex_photo, width=68, height=68)

        def add_tex_action() -> None:
            fn = filedialog.askopenfilename(
                title="Select Texture Image",
                initialdir=str(sample_dir) if sample_dir.exists() else ".",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.webp"), ("All files", "*.*")],
                parent=dialog,
            )
            if fn:
                try:
                    loaded = Image.open(fn).convert("RGB")
                    update_tex_preview(loaded, Path(fn).name)
                except Exception as err:
                    tk.messagebox.showerror("Invalid Image", str(err), parent=dialog)

        def clear_tex_action() -> None:
            update_tex_preview(None, "None (Direct Coat Color)")

        ttk.Button(tex_btn_row, text="+ Add / Change Texture", command=add_tex_action).pack(side="left", padx=(0, 6))
        ttk.Button(tex_btn_row, text="Clear Texture", command=clear_tex_action).pack(side="left")

        # Load default texture.png if exists
        default_tex_path = sample_dir / "texture.png"
        if default_tex_path.exists():
            try:
                update_tex_preview(Image.open(default_tex_path).convert("RGB"), "texture.png")
            except Exception:
                pass
        else:
            update_tex_preview(None, "None (Direct Coat Color)")

        # --- Section 3: Options ---
        opt_frame = ttk.Frame(content_frame)
        opt_frame.pack(fill="x", pady=6)

        ttk.Label(opt_frame, text="Texture Mode:").grid(row=0, column=0, sticky="w", pady=2)
        mode_cb = ttk.Combobox(opt_frame, values=["Tiled", "Resized", "Direct Mask Color"], state="readonly", width=16)
        mode_cb.current(0)
        mode_cb.grid(row=0, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(opt_frame, text="Mask Threshold (>):").grid(row=0, column=2, sticky="w", padx=(10, 4), pady=2)
        thresh_entry = ttk.Entry(opt_frame, width=6)
        thresh_entry.insert(0, "135")
        thresh_entry.grid(row=0, column=3, sticky="w", pady=2)

        shading_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Preserve folds & shading", variable=shading_var).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=6
        )

        result: list[tuple[Image.Image, Image.Image | None, str, bool, int, str, str] | None] = [None]

        def on_ok() -> None:
            if current_mask_img[0] is None:
                tk.messagebox.showwarning("Mask Required", "Please add a mask/coat image first.", parent=dialog)
                return
            m_mode_raw = mode_cb.get()
            if m_mode_raw == "Tiled":
                mode_str = "tiled"
            elif m_mode_raw == "Resized":
                mode_str = "resized"
            else:
                mode_str = "overlay"
            try:
                t_val = int(thresh_entry.get().strip() or "135")
            except ValueError:
                t_val = 135
            result[0] = (
                current_mask_img[0],
                current_tex_img[0],
                mode_str,
                shading_var.get(),
                t_val,
                current_mask_name[0],
                current_tex_name[0],
            )
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right")
        ttk.Button(btn_frame, text="Apply", command=on_ok).pack(side="right", padx=(0, 8))

        dialog.wait_window()
        return result[0]


