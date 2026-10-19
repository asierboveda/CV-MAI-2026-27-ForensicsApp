"""Histogram of Oriented Gradients (HOG) extraction and visualization (Set 4.3).

Extracts and visualizes HOG features with user-configurable:
- orientations
- pixels_per_cell
- cells_per_block
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from skimage import exposure, feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def compute_hog(
    image: Image.Image | np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (6, 6),
    cells_per_block: tuple[int, int] = (3, 3),
) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """Compute HOG descriptor and visualization image using skimage.feature.hog.

    Args:
        image: PIL Image or NumPy array.
        orientations: Number of gradient orientation bins.
        pixels_per_cell: (y, x) size of cell in pixels.
        cells_per_block: (y, x) number of cells in each block.

    Returns:
        fd: 1D feature descriptor vector.
        hog_image: Raw 2D array returned by skimage.feature.hog.
        visual_image: Rescaled 8-bit PIL Image for clear visual inspection.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = image

    channel_axis = 2 if arr.ndim == 3 and arr.shape[2] in (3, 4) else None
    if channel_axis == 2 and arr.shape[2] == 4:
        arr = arr[..., :3]

    fd, hog_image = feature.hog(
        arr,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        channel_axis=channel_axis,
    )

    if hog_image.max() > 0:
        hog_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        u8 = np.clip(hog_rescaled * 255.0, 0, 255).astype(np.uint8)
    else:
        u8 = np.zeros_like(hog_image, dtype=np.uint8)

    visual_image = Image.fromarray(u8, mode="L")
    return fd, hog_image, visual_image


def create_hog_comparison_image(
    image: Image.Image,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (6, 6),
    cells_per_block: tuple[int, int] = (3, 3),
    orientations_list: tuple[int, ...] | None = None,
) -> Image.Image:
    """Create a composite comparison image (Original vs HOG)."""
    if orientations_list is not None and len(orientations_list) > 1:
        num_cols = 1 + len(orientations_list)
        fig = Figure(figsize=(3.5 * num_cols, 4), dpi=100)
        ax0 = fig.add_subplot(1, num_cols, 1)
        ax0.imshow(image.convert("RGB"))
        ax0.set_title("Original image")
        for i, o in enumerate(orientations_list, start=2):
            ax = fig.add_subplot(1, num_cols, i)
            _, _, vis_pil = compute_hog(
                image,
                orientations=o,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
            )
            ax.imshow(vis_pil, cmap="gray")
            ax.set_title(f"HOG (orientation={o})")
    else:
        fig = Figure(figsize=(8, 4), dpi=100)
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)
        ax0.imshow(image.convert("RGB"))
        ax0.set_title("Original image")
        _, _, vis_pil = compute_hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
        )
        ax1.imshow(vis_pil, cmap="gray")
        ax1.set_title(f"HOG (orientations={orientations})")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    fig.tight_layout()
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    composite = Image.frombuffer(
        "RGBA",
        fig.canvas.get_width_height(),
        rgba_buffer,
        "raw",
        "RGBA",
        0,
        1,
    ).convert("RGB")
    return composite


class HOGComparisonWindow(tk.Toplevel):
    """Window showing Original image and HOG visualization with selected parameters."""

    def __init__(
        self,
        parent: tk.Misc | None,
        image: Image.Image,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (6, 6),
        cells_per_block: tuple[int, int] = (3, 3),
        vis_image: Image.Image | None = None,
        descriptor_len: int | None = None,
        orientations_list: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("HOG Feature Extraction & Visualization")
        self.geometry("920x480")
        self.minsize(720, 360)

        if vis_image is None or descriptor_len is None:
            fd, _, vis_image = compute_hog(
                image,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
            )
            descriptor_len = len(fd)

        fig = Figure(figsize=(9, 4.5), dpi=100)
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)

        ax0.imshow(image.convert("RGB"))
        ax0.set_title("Original image", fontsize=11, fontweight="bold")
        ax0.axis("on")

        ax1.imshow(vis_image, cmap="gray")
        ax1.set_title(
            f"HOG Visualization\n(orientations={orientations}, "
            f"cell={pixels_per_cell[1]}×{pixels_per_cell[0]}, "
            f"block={cells_per_block[1]}×{cells_per_block[0]})",
            fontsize=10,
            fontweight="bold",
        )
        ax1.axis("on")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()

        info = (
            f"Orientations: {orientations} bins | "
            f"Cell size: {pixels_per_cell[1]}×{pixels_per_cell[0]} px | "
            f"Block size: {cells_per_block[1]}×{cells_per_block[0]} cells | "
            f"Descriptor length: {descriptor_len:,} features"
        )
        ttk.Label(self, text=info, padding=6, relief="sunken").pack(fill="x")


class HOGDialog(simpledialog.Dialog):
    """Configuration dialog for HOG parameters."""

    def __init__(self, parent: tk.Misc, image_size: tuple[int, int]) -> None:
        self.image_size = image_size
        super().__init__(parent, "HOG Feature Extraction Configs")

    def body(self, master: tk.Frame) -> tk.Widget:
        row = 0

        ttk.Label(
            master,
            text="Configure HOG Parameters:",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 8))
        row += 1

        ttk.Label(master, text="Orientations (bins):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        self.orient_entry = ttk.Entry(master, width=12)
        self.orient_entry.insert(0, "9")
        self.orient_entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
        row += 1

        ttk.Label(master, text="Pixels per cell (width, height):").grid(
            row=row, column=0, sticky="w", padx=6, pady=4
        )
        cell_frame = ttk.Frame(master)
        cell_frame.grid(row=row, column=1, sticky="w", padx=6, pady=4)

        self.cell_w_entry = ttk.Entry(cell_frame, width=5)
        self.cell_w_entry.insert(0, "6")
        self.cell_w_entry.pack(side="left")
        ttk.Label(cell_frame, text=" × ").pack(side="left")
        self.cell_h_entry = ttk.Entry(cell_frame, width=5)
        self.cell_h_entry.insert(0, "6")
        self.cell_h_entry.pack(side="left")
        row += 1

        ttk.Label(master, text="Cells per block (width, height):").grid(
            row=row, column=0, sticky="w", padx=6, pady=4
        )
        block_frame = ttk.Frame(master)
        block_frame.grid(row=row, column=1, sticky="w", padx=6, pady=4)

        self.block_w_entry = ttk.Entry(block_frame, width=5)
        self.block_w_entry.insert(0, "3")
        self.block_w_entry.pack(side="left")
        ttk.Label(block_frame, text=" × ").pack(side="left")
        self.block_h_entry = ttk.Entry(block_frame, width=5)
        self.block_h_entry.insert(0, "3")
        self.block_h_entry.pack(side="left")
        row += 1

        return self.orient_entry

    def validate(self) -> bool:
        try:
            orient = int(self.orient_entry.get())
            if not (1 <= orient <= 36):
                messagebox.showerror(
                    "Invalid Orientations", "Orientations must be between 1 and 36.", parent=self
                )
                return False
        except ValueError:
            messagebox.showerror("Invalid Input", "Orientations must be an integer.", parent=self)
            return False

        try:
            cw = int(self.cell_w_entry.get())
            ch = int(self.cell_h_entry.get())
            if cw < 1 or ch < 1:
                messagebox.showerror("Invalid Cell Size", "Cell dimensions must be >= 1.", parent=self)
                return False
        except ValueError:
            messagebox.showerror("Invalid Input", "Cell dimensions must be positive integers.", parent=self)
            return False

        try:
            bw = int(self.block_w_entry.get())
            bh = int(self.block_h_entry.get())
            if bw < 1 or bh < 1:
                messagebox.showerror("Invalid Block Size", "Block dimensions must be >= 1.", parent=self)
                return False
        except ValueError:
            messagebox.showerror("Invalid Input", "Block dimensions must be positive integers.", parent=self)
            return False

        iw, ih = self.image_size
        if (cw * bw > iw) or (ch * bh > ih):
            messagebox.showerror(
                "Block Too Large",
                f"Block size ({cw * bw} × {ch * bh} px) exceeds image size ({iw} × {ih} px).",
                parent=self,
            )
            return False

        return True

    def apply(self) -> None:
        self.result = {
            "orientations": int(self.orient_entry.get()),
            "pixels_per_cell": (int(self.cell_h_entry.get()), int(self.cell_w_entry.get())),
            "cells_per_block": (int(self.block_h_entry.get()), int(self.block_w_entry.get())),
        }


class HOGTool(ForensicsTool):
    """Tool implementing HOG extraction and visualization."""

    tool_id = "hog_visualization"
    title = "HOG visualization"
    category = "Set 4"
    description = "Extract Histogram of Oriented Gradients (HOG) with configurable orientations, cell, and block sizes."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        dialog = HOGDialog(parent, document.current.size)
        if not dialog.result:
            return None

        cfg = dialog.result
        orientations = cfg["orientations"]
        pixels_per_cell = cfg["pixels_per_cell"]
        cells_per_block = cfg["cells_per_block"]

        fd, raw_hog, vis_image = compute_hog(
            document.current,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
        )

        HOGComparisonWindow(
            parent,
            image=document.current,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            vis_image=vis_image,
            descriptor_len=len(fd),
        )

        details = {
            "Operation": "HOG Feature Extraction",
            "Descriptor length": f"{len(fd):,} features",
            "Orientations": orientations,
            "Pixels per cell": f"{pixels_per_cell[1]} × {pixels_per_cell[0]} px",
            "Cells per block": f"{cells_per_block[1]} × {cells_per_block[0]} cells",
            "Block size": f"{pixels_per_cell[1] * cells_per_block[1]} × {pixels_per_cell[0] * cells_per_block[0]} px",
            "Max gradient magnitude": f"{raw_hog.max():.2f}",
        }

        return ToolResult(
            image=None,
            message=(
                f"HOG extracted ({orientations} orientations, "
                f"cell {pixels_per_cell[1]}×{pixels_per_cell[0]}, "
                f"block {cells_per_block[1]}×{cells_per_block[0]}). "
                f"Descriptor size: {len(fd):,}."
            ),
            details=details,
        )
