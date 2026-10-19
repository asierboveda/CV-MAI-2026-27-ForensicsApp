"""Normalized cross-correlation tool (Set 4.1).

Computes normalized cross-correlation between a target image and a template.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk
from skimage import color, feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def find_default_template_path() -> str:
    """Look for default course template images if present."""
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        base_dir / "images" / "Set4_images" / "template.png",
        base_dir / "images" / "Set4_images" / "template_br.png",
        base_dir / "images" / "Set4_images" / "person_template.bmp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def compute_cross_correlation(
    image: Image.Image | np.ndarray,
    template: Image.Image | np.ndarray,
    pad_input: bool = True,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """Compute normalized cross-correlation between image and template.

    Args:
        image: Source PIL Image or NumPy array.
        template: Template PIL Image or NumPy array.
        pad_input: If True, pad input to preserve image dimensions.

    Returns:
        corr_map: 2D array of correlation values in [-1, 1].
        peak_pos: (row, col) coordinates of the highest correlation peak.
        peak_val: Float value of the peak correlation.
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
        img_gray = color.rgb2gray(img_rgb)
    else:
        img_gray = image if image.ndim == 2 else color.rgb2gray(image)

    if isinstance(template, Image.Image):
        tmpl_rgb = np.array(template.convert("RGB"))
        tmpl_gray = color.rgb2gray(tmpl_rgb)
    else:
        tmpl_gray = template if template.ndim == 2 else color.rgb2gray(template)

    th, tw = tmpl_gray.shape[:2]
    ih, iw = img_gray.shape[:2]

    if th > ih or tw > iw:
        raise ValueError(
            f"Template size ({tw} × {th}) is larger than image size ({iw} × {ih})."
        )

    corr = feature.match_template(img_gray, tmpl_gray, pad_input=pad_input)
    corr = np.clip(corr, -1.0, 1.0)
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = float(corr[peak_y, peak_x])
    return corr, (int(peak_y), int(peak_x)), peak_val


def correlation_to_image(corr_map: np.ndarray, colormap: str = "gray") -> Image.Image:
    """Convert normalized correlation map to an 8-bit displayable PIL Image."""
    c_min = float(corr_map.min())
    c_max = float(corr_map.max())
    if c_max > c_min:
        norm = (corr_map - c_min) / (c_max - c_min)
    else:
        norm = np.zeros_like(corr_map)

    if colormap == "gray":
        u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(u8, mode="L")

    cmap = matplotlib.colormaps.get_cmap(colormap)
    rgba = cmap(norm)
    rgb_u8 = np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB")


def create_comparison_image(
    image: Image.Image,
    template: Image.Image,
    corr_map: np.ndarray,
    peak_pos: tuple[int, int],
    threshold: float = 0.8,
) -> Image.Image:
    """Create a multi-panel composite image similar to Set 4.1 slide."""
    fig = Figure(figsize=(14, 4), dpi=100)
    ax0 = fig.add_subplot(1, 4, 1)
    ax1 = fig.add_subplot(1, 4, 2)
    ax2 = fig.add_subplot(1, 4, 3)
    ax3 = fig.add_subplot(1, 4, 4)

    img_rgb = image.convert("RGB")
    ax0.imshow(img_rgb)
    ax0.set_title("Original image")
    ax0.axis("on")

    tmpl_rgb = template.convert("RGB")
    ax1.imshow(tmpl_rgb)
    ax1.set_title("Template")
    ax1.axis("on")

    py, px = peak_pos
    im_corr = ax2.imshow(corr_map, cmap="gray")
    ax2.plot(px, py, "ro", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    ax2.set_title("Normalized cross-corr image")
    fig.colorbar(im_corr, ax=ax2, shrink=0.7)
    ax2.axis("on")

    ax3.imshow(img_rgb)
    tw, th = template.size
    import matplotlib.patches as patches
    rect = patches.Rectangle(
        (px - tw // 2, py - th // 2), tw, th,
        linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--"
    )
    ax3.add_patch(rect)
    ax3.set_title(f"Matched template (peak at {px}, {py})")
    ax3.axis("on")

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


class CorrelationPlotWindow(tk.Toplevel):
    """Dedicated interactive window showing normalized cross-correlation overview."""

    def __init__(
        self,
        parent: tk.Misc,
        corr: np.ndarray,
        peak_pos: tuple[int, int],
        peak_val: float,
        original_img: Image.Image | None = None,
        template_img: Image.Image | None = None,
        threshold: float | None = None,
        title: str = "Normalized Cross-Correlation Overview",
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.geometry("1180x480")
        self.minsize(860, 380)

        self.corr = corr
        self.peak_pos = peak_pos
        self.peak_val = peak_val
        self.original_img = original_img
        self.template_img = template_img
        self.threshold = threshold

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        tab_overview = ttk.Frame(notebook)
        notebook.add(tab_overview, text="Slide 4.1 Comparison Overview")
        self._build_overview_tab(tab_overview)

    def _build_overview_tab(self, parent: ttk.Frame) -> None:
        fig = Figure(figsize=(11.5, 4.5), dpi=100)
        ax0 = fig.add_subplot(1, 4, 1)
        ax1 = fig.add_subplot(1, 4, 2)
        ax2 = fig.add_subplot(1, 4, 3)
        ax3 = fig.add_subplot(1, 4, 4)

        py, px = self.peak_pos

        if self.original_img is not None:
            ax0.imshow(self.original_img.convert("RGB"))
        ax0.set_title("Original image")
        ax0.axis("on")

        if self.template_img is not None:
            ax1.imshow(self.template_img.convert("RGB"))
            tw, th = self.template_img.size
            ax1.set_title(f"Template ({tw}×{th} px)")
        else:
            ax1.set_title("Template")
        ax1.axis("on")

        im_corr = ax2.imshow(self.corr, cmap="gray")
        fig.colorbar(im_corr, ax=ax2, shrink=0.7)
        ax2.plot(px, py, "ro", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax2.set_title(f"Normalized cross-corr\n(Peak: {self.peak_val:.3f})")
        ax2.axis("on")

        if self.original_img is not None:
            ax3.imshow(self.original_img.convert("RGB"))
            if self.template_img is not None:
                tw, th = self.template_img.size
                import matplotlib.patches as patches
                rect = patches.Rectangle(
                    (px - tw // 2, py - th // 2), tw, th,
                    linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--"
                )
                ax3.add_patch(rect)
        ax3.set_title(f"Matched template\n(Peak at X={px}, Y={py})")
        ax3.axis("on")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        info = (
            f"Peak correlation: {self.peak_val:.4f} at (X={px}, Y={py}) | "
            f"Range: [{self.corr.min():.3f}, {self.corr.max():.3f}] | "
            f"Threshold: {self.threshold if self.threshold is not None else 0.8:.2f}"
        )
        ttk.Label(parent, text=info, padding=6, relief="sunken").pack(fill="x")


class CrossCorrelationDialog(simpledialog.Dialog):
    """Dialog for configuring normalized cross-correlation parameters."""

    def __init__(self, parent: tk.Misc, image_size: tuple[int, int]) -> None:
        self.image_size = image_size
        self.template_path = find_default_template_path()
        self.pad_input = True
        self.show_3d_plot = True
        self.show_overview = True
        self.threshold = 0.80
        self.output_mode = "Correlation Map (Normalized Grayscale)"
        self.preview_image: ImageTk.PhotoImage | None = None
        super().__init__(parent, "Normalized Cross-Correlation")

    def body(self, master: tk.Frame) -> tk.Widget:
        row = 0

        lbl_tmpl = ttk.Label(master, text="Template image:", font=("TkDefaultFont", 9, "bold"))
        lbl_tmpl.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 2))
        row += 1

        self.tmpl_entry = ttk.Entry(master, width=44)
        self.tmpl_entry.insert(0, self.template_path)
        self.tmpl_entry.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=3)

        btn_browse = ttk.Button(master, text="Browse…", command=self._browse_template)
        btn_browse.grid(row=row, column=2, padx=6, pady=3)
        row += 1

        self.lbl_preview_info = ttk.Label(master, text="No template chosen")
        self.lbl_preview_info.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=2)
        row += 1

        self.preview_canvas = tk.Canvas(master, width=80, height=80, bg="#f3f4f6", highlightthickness=1)
        self.preview_canvas.grid(row=row, column=0, columnspan=3, padx=6, pady=4, sticky="w")
        row += 1

        ttk.Separator(master, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        self.pad_var = tk.BooleanVar(value=True)
        chk_pad = ttk.Checkbutton(
            master, text="Pad input (output dimensions match search image)", variable=self.pad_var
        )
        chk_pad.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=3)
        row += 1

        ttk.Label(master, text="Peak detection threshold (-1.0 to 1.0):").grid(
            row=row, column=0, sticky="w", padx=6, pady=4
        )
        self.thresh_entry = ttk.Entry(master, width=12)
        self.thresh_entry.insert(0, "0.80")
        self.thresh_entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
        row += 1

        self._update_preview()
        return self.tmpl_entry

    def _browse_template(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select Template Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
            parent=self,
        )
        if file_path:
            self.tmpl_entry.delete(0, tk.END)
            self.tmpl_entry.insert(0, file_path)
            self._update_preview()

    def _update_preview(self) -> None:
        path_str = self.tmpl_entry.get().strip()
        if not path_str or not Path(path_str).exists():
            self.lbl_preview_info.configure(text="No valid template selected.")
            self.preview_canvas.delete("all")
            return
        try:
            with Image.open(path_str) as img:
                tw, th = img.size
                self.lbl_preview_info.configure(
                    text=f"Template: {Path(path_str).name} ({tw} × {th} px, {img.mode})"
                )
                preview = img.copy()
                preview.thumbnail((76, 76), Image.Resampling.NEAREST)
                self.preview_image = ImageTk.PhotoImage(preview)
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(40, 40, image=self.preview_image)
        except Exception as error:
            self.lbl_preview_info.configure(text=f"Could not load preview: {error}")

    def validate(self) -> bool:
        path_str = self.tmpl_entry.get().strip()
        if not path_str:
            messagebox.showerror("Missing Template", "Please select a template image.", parent=self)
            return False
        if not Path(path_str).exists():
            messagebox.showerror("File Not Found", f"Template file not found:\n{path_str}", parent=self)
            return False

        try:
            with Image.open(path_str) as img:
                tw, th = img.size
            iw, ih = self.image_size
            if tw > iw or th > ih:
                messagebox.showerror(
                    "Template Too Large",
                    f"Template dimensions ({tw} × {th}) cannot exceed image dimensions ({iw} × {ih}).",
                    parent=self,
                )
                return False
        except Exception as error:
            messagebox.showerror("Invalid Image", f"Could not read template image: {error}", parent=self)
            return False

        try:
            thresh = float(self.thresh_entry.get())
            if not (-1.0 <= thresh <= 1.0):
                messagebox.showerror("Invalid Threshold", "Threshold must be between -1.0 and 1.0.", parent=self)
                return False
        except ValueError:
            messagebox.showerror("Invalid Input", "Threshold must be a valid numeric value.", parent=self)
            return False

        return True

    def apply(self) -> None:
        self.result = {
            "template_path": self.tmpl_entry.get().strip(),
            "pad_input": self.pad_var.get(),
            "threshold": float(self.thresh_entry.get()),
            "show_3d_plot": False,
            "show_overview": True,
            "output_mode": "Slide 4.1 Comparison Overview",
        }


class NormalizedCrossCorrelationTool(ForensicsTool):
    """Tool implementing Set 4.1 Normalized cross-correlation."""

    tool_id = "cross_correlation"
    title = "Normalized cross-correlation"
    category = "Set 4"
    description = "Compute normalized cross-correlation with a template and display comparison overview with colorbar."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        dialog = CrossCorrelationDialog(parent, document.current.size)
        if not dialog.result:
            return None

        cfg = dialog.result
        template_path = cfg["template_path"]
        pad_input = cfg["pad_input"]
        threshold = cfg["threshold"]

        with Image.open(template_path) as tmpl_file:
            template = tmpl_file.copy()

        corr, peak_pos, peak_val = compute_cross_correlation(
            document.current, template, pad_input=pad_input
        )
        py, px = peak_pos

        CorrelationPlotWindow(
            parent,
            corr=corr,
            peak_pos=peak_pos,
            peak_val=peak_val,
            original_img=document.current,
            template_img=template,
            threshold=threshold,
            title="Set 4.1: Normalized Cross-Correlation Overview",
        )

        details = {
            "Operation": "Normalized Cross-Correlation",
            "Template": Path(template_path).name,
            "Template size": f"{template.width} × {template.height}",
            "Peak correlation": f"{peak_val:.4f}",
            "Peak position (X, Y)": f"({px}, {py})",
            "Threshold": threshold,
            "Pad input": pad_input,
            "Correlation min/max": f"{corr.min():.3f} / {corr.max():.3f}",
            "Correlation mean": f"{corr.mean():.3f}",
        }

        return ToolResult(
            image=None,
            message=(
                f"Normalized cross-correlation completed. Peak {peak_val:.4f} at ({px}, {py})."
            ),
            details=details,
        )
