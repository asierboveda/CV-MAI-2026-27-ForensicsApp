"""Template matching configs tool (Set 4.2).

Performs template matching with configurable rotation angle, correlation threshold,
and multiple superimposition styles inspired by Set 3:
1. Red highlight on image
2. Transparent red overlay
3. Darken everything except matches
4. Contours / Bounding box outline
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk
from skimage import color, feature, transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cross_correlation import find_default_template_path


def rotate_template(
    template: Image.Image | np.ndarray, angle: float
) -> np.ndarray:
    """Rotate template image by angle in degrees counter-clockwise."""
    if isinstance(template, Image.Image):
        rgb = np.array(template.convert("RGB"))
        gray = color.rgb2gray(rgb)
    else:
        gray = template if template.ndim == 2 else color.rgb2gray(template)

    if abs(angle) < 1e-4:
        return gray
    return transform.rotate(gray, angle, resize=False)


def match_template_rotated(
    image: Image.Image | np.ndarray,
    template: Image.Image | np.ndarray,
    angle: float = 0.0,
    pad_input: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float]:
    """Rotate template by angle and compute normalized cross-correlation.

    Returns:
        corr_map: 2D array of correlation values.
        rotated_template: 2D array of rotated template.
        peak_pos: (row, col) coordinates of highest correlation.
        peak_val: Maximum correlation value.
    """
    if isinstance(image, Image.Image):
        img_gray = color.rgb2gray(np.array(image.convert("RGB")))
    else:
        img_gray = image if image.ndim == 2 else color.rgb2gray(image)

    tmpl_rot = rotate_template(template, angle)
    th, tw = tmpl_rot.shape[:2]
    ih, iw = img_gray.shape[:2]

    if th > ih or tw > iw:
        raise ValueError(
            f"Template size ({tw} × {th}) is larger than image size ({iw} × {ih})."
        )

    corr = feature.match_template(img_gray, tmpl_rot, pad_input=pad_input)
    corr = np.clip(corr, -1.0, 1.0)
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = float(corr[py, px])
    return corr, tmpl_rot, (int(py), int(px)), peak_val


def superimpose_detection(
    image: Image.Image,
    template_size: tuple[int, int],
    corr_map: np.ndarray,
    threshold: float = 0.8,
    style: int = 1,
    min_distance: int = 5,
    dot_radius: int = 3,
) -> tuple[Image.Image, list[tuple[int, int]], int]:
    """Apply one of the 4 superimposition styles from Set 3 onto the search image.

    Styles:
        1: Red highlight on image (red dot/circle at match)
        2: Transparent red overlay over matched region
        3: Darken everything except matches (dimmed background)
        4: Bounding box / contour outline around matched template region

    Returns:
        output_image: Superimposed PIL image.
        peaks: List of (row, col) coordinates for detected matches.
        matches_count: Total number of matches detected.
    """
    base_rgb = np.array(image.convert("RGB"))
    h_img, w_img = base_rgb.shape[:2]
    tw, th = template_size

    peaks_arr = feature.peak_local_max(
        corr_map, min_distance=max(3, min_distance), threshold_abs=threshold
    )
    peaks = [(int(r), int(c)) for r, c in peaks_arr]

    if not peaks:
        gy, gx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        if float(corr_map[gy, gx]) >= threshold:
            peaks = [(int(gy), int(gx))]

    y_grid, x_grid = np.ogrid[:h_img, :w_img]

    if style == 1:
        result = base_rgb.copy()
        for py, px in peaks:
            dist_sq = (y_grid - py) ** 2 + (x_grid - px) ** 2
            result[dist_sq <= dot_radius ** 2] = [255, 0, 0]
        return Image.fromarray(result), peaks, len(peaks)

    elif style == 2:
        blended = base_rgb.copy().astype(np.float32)
        alpha = 0.6
        mask = np.zeros((h_img, w_img), dtype=bool)

        for py, px in peaks:
            y1 = max(0, py - th // 2)
            y2 = min(h_img, py + th // 2 + 1)
            x1 = max(0, px - tw // 2)
            x2 = min(w_img, px + tw // 2 + 1)
            mask[y1:y2, x1:x2] = True

        blended[mask] = (1 - alpha) * blended[mask] + alpha * np.array([255, 0, 0], dtype=np.float32)
        for py, px in peaks:
            dist_sq = (y_grid - py) ** 2 + (x_grid - px) ** 2
            blended[dist_sq <= dot_radius ** 2] = [255, 0, 0]

        result = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(result), peaks, len(peaks)

    elif style == 3:
        dimmed = (base_rgb * 0.4).astype(np.uint8)
        for py, px in peaks:
            y1 = max(0, py - th // 2)
            y2 = min(h_img, py + th // 2 + 1)
            x1 = max(0, px - tw // 2)
            x2 = min(w_img, px + tw // 2 + 1)
            dimmed[y1:y2, x1:x2] = base_rgb[y1:y2, x1:x2]
            dist_sq = (y_grid - py) ** 2 + (x_grid - px) ** 2
            dimmed[dist_sq <= dot_radius ** 2] = [255, 0, 0]
        return Image.fromarray(dimmed), peaks, len(peaks)

    else:
        result = base_rgb.copy()
        for py, px in peaks:
            y1 = max(0, py - th // 2)
            y2 = min(h_img, py + th // 2 + 1)
            x1 = max(0, px - tw // 2)
            x2 = min(w_img, px + tw // 2 + 1)

            thickness = 2
            result[y1 : min(h_img, y1 + thickness), x1:x2] = [255, 0, 0]
            result[max(0, y2 - thickness) : y2, x1:x2] = [255, 0, 0]
            result[y1:y2, x1 : min(w_img, x1 + thickness)] = [255, 0, 0]
            result[y1:y2, max(0, x2 - thickness) : x2] = [255, 0, 0]

            dist_sq = (y_grid - py) ** 2 + (x_grid - px) ** 2
            result[dist_sq <= dot_radius ** 2] = [255, 0, 0]

        return Image.fromarray(result), peaks, len(peaks)


class TemplateMatchingResultWindow(tk.Toplevel):
    """Window showing Original image, Template, Detections (with red dots), and 2D correlation matrix."""

    def __init__(
        self,
        parent: tk.Misc,
        original_img: Image.Image,
        template_img: Image.Image,
        superimposed_img: Image.Image,
        corr_map: np.ndarray,
        peaks: list[tuple[int, int]],
        peak_val: float,
        peak_pos: tuple[int, int],
        threshold: float,
        angle: float,
        template_name: str,
        template_size: tuple[int, int],
        style_name: str,
        title: str = "Template Matching Results",
        **_kwargs,
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.geometry("1280x520")
        self.minsize(920, 400)

        fig = Figure(figsize=(13.5, 4.4), dpi=100)
        ax0 = fig.add_subplot(1, 4, 1)
        ax1 = fig.add_subplot(1, 4, 2)
        ax2 = fig.add_subplot(1, 4, 3)
        ax3 = fig.add_subplot(1, 4, 4)

        ax0.imshow(original_img.convert("RGB"))
        ax0.set_title("1. Original search image", fontsize=10, fontweight="bold")
        ax0.axis("on")

        ax1.imshow(template_img.convert("RGB"))
        tw, th = template_size
        ax1.set_title(f"2. Template ({tw}×{th} px)\nangle = {angle:g}°", fontsize=10, fontweight="bold")
        ax1.axis("on")

        ax2.imshow(superimposed_img.convert("RGB"))
        num_matches = len(peaks)
        ax2.set_title(
            f"3. Detections ({num_matches} match{'es' if num_matches != 1 else ''})\n(Red dots / Darkened bg)",
            fontsize=10,
            fontweight="bold",
        )
        ax2.axis("on")

        im_corr = ax3.imshow(corr_map, cmap="viridis")
        fig.colorbar(im_corr, ax=ax3, shrink=0.72, pad=0.03)
        if peaks:
            py_list = [p[0] for p in peaks]
            px_list = [p[1] for p in peaks]
            ax3.scatter(
                px_list,
                py_list,
                color="red",
                s=36,
                edgecolor="white",
                linewidth=1.2,
                label=f"Matches (≥ {threshold:.2f})",
                zorder=5,
            )
            ax3.legend(loc="upper right", fontsize=8)
        ax3.set_title(
            f"4. 2D Corr Matrix (max r = {peak_val:.3f})\nPoints ≥ {threshold:.2f} marked",
            fontsize=10,
            fontweight="bold",
        )
        ax3.axis("on")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()

        info = (
            f"Template: {template_name} ({tw}×{th} px) | "
            f"Rotation: {angle:g}° | "
            f"Threshold: {threshold:.2f} | "
            f"Matches found: {num_matches} | "
            f"Peak correlation: {peak_val:.4f} at (X={peak_pos[1]}, Y={peak_pos[0]}) | "
            f"Style: {style_name}"
        )
        ttk.Label(self, text=info, padding=6, relief="sunken").pack(fill="x")


CorrelationPlotWindow = TemplateMatchingResultWindow


class TemplateMatchingDialog(simpledialog.Dialog):
    """Configuration dialog for Set 4.2 Template Matching."""

    def __init__(self, parent: tk.Misc, image_size: tuple[int, int]) -> None:
        self.image_size = image_size
        self.template_path = find_default_template_path()
        self.preview_image: ImageTk.PhotoImage | None = None
        super().__init__(parent, "Template Matching Configs")

    def body(self, master: tk.Frame) -> tk.Widget:
        row = 0

        ttk.Label(master, text="Template image:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 2)
        )
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

        ttk.Label(master, text="Rotation angle a (degrees):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        self.angle_entry = ttk.Entry(master, width=8)
        self.angle_entry.insert(0, "0.0")
        self.angle_scale = ttk.Scale(master, from_=-180, to=180, orient="horizontal", command=self._on_angle_slide)
        self.angle_scale.set(0.0)
        self.angle_scale.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        self.angle_entry.grid(row=row, column=2, sticky="w", padx=6, pady=4)
        self.angle_entry.bind("<FocusOut>", self._on_angle_entry)
        row += 1

        ttk.Label(master, text="Correlation threshold (0.0 – 1.0):").grid(
            row=row, column=0, sticky="w", padx=6, pady=4
        )
        self.thresh_entry = ttk.Entry(master, width=8)
        self.thresh_entry.insert(0, "0.80")
        self.thresh_scale = ttk.Scale(master, from_=0.0, to=1.0, orient="horizontal", command=self._on_thresh_slide)
        self.thresh_scale.set(0.80)
        self.thresh_scale.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        self.thresh_entry.grid(row=row, column=2, sticky="w", padx=6, pady=4)
        self.thresh_entry.bind("<FocusOut>", self._on_thresh_entry)
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

    def _on_angle_slide(self, val: str) -> None:
        if not hasattr(self, "angle_entry"):
            return
        try:
            v = float(val)
            self.angle_entry.delete(0, tk.END)
            self.angle_entry.insert(0, f"{v:.1f}")
        except ValueError:
            pass

    def _on_angle_entry(self, _event=None) -> None:
        try:
            v = float(self.angle_entry.get())
            self.angle_scale.set(v)
        except ValueError:
            pass

    def _on_thresh_slide(self, val: str) -> None:
        if not hasattr(self, "thresh_entry"):
            return
        try:
            v = float(val)
            self.thresh_entry.delete(0, tk.END)
            self.thresh_entry.insert(0, f"{v:.2f}")
        except ValueError:
            pass

    def _on_thresh_entry(self, _event=None) -> None:
        try:
            v = float(self.thresh_entry.get())
            self.thresh_scale.set(v)
        except ValueError:
            pass

    def validate(self) -> bool:
        path_str = self.tmpl_entry.get().strip()
        if not path_str or not Path(path_str).exists():
            messagebox.showerror("Invalid Template", "Please select an existing template image.", parent=self)
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
            messagebox.showerror("Invalid Image", f"Could not open template image: {error}", parent=self)
            return False

        try:
            angle = float(self.angle_entry.get())
            if not (-360.0 <= angle <= 360.0):
                messagebox.showerror("Invalid Angle", "Rotation angle must be between -360 and 360 degrees.", parent=self)
                return False
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric rotation angle.", parent=self)
            return False

        try:
            thresh = float(self.thresh_entry.get())
            if not (0.0 <= thresh <= 1.0):
                messagebox.showerror("Invalid Threshold", "Threshold must be between 0.0 and 1.0.", parent=self)
                return False
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric threshold.", parent=self)
            return False

        return True

    def apply(self) -> None:
        self.result = {
            "template_path": self.tmpl_entry.get().strip(),
            "angle": float(self.angle_entry.get()),
            "threshold": float(self.thresh_entry.get()),
            "style_idx": 3,
            "style_name": "Darken everything except template detections",
            "show_3d_plot": False,
        }


class TemplateMatchingTool(ForensicsTool):
    """Tool implementing Set 4.2 Template matching configs."""

    tool_id = "template_matching"
    title = "Template matching"
    category = "Set 4"
    description = "Match a template with configurable rotation, threshold, and Set 3 superimposition styles."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        dialog = TemplateMatchingDialog(parent, document.current.size)
        if not dialog.result:
            return None

        cfg = dialog.result
        template_path = cfg["template_path"]
        angle = cfg["angle"]
        threshold = cfg["threshold"]
        style_idx = cfg.get("style_idx", 3)
        style_name = cfg.get("style_name", "Darken everything except template detections")

        with Image.open(template_path) as tmpl_file:
            template = tmpl_file.copy()

        corr, tmpl_rot, peak_pos, peak_val = match_template_rotated(
            document.current, template, angle=angle, pad_input=True
        )
        py, px = peak_pos

        output_image, peaks, num_matches = superimpose_detection(
            document.current,
            template_size=template.size,
            corr_map=corr,
            threshold=threshold,
            style=style_idx,
        )

        rotated_pil = Image.fromarray(
            np.clip(tmpl_rot * 255.0, 0, 255).astype(np.uint8)
        )

        CorrelationPlotWindow(
            parent,
            original_img=document.current,
            template_img=rotated_pil,
            superimposed_img=output_image,
            corr_map=corr,
            peaks=peaks,
            peak_val=peak_val,
            peak_pos=peak_pos,
            threshold=threshold,
            angle=angle,
            template_name=Path(template_path).name,
            template_size=template.size,
            style_name=style_name,
        )

        match_desc = (
            f"Found {num_matches} match(es) exceeding threshold {threshold:.2f}."
            if num_matches > 0
            else f"No peaks exceeded threshold {threshold:.2f} (max peak {peak_val:.4f})."
        )

        details = {
            "Operation": "Template Matching",
            "Template": Path(template_path).name,
            "Template size": f"{template.width} × {template.height}",
            "Rotation angle": f"{angle:g}°",
            "Threshold": threshold,
            "Matches found": num_matches,
            "Peak correlation": f"{peak_val:.4f}",
            "Peak position (X, Y)": f"({px}, {py})",
            "Superimposition style": style_name,
            "Correlation min/max": f"{corr.min():.3f} / {corr.max():.3f}",
        }

        return ToolResult(
            image=None,
            message=f"Template matching (angle {angle:g}°, threshold {threshold:.2f}): {match_desc}",
            details=details,
        )
