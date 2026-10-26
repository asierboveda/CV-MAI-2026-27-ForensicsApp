"""Feature matching with a second image for Set 5.3."""

from __future__ import annotations

import io
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, feature
from skimage.feature import SIFT, match_descriptors, plot_matched_features
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def match_features_between(
    image1: Image.Image,
    image2: Image.Image,
    method: str = "ORB",
    cross_check: bool = True,
    max_ratio: float = 0.8,
    metric: str | None = None,
    n_keypoints: int = 500,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
    title1: str = "Image 1",
    title2: str = "Image 2",
    ransac_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image]:
    """Extract features from two images, match descriptors, and render side-by-side plot.

    Returns (keypoints1, keypoints2, matches, visualization_image).
    """
    gray1 = color.rgb2gray(np.array(image1.convert("RGB")))
    gray2 = color.rgb2gray(np.array(image2.convert("RGB")))

    method_upper = method.upper()
    if method_upper == "ORB":
        extractor = feature.ORB(n_keypoints=n_keypoints)
        try:
            extractor.detect_and_extract(gray1)
            kp1, desc1 = extractor.keypoints, extractor.descriptors
        except RuntimeError:
            kp1, desc1 = np.empty((0, 2)), np.empty((0, 256), dtype=bool)

        try:
            extractor.detect_and_extract(gray2)
            kp2, desc2 = extractor.keypoints, extractor.descriptors
        except RuntimeError:
            kp2, desc2 = np.empty((0, 2)), np.empty((0, 256), dtype=bool)

        chosen_metric = "hamming" if metric in (None, "auto") else metric

    elif method_upper == "SIFT":
        extractor = SIFT(n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min)
        extractor.detect_and_extract(gray1)
        kp1, desc1 = extractor.keypoints, extractor.descriptors

        extractor.detect_and_extract(gray2)
        kp2, desc2 = extractor.keypoints, extractor.descriptors

        chosen_metric = "euclidean" if metric in (None, "auto") else metric
    else:
        raise ValueError(f"Unknown feature extractor method: {method}")

    # Compute matches
    if len(desc1) == 0 or len(desc2) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = match_descriptors(
            desc1,
            desc2,
            cross_check=cross_check,
            max_ratio=max_ratio,
            metric=chosen_metric,
        )

    # We need 4 points to estimate an homography

    if ransac_threshold is not None and len(matches) >= 4:
        src_points = kp1[matches[:, 0]][:, ::-1]
        dst_points = kp2[matches[:, 1]][:, ::-1]

        model, inliers = ransac(
            (src_points, dst_points),
            ProjectiveTransform,
            min_samples=4,
            residual_threshold=ransac_threshold,
            max_trials=1000,
        )

        if model is not None and inliers is not None:
            matches = matches[inliers]
    # Render side-by-side plot with rainbow match lines (matching PPT slide 4 & 5)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    plot_matched_features(
        gray1,
        gray2,
        keypoints0=kp1,
        keypoints1=kp2,
        matches=matches,
        ax=ax,
    )
    ax.set_title(f"{title1} <---> {title2} (matches={len(matches)})", fontsize=11)
    ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    result_image = Image.open(buf).convert("RGB")

    return kp1, kp2, matches, result_image


class FeatureMatchingDialog(simpledialog.Dialog):
    """Dialog for configuring feature matching and picking the second image."""

    def __init__(self, parent: tk.Misc, title: str = "Feature Matching") -> None:
        self.second_path = ""
        self.method = "ORB"
        self.cross_check = True
        self.max_ratio = 0.8
        self.n_keypoints = 500
        self.n_octaves = 8
        self.n_scales = 3
        self.sigma_min = 1.6
        self.result: dict | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        # File selector for Image 2
        ttk.Label(master, text="Second Image:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        file_frame = ttk.Frame(master)
        file_frame.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        self.path_entry = ttk.Entry(file_frame, width=28)
        self.path_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="Browse...", command=self._browse_image).pack(side="left", padx=(4, 0))

        # Extractor method
        ttk.Label(master, text="Method:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.method_combo = ttk.Combobox(
            master,
            values=["ORB", "SIFT"],
            state="readonly",
        )
        self.method_combo.set("ORB")
        self.method_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        self.method_combo.bind("<<ComboboxSelected>>", self._on_method_change)

        # Cross check
        self.cross_check_var = tk.BooleanVar(value=True)
        self.cross_check_cb = ttk.Checkbutton(
            master,
            text="Cross Check (mutual best matches)",
            variable=self.cross_check_var,
        )
        self.cross_check_cb.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        # Max ratio
        ttk.Label(master, text="Max Ratio (Lowe's ratio test):").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.max_ratio_entry = ttk.Entry(master)
        self.max_ratio_entry.insert(0, "0.8")
        self.max_ratio_entry.grid(row=3, column=1, sticky="ew", padx=6, pady=4)

        # ORB config
        self.n_keypoints_label = ttk.Label(master, text="ORB n_keypoints:")
        self.n_keypoints_label.grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self.n_keypoints_entry = ttk.Entry(master)
        self.n_keypoints_entry.insert(0, "500")
        self.n_keypoints_entry.grid(row=4, column=1, sticky="ew", padx=6, pady=4)

        # SIFT config (initially hidden)
        self.n_octaves_label = ttk.Label(master, text="SIFT n_octaves:")
        self.n_octaves_entry = ttk.Entry(master)
        self.n_octaves_entry.insert(0, "8")

        self.n_scales_label = ttk.Label(master, text="SIFT n_scales:")
        self.n_scales_entry = ttk.Entry(master)
        self.n_scales_entry.insert(0, "3")

        self.sigma_min_label = ttk.Label(master, text="SIFT sigma_min:")
        self.sigma_min_entry = ttk.Entry(master)
        self.sigma_min_entry.insert(0, "1.6")

        return self.path_entry

    def _browse_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select second image for matching",
            filetypes=[
                ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp")),
            ],
            parent=self,
        )
        if filename:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, filename)

    def _on_method_change(self, _event=None) -> None:
        method = self.method_combo.get()
        if method == "ORB":
            self.n_octaves_label.grid_remove()
            self.n_octaves_entry.grid_remove()
            self.n_scales_label.grid_remove()
            self.n_scales_entry.grid_remove()
            self.sigma_min_label.grid_remove()
            self.sigma_min_entry.grid_remove()

            self.n_keypoints_label.grid(row=4, column=0, sticky="w", padx=6, pady=4)
            self.n_keypoints_entry.grid(row=4, column=1, sticky="ew", padx=6, pady=4)
        else:
            self.n_keypoints_label.grid_remove()
            self.n_keypoints_entry.grid_remove()

            self.n_octaves_label.grid(row=4, column=0, sticky="w", padx=6, pady=4)
            self.n_octaves_entry.grid(row=4, column=1, sticky="ew", padx=6, pady=4)
            self.n_scales_label.grid(row=5, column=0, sticky="w", padx=6, pady=4)
            self.n_scales_entry.grid(row=5, column=1, sticky="ew", padx=6, pady=4)
            self.sigma_min_label.grid(row=6, column=0, sticky="w", padx=6, pady=4)
            self.sigma_min_entry.grid(row=6, column=1, sticky="ew", padx=6, pady=4)

    def validate(self) -> bool:
        second_path = self.path_entry.get().strip()
        if not second_path or not Path(second_path).is_file():
            messagebox.showerror("Invalid File", "Please select a valid second image file.", parent=self)
            return False

        try:
            ratio = float(self.max_ratio_entry.get())
            if not 0.0 < ratio <= 1.0:
                messagebox.showerror("Invalid Ratio", "Max ratio must be between 0.0 and 1.0.", parent=self)
                return False
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Max Ratio.", parent=self)
            return False

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
                int(self.n_octaves_entry.get())
                int(self.n_scales_entry.get())
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
        metric = "hamming" if method == "ORB" else "euclidean"

        self.result = {
            "second_path": self.path_entry.get().strip(),
            "method": method,
            "cross_check": self.cross_check_var.get(),
            "max_ratio": float(self.max_ratio_entry.get()),
            "metric": metric,
            "n_keypoints": int(self.n_keypoints_entry.get()) if method == "ORB" else 500,
            "n_octaves": int(self.n_octaves_entry.get()) if method == "SIFT" else 8,
            "n_scales": int(self.n_scales_entry.get()) if method == "SIFT" else 3,
            "sigma_min": float(self.sigma_min_entry.get()) if method == "SIFT" else 1.6,
        }


class FeatureMatchingTool(ForensicsTool):
    tool_id = "feature_matching"
    title = "Feature matching"
    category = "Set 5"
    description = "Match descriptors (ORB, SIFT) with a second image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        dialog = FeatureMatchingDialog(parent)
        if dialog.result is None:
            return None

        cfg = dialog.result
        second_path = Path(cfg["second_path"])
        try:
            with Image.open(second_path) as second_img:
                image2 = second_img.copy()
        except Exception as error:
            messagebox.showerror("Error opening second image", str(error), parent=parent)
            return None

        ref_name = document.path.name if document.path else "Reference"
        second_name = second_path.name

        method = cfg["method"]
        ransac_threshold = simpledialog.askfloat(
            "RANSAC",
            "Geometric error threshold in pixels.\n"
            "OK: apply RANSAC. Cancel: continue without RANSAC.",
            initialvalue=3.0,
            minvalue=0.001,
            parent=parent,
        )
        kp1, kp2, matches, visual = match_features_between(
            document.current,
            image2,
            method=method,
            cross_check=cfg["cross_check"],
            max_ratio=cfg["max_ratio"],
            metric=cfg["metric"],
            n_keypoints=cfg["n_keypoints"],
            n_octaves=cfg["n_octaves"],
            n_scales=cfg["n_scales"],
            sigma_min=cfg["sigma_min"],
            title1=ref_name,
            title2=second_name,
            ransac_threshold=ransac_threshold,
        )

        chosen_metric = cfg["metric"] or ("hamming" if method == "ORB" else "euclidean")
        details = {
            "Operation": "Feature matching",
            "Method": method,
            "Matches found": len(matches),
            "Keypoints (Image 1)": len(kp1),
            "Keypoints (Image 2)": len(kp2),
            "Second image": second_name,
            "Cross check": cfg["cross_check"],
            "Max ratio": cfg["max_ratio"],
            "Metric": chosen_metric,
        }

        return ToolResult(
            image=visual,
            message=f"Matched {len(matches)} features with {second_name} using {method}.",
            details=details,
        )
