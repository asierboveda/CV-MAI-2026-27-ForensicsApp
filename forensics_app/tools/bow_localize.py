"""Set 7 — Visual Word Spatial Localization.

Displays where a selected visual word occurs across the image by drawing
small pinpoint dots on matching SIFT keypoints, alongside the visual word's
representative feature patch.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .bow_utils import (
    extract_keypoints_and_assignments,
    load_kmeans,
    load_visual_words,
)


class VisualWordLocalizationWindow(tk.Toplevel):
    """Simple dialog to select a visual word and view its feature patch & locations."""

    def __init__(
        self,
        parent: tk.Misc,
        image: Image.Image,
        image_name: str,
        kmeans,
        visual_words_records: list[dict],
        initial_word: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.title(f"Visual Word Localization — {image_name}")
        self.geometry("960x650")
        self.minsize(720, 480)

        self.current_image = image.convert("RGB")
        self.current_name = image_name
        self.kmeans = kmeans
        self.visual_words_records = visual_words_records
        self.k = kmeans.n_clusters

        # Extract SIFT features & cluster assignments
        self.keypoints, self.descriptors, self.assignments = (
            extract_keypoints_and_assignments(self.current_image, self.kmeans)
        )

        # Count occurrences for all words 0 .. K-1
        self.counts = {
            w: int(np.count_nonzero(self.assignments == w))
            for w in range(self.k)
        }

        # Determine initial selection
        if initial_word is not None and 0 <= initial_word < self.k:
            default_vw = initial_word
        elif len(self.assignments) > 0:
            default_vw = int(max(self.counts, key=self.counts.get))
        else:
            default_vw = 0

        self.vw_var = tk.StringVar(value=f"vw {default_vw}")
        self._photo_cache: list[ImageTk.PhotoImage] = []

        self._build_ui()
        self._on_vw_changed()

    def _build_ui(self) -> None:
        # Header toolbar
        toolbar = ttk.Frame(self, padding=(12, 10))
        toolbar.pack(fill="x")

        ttk.Label(
            toolbar,
            text=f"Visual Word Spatial Localization — {self.current_name}",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side="left")

        ttk.Button(
            toolbar,
            text="Close",
            command=self.destroy,
        ).pack(side="right")

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Control strip: simple dropdown
        controls = ttk.Frame(self, padding=(12, 8))
        controls.pack(fill="x")

        ttk.Label(
            controls,
            text="Select visual word:",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side="left", padx=(0, 8))

        vw_options = [f"vw {i}" for i in range(self.k)]
        self.vw_cb = ttk.Combobox(
            controls,
            textvariable=self.vw_var,
            values=vw_options,
            state="readonly",
            width=10,
        )
        self.vw_cb.pack(side="left", padx=(0, 16))
        self.vw_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_vw_changed())

        self.quick_info = ttk.Label(
            controls,
            text="",
            font=("TkDefaultFont", 9),
        )
        self.quick_info.pack(side="left")

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Main body: Left Sidebar (Feature Info) + Right Canvas (Image with Dots)
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)

        # Left sidebar: Feature Correspondence Card
        sidebar = ttk.Frame(body, width=220)
        sidebar.pack(side="left", fill="y", padx=(0, 10))
        sidebar.pack_propagate(False)

        card = ttk.LabelFrame(sidebar, text="Feature Correspondence", padding=10)
        card.pack(fill="both", expand=True)

        self.lbl_vw_title = ttk.Label(
            card,
            text="vw 0",
            font=("TkDefaultFont", 12, "bold"),
            foreground="#1d4ed8",
        )
        self.lbl_vw_title.pack(anchor="w", pady=(0, 4))

        ttk.Label(
            card,
            text="Representative Patch:",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(anchor="w", pady=(6, 2))

        self.lbl_patch = ttk.Label(card, relief="groove")
        self.lbl_patch.pack(anchor="center", pady=6)

        self.lbl_count = ttk.Label(
            card,
            text="",
            font=("TkDefaultFont", 9),
            justify="left",
        )
        self.lbl_count.pack(anchor="w", pady=(6, 10))

        ttk.Separator(card, orient="horizontal").pack(fill="x", pady=6)

        ttk.Label(
            card,
            text="Red dots on the image pinpoint where this visual word occurs.",
            font=("TkDefaultFont", 8),
            foreground="#555555",
            wraplength=180,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        # Right: Scrollable Image Canvas
        img_container = ttk.Frame(body)
        img_container.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(img_container, background="#ffffff", highlightthickness=1, highlightbackground="#e2e8f0")
        self.scrollbar_y = ttk.Scrollbar(img_container, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(img_container, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)

        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.lbl_canvas_img = ttk.Label(self.canvas)
        self.canvas_window_id = self.canvas.create_window((0, 0), window=self.lbl_canvas_img, anchor="nw")

        self.lbl_canvas_img.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

    def _get_selected_vw_index(self) -> int:
        val = self.vw_var.get().strip()
        try:
            return int(val.replace("vw", "").strip())
        except ValueError:
            return 0

    def _on_vw_changed(self) -> None:
        self._photo_cache.clear()
        vw_idx = self._get_selected_vw_index()
        count = self.counts.get(vw_idx, 0)
        total_kps = max(1, len(self.keypoints))
        pct = (count / total_kps) * 100

        # 1. Update Left Card Info
        self.lbl_vw_title.configure(text=f"Visual Word: vw {vw_idx}")
        self.lbl_count.configure(
            text=(
                f"Occurrences: {count} keypoints\n"
                f"Share: {pct:.1f}% of features\n"
                f"Total image keypoints: {len(self.keypoints)}"
            )
        )
        self.quick_info.configure(
            text=f"Found {count} keypoints matching vw {vw_idx} ({pct:.1f}%)"
        )

        # 2. Update Representative Patch thumbnail
        rec = self.visual_words_records[vw_idx] if vw_idx < len(self.visual_words_records) else {}
        if rec and "patch" in rec:
            patch_img = rec["patch"].resize((76, 76), Image.Resampling.NEAREST)
            photo_patch = ImageTk.PhotoImage(patch_img)
            self._photo_cache.append(photo_patch)
            self.lbl_patch.configure(image=photo_patch)
        else:
            self.lbl_patch.configure(text="[No patch]", image="")

        # 3. Draw image with small pinpoint dots for this visual word
        annotated = self.current_image.copy()
        draw = ImageDraw.Draw(annotated)
        radius = 2.5  # Small dot size

        for idx, (row, col) in enumerate(self.keypoints):
            if self.assignments[idx] == vw_idx:
                draw.ellipse(
                    [(col - radius, row - radius), (col + radius, row + radius)],
                    fill="#ef4444",
                    outline="#991b1b",
                    width=1,
                )

        photo_img = ImageTk.PhotoImage(annotated)
        self._photo_cache.append(photo_img)
        self.lbl_canvas_img.configure(image=photo_img)
