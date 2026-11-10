"""Set 7.2 — BoW Histogram Tool & Ranked Visual Word Carousel.

Assigns SIFT keypoints from the loaded image to visual words using the
pre-trained KMeans codebook, and displays the image, its BoW frequency histogram,
and a horizontally scrollable carousel of visual words in rank order.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_utils import (
    TEST_DIR,
    build_bow_histogram,
    get_active_model_dir,
    load_kmeans,
    load_visual_words,
    prompt_codebook_selection,
    render_single_image_histogram_with_top_words,
)


class BoWHistogramWindow(tk.Toplevel):
    """Dialog window displaying a single image, its histogram, and a visual word carousel."""

    def __init__(
        self,
        parent: tk.Misc,
        image: Image.Image,
        image_name: str,
        kmeans,
        visual_words_records: list[dict],
    ) -> None:
        super().__init__(parent)
        self.title(f"BoW Histogram — {image_name}")
        self.geometry("980x700")
        self.minsize(740, 500)

        self.kmeans = kmeans
        self.visual_words_records = visual_words_records
        self.current_image = image
        self.current_name = image_name
        self.composite_image: Image.Image | None = None
        self._photo_cache: list[ImageTk.PhotoImage] = []

        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        # Header toolbar
        toolbar = ttk.Frame(self, padding=(12, 10))
        toolbar.pack(fill="x")

        ttk.Label(
            toolbar,
            text="BoW Histogram & Ranked Visual Words",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side="left")

        btn_box = ttk.Frame(toolbar)
        btn_box.pack(side="right")

        ttk.Button(
            btn_box,
            text="Change Codebook…",
            command=self._on_change_codebook,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            btn_box,
            text="Localize words on image…",
            command=self._on_localize,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            btn_box,
            text="Open another image…",
            command=self._on_choose_image,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            btn_box,
            text="Close",
            command=self.destroy,
        ).pack(side="left")

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Scrollable container for the window content
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, background="#ffffff", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.content_frame = ttk.Frame(self.canvas, padding=12)
        self.window_id = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        self.content_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.window_id, width=event.width),
        )

        self.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event) -> None:
        if self.canvas.winfo_exists():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_localize(self) -> None:
        from .bow_localize import VisualWordLocalizationWindow
        VisualWordLocalizationWindow(
            self,
            image=self.current_image.copy(),
            image_name=self.current_name,
            kmeans=self.kmeans,
            visual_words_records=self.visual_words_records,
        )

    def _on_change_codebook(self) -> None:
        chosen = prompt_codebook_selection(self, force_prompt=True)
        if chosen:
            try:
                self.kmeans = load_kmeans()
                self.visual_words_records = load_visual_words()
                self._refresh()
                messagebox.showinfo(
                    "Codebook Switched",
                    f"Switched to codebook '{chosen.name}'.",
                    parent=self,
                )
            except Exception as exc:
                messagebox.showerror("Error Loading Codebook", str(exc), parent=self)

    def _on_choose_image(self) -> None:
        initialdir = str(TEST_DIR) if TEST_DIR.exists() else None
        filename = filedialog.askopenfilename(
            parent=self,
            title="Select an image to inspect",
            initialdir=initialdir,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
        )
        if not filename:
            return

        try:
            self.current_image = Image.open(filename).convert("RGB")
            self.current_name = Path(filename).name
            self.title(f"BoW Histogram — {self.current_name}")
            self._refresh()
        except Exception as exc:
            messagebox.showerror("Error opening image", str(exc), parent=self)

    def _refresh(self) -> None:
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self._photo_cache.clear()

        # 1. Compute BoW histogram
        hist = build_bow_histogram(self.current_image, self.kmeans)

        # 2. Render clean side-by-side [Image | Histogram] (no rank numbers on bars)
        composite, top5_info = render_single_image_histogram_with_top_words(
            self.current_image,
            hist,
            self.visual_words_records,
            image_name=self.current_name,
        )
        self.composite_image = composite

        # Scale composite image to fit width nicely
        target_w = max(680, min(940, composite.width))
        aspect = composite.height / composite.width
        target_h = int(target_w * aspect)
        scaled_composite = composite.resize((target_w, target_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(scaled_composite)
        self._photo_cache.append(photo)

        lbl = ttk.Label(self.content_frame, image=photo)
        lbl.pack(pady=(0, 10))

        # 3. Horizontally Scrollable Carousel displaying Visual Words in Rank Order
        ranked_indices = np.argsort(hist)[::-1]
        active_indices = [int(i) for i in ranked_indices if hist[i] > 0]
        if not active_indices:
            active_indices = [int(i) for i in ranked_indices[:10]]

        carousel_frame = ttk.LabelFrame(
            self.content_frame,
            text=f"Visual Words in Rank Order ({len(active_indices)} active words in this image)",
            padding=8,
        )
        carousel_frame.pack(fill="x", pady=(6, 10))

        # Carousel Canvas + Horizontal Scrollbar
        carousel_container = ttk.Frame(carousel_frame)
        carousel_container.pack(fill="x", expand=True)

        carousel_canvas = tk.Canvas(
            carousel_container,
            height=160,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#e2e8f0",
        )
        h_scrollbar = ttk.Scrollbar(
            carousel_container,
            orient="horizontal",
            command=carousel_canvas.xview,
        )
        carousel_canvas.configure(xscrollcommand=h_scrollbar.set)

        carousel_canvas.pack(fill="x", expand=True)
        h_scrollbar.pack(fill="x", pady=(4, 0))

        carousel_inner = ttk.Frame(carousel_canvas, padding=4)
        carousel_canvas.create_window((0, 0), window=carousel_inner, anchor="nw")

        carousel_inner.bind(
            "<Configure>",
            lambda _e: carousel_canvas.configure(scrollregion=carousel_canvas.bbox("all")),
        )

        # Support MouseWheel for horizontal scrolling over the carousel
        def _on_horizontal_wheel(event) -> None:
            if carousel_canvas.winfo_exists():
                carousel_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        carousel_canvas.bind("<MouseWheel>", _on_horizontal_wheel)
        carousel_inner.bind("<MouseWheel>", _on_horizontal_wheel)

        # Populate carousel with ranked visual word cards
        for rank, vw_idx in enumerate(active_indices, start=1):
            freq = float(hist[vw_idx])
            rec = (
                self.visual_words_records[vw_idx]
                if vw_idx < len(self.visual_words_records)
                else {}
            )

            card = ttk.Frame(carousel_inner, relief="solid", padding=6)
            card.pack(side="left", padx=5, pady=4)

            # Rank badge (Top 5 highlighted with distinct color)
            rank_color = "#b45309" if rank <= 5 else "#374151"
            ttk.Label(
                card,
                text=f"Rank #{rank}",
                font=("TkDefaultFont", 9, "bold"),
                foreground=rank_color,
            ).pack(anchor="center")

            # Visual Word Label
            ttk.Label(
                card,
                text=f"vw {vw_idx}",
                font=("TkDefaultFont", 10, "bold"),
                foreground="#1d4ed8",
            ).pack(anchor="center")

            # Representative Image Patch
            if rec and "patch" in rec:
                scaled_patch = rec["patch"].resize((56, 56), Image.Resampling.NEAREST)
                photo_patch = ImageTk.PhotoImage(scaled_patch)
                self._photo_cache.append(photo_patch)
                ttk.Label(card, image=photo_patch, relief="groove").pack(
                    anchor="center", pady=3
                )

            # Frequency labels
            ttk.Label(
                card,
                text=f"freq: {freq:.4f}",
                font=("Consolas", 8),
            ).pack(anchor="center")

            ttk.Label(
                card,
                text=f"({freq * 100:.1f}%)",
                font=("Consolas", 8),
                foreground="#6b7280",
            ).pack(anchor="center")


class BoWHistogramTool(ForensicsTool):
    """BoW Histogram Tool."""

    tool_id = "bow_histogram"
    title = "BoW Histogram"
    category = "Set 7"
    description = (
        "Display the BoW frequency histogram and a horizontal carousel of visual words "
        "in rank order for the current image."
    )
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # Offer user choice of default or custom codebook
        chosen_dir = prompt_codebook_selection(parent)
        if chosen_dir is None:
            return None

        try:
            kmeans = load_kmeans()
            visual_words_records = load_visual_words()
        except FileNotFoundError as exc:
            messagebox.showerror("Models not found", str(exc), parent=parent)
            return None

        # Build histogram for the current working image
        hist = build_bow_histogram(document.current, kmeans)
        name = document.path.name if document.path else "Current image"

        # Open dedicated dialog window
        dialog = BoWHistogramWindow(
            parent,
            image=document.current.copy(),
            image_name=name,
            kmeans=kmeans,
            visual_words_records=visual_words_records,
        )

        composite = dialog.composite_image or render_single_image_histogram_with_top_words(
            document.current,
            hist,
            visual_words_records,
            image_name=name,
        )[0]

        ranked_indices = np.argsort(hist)[::-1]
        top5_indices = ranked_indices[:5]
        non_zero = int(np.count_nonzero(hist))

        details = {
            "Operation": "BoW Histogram & Ranked Visual Words",
            "Image": name,
            "Active visual words": f"{non_zero}/{kmeans.n_clusters}",
        }
        for rank, idx in enumerate(top5_indices, start=1):
            details[f"Rank #{rank}"] = f"vw {idx} (frequency: {hist[idx]:.4f}, {hist[idx]*100:.1f}%)"

        top_word = int(top5_indices[0])
        # Do not replace current image in main window: return image=None
        return ToolResult(
            image=None,
            message=(
                f"BoW histogram for {name}: dominant word = vw {top_word} "
                f"({hist[top_word]*100:.1f}% frequency)."
            ),
            details=details,
        )
