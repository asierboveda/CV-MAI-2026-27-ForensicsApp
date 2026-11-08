"""Set 7.1 — Visual Words Codebook Viewer.

Displays the 5×6 overview grid of the 30 visual words (KMeans cluster centroids)
and their representative image patches (vw 0 .. vw 29).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_utils import (
    load_visual_words,
    load_visual_words_grid,
)


class VisualWordsWindow(tk.Toplevel):
    """Window displaying the 5×6 overview grid of all 30 visual words."""

    def __init__(self, parent: tk.Misc, grid_image: Image.Image, num_words: int = 30) -> None:
        super().__init__(parent)
        self.title("Visual Words Codebook (Overview Grid) — Set 7.1")
        self.geometry("860x720")
        self.minsize(620, 480)

        self.grid_image = grid_image
        self.num_words = num_words
        self._photo_ref: ImageTk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        # Header banner
        header = ttk.Frame(self, padding=(14, 12))
        header.pack(fill="x")

        ttk.Label(
            header,
            text=f"Visual Words Codebook (K = {self.num_words}) — Overview Grid",
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            header,
            text=(
                "Each visual word represents a KMeans cluster centroid in SIFT feature space (128-D).\n"
                "The 5 × 6 grid below shows the representative image patch closest to each centroid (vw 0 … vw 29)."
            ),
            font=("TkDefaultFont", 9),
            foreground="#555555",
        ).pack(anchor="w", pady=(3, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Scrollable container for the grid
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, background="#ffffff", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.image_label = ttk.Label(self.canvas, background="#ffffff")
        self.window_id = self.canvas.create_window((0, 0), window=self.image_label, anchor="nw")

        self.image_label.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            self._on_canvas_resize,
        )

        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self._render_grid()

    def _on_mousewheel(self, event) -> None:
        if self.canvas.winfo_exists():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_canvas_resize(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _render_grid(self) -> None:
        # Scale to fit width while maintaining crisp visual quality
        target_w = max(600, min(1000, self.grid_image.width))
        aspect = self.grid_image.height / self.grid_image.width
        target_h = int(target_w * aspect)

        scaled = self.grid_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        self._photo_ref = ImageTk.PhotoImage(scaled)
        self.image_label.configure(image=self._photo_ref)


class VisualWordsTool(ForensicsTool):
    """Set 7.1 — View the visual words codebook overview grid."""

    tool_id = "bow_visual_words"
    title = "View Visual Words"
    category = "Set 7"
    description = "View the 5 × 6 visual words codebook overview grid (Set 7.1)."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        try:
            records = load_visual_words()
            grid_img = load_visual_words_grid()
        except FileNotFoundError as exc:
            tk.messagebox.showerror("Models not found", str(exc), parent=parent)
            return None

        # Open overview grid dialog
        VisualWordsWindow(parent, grid_img, num_words=len(records))

        return ToolResult(
            image=grid_img,
            message=f"Displayed {len(records)} visual words in 5 × 6 overview grid.",
            details={
                "Operation": "Visual Words Codebook",
                "Visual words (K)": len(records),
                "Clustering method": "KMeans (Euclidean distance)",
                "Descriptor": "SIFT (128-D)",
                "Layout": "5 × 6 Overview Grid",
            },
        )
