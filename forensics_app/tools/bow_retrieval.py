"""Set 7 Extension — Content-Based Image Retrieval (CBIR).

Retrieves the most similar images from the Caltech-101 subset database using
Bag-of-Visual-Words histogram matching (Chi-squared, Histogram Intersection,
or Cosine similarity).
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_utils import (
    build_bow_histogram,
    compute_histogram_similarities,
    load_classes,
    load_classifier,
    load_kmeans,
    load_train_histograms,
    load_train_image_records,
    render_retrieval_composite,
)


class BoWRetrievalWindow(tk.Toplevel):
    """Interactive visual search window displaying retrieved matches."""

    def __init__(
        self,
        parent: tk.Misc,
        query_image: Image.Image,
        query_name: str,
        kmeans,
        train_hists: np.ndarray,
        train_records: list[tuple[Path, str]],
        classes: list[str],
        document: ImageDocument | None = None,
    ) -> None:
        super().__init__(parent)
        self.title(f"CBIR: Similar Images — {query_name}")
        self.geometry("1100x700")
        self.minsize(800, 500)

        self.query_image = query_image
        self.query_name = query_name
        self.kmeans = kmeans
        self.train_hists = train_hists
        self.train_records = train_records
        self.classes = classes
        self.document = document

        # Compute query histogram
        self.query_hist = build_bow_histogram(self.query_image, self.kmeans)

        # UI state variables
        self.metric_var = tk.StringVar(value="Chi-squared")
        self.top_k_var = tk.IntVar(value=5)

        self._photo_cache: list[ImageTk.PhotoImage] = []
        self._build_ui()
        self._run_search()

    def _build_ui(self) -> None:
        # Header toolbar
        toolbar = ttk.Frame(self, padding=(12, 10))
        toolbar.pack(fill="x")

        ttk.Label(
            toolbar,
            text="Content-Based Image Retrieval (CBIR) - Visual Search",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side="left")

        ttk.Button(
            toolbar,
            text="Close",
            command=self.destroy,
        ).pack(side="right")

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Controls & Search Configuration
        controls = ttk.Frame(self, padding=(12, 8))
        controls.pack(fill="x")

        ttk.Label(controls, text="Similarity metric:", font=("TkDefaultFont", 9, "bold")).pack(
            side="left", padx=(0, 6)
        )
        metric_cb = ttk.Combobox(
            controls,
            textvariable=self.metric_var,
            values=["Chi-squared", "Histogram Intersection", "Cosine"],
            state="readonly",
            width=22,
        )
        metric_cb.pack(side="left", padx=(0, 16))
        metric_cb.bind("<<ComboboxSelected>>", lambda _e: self._run_search())

        ttk.Label(controls, text="Top matches:").pack(side="left", padx=(0, 6))
        top_k_cb = ttk.Combobox(
            controls,
            textvariable=self.top_k_var,
            values=[3, 5, 8],
            state="readonly",
            width=4,
        )
        top_k_cb.pack(side="left", padx=(0, 16))
        top_k_cb.bind("<<ComboboxSelected>>", lambda _e: self._run_search())

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Scrollable container for results gallery
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, background="#f8fafc", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.results_frame = ttk.Frame(self.canvas, padding=12)
        self.window_id = self.canvas.create_window((0, 0), window=self.results_frame, anchor="nw")

        self.results_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.window_id, width=event.width),
        )

    def _run_search(self) -> None:
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self._photo_cache.clear()

        metric_map = {
            "Chi-squared": "chi2",
            "Histogram Intersection": "intersection",
            "Cosine": "cosine",
        }
        metric_key = metric_map.get(self.metric_var.get(), "chi2")
        scores = compute_histogram_similarities(self.query_hist, self.train_hists, metric=metric_key)

        ranked_indices = np.argsort(scores)[::-1]
        K_matches = min(int(self.top_k_var.get()), len(ranked_indices))
        top_indices = ranked_indices[:K_matches]

        # Top section: Query Summary Box
        query_box = ttk.LabelFrame(self.results_frame, text="Query Image", padding=10)
        query_box.pack(fill="x", pady=(0, 12))

        q_thumb = self.query_image.copy()
        q_thumb.thumbnail((120, 120), Image.Resampling.LANCZOS)
        photo_q = ImageTk.PhotoImage(q_thumb)
        self._photo_cache.append(photo_q)

        ttk.Label(query_box, image=photo_q).pack(side="left", padx=(0, 14))

        q_info = ttk.Frame(query_box)
        q_info.pack(side="left", fill="both", expand=True)

        ttk.Label(
            q_info,
            text=f"Filename: {self.query_name}",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w")

        # Results Grid Cards
        grid_frame = ttk.Frame(self.results_frame)
        grid_frame.pack(fill="both", expand=True)

        for col_idx, db_idx in enumerate(top_indices):
            path, cls_name = self.train_records[db_idx]
            sim_score = float(scores[db_idx])

            card = ttk.Frame(grid_frame, relief="solid", padding=8)
            card.grid(row=0, column=col_idx, padx=6, pady=6, sticky="n")

            # Rank Badge
            rank_color = "#b45309" if col_idx == 0 else "#2563eb"
            ttk.Label(
                card,
                text=f"Rank #{col_idx + 1}",
                font=("TkDefaultFont", 11, "bold"),
                foreground=rank_color,
            ).pack(anchor="center")

            # Image thumbnail
            try:
                db_img = Image.open(path).convert("RGB")
                db_thumb = db_img.copy()
                db_thumb.thumbnail((120, 120), Image.Resampling.LANCZOS)
                photo_db = ImageTk.PhotoImage(db_thumb)
                self._photo_cache.append(photo_db)
                ttk.Label(card, image=photo_db, relief="groove").pack(
                    anchor="center", pady=6
                )
            except Exception:
                ttk.Label(card, text="[Preview unavailable]").pack(anchor="center", pady=20)

            # Class badge
            ttk.Label(
                card,
                text=f"[{cls_name}]",
                font=("TkDefaultFont", 9, "bold"),
                foreground="#16a34a",
            ).pack(anchor="center")

            # Filename
            ttk.Label(
                card,
                text=path.name,
                font=("Consolas", 8),
                foreground="#555555",
            ).pack(anchor="center")

            # Similarity Score
            ttk.Label(
                card,
                text=f"Similarity: {sim_score * 100:.1f}%",
                font=("TkDefaultFont", 9, "bold"),
                foreground="#1e293b",
            ).pack(anchor="center", pady=(4, 6))

    def _load_image_into_document(self, path: Path) -> None:
        if self.document is not None:
            self.document.load(path)
            messagebox.showinfo(
                "Image Loaded",
                f"Loaded '{path.name}' into the main application workspace.",
                parent=self,
            )


class BoWRetrievalTool(ForensicsTool):
    """Set 7 Extension — Content-Based Image Retrieval (CBIR) Tool."""

    tool_id = "bow_retrieval"
    title = "Find similar images (CBIR)"
    category = "Set 7"
    description = (
        "Retrieve the most similar images from the dataset using Bag-of-Visual-Words "
        "histogram matching."
    )
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        try:
            kmeans = load_kmeans()
            train_hists = load_train_histograms()
            train_records = load_train_image_records()
            classes = load_classes()
        except FileNotFoundError as exc:
            messagebox.showerror("Models not found", str(exc), parent=parent)
            return None

        name = document.path.name if document.path else "Current image"

        # Compute query BoW histogram
        query_hist = build_bow_histogram(document.current, kmeans)
        if query_hist.sum() == 0:
            messagebox.showwarning(
                "No features",
                "SIFT found no keypoints — cannot retrieve similar images.",
                parent=parent,
            )
            return None

        # Compute Chi-squared similarity scores across all database histograms
        scores = compute_histogram_similarities(query_hist, train_hists, metric="chi2")
        ranked_indices = np.argsort(scores)[::-1]
        top_k = min(5, len(ranked_indices))

        top_matches: list[dict] = []
        details: dict[str, str] = {
            "Operation": "Content-Based Image Retrieval (CBIR)",
            "Query image": name,
            "Metric": "Chi-squared similarity",
            "Database images searched": str(len(train_records)),
        }

        for rank, db_idx in enumerate(ranked_indices[:top_k], start=1):
            path, cls_name = train_records[db_idx]
            sim = float(scores[db_idx])
            top_matches.append({
                "rank": rank,
                "path": path,
                "class": cls_name,
                "similarity": sim,
            })
            details[f"Match #{rank}"] = f"[{cls_name}] {path.name} (similarity: {sim*100:.1f}%)"

        # Render composite comparison figure
        composite = render_retrieval_composite(
            document.current,
            query_name=name,
            query_hist=query_hist,
            top_matches=top_matches,
            metric_name="Chi-squared",
        )

        # Open interactive search dialog
        BoWRetrievalWindow(
            parent,
            query_image=document.current.copy(),
            query_name=name,
            kmeans=kmeans,
            train_hists=train_hists,
            train_records=train_records,
            classes=classes,
            document=document,
        )

        top_match = top_matches[0]
        return ToolResult(
            image=None,
            message=(
                f"CBIR search complete: Top match is [{top_match['class']}] {top_match['path'].name} "
                f"({top_match['similarity']*100:.1f}% similarity)."
            ),
            details=details,
        )

