"""Set 7.3 — BoW Classification Tool.

Workflow
--------
1. Build the BoW histogram for the loaded image (identical to Set 7.2).
2. Compute the chi-squared kernel row: K(query, X_train).
3. Call clf.predict_proba → ranked class probabilities.
4. Return annotated histogram image with 'p(class) = prob' overlay.

The classifier is a Support Vector Machine trained with a precomputed
chi-squared kernel — the standard choice for histogram-based image features
because it correctly handles near-zero bins and multinomial distributions.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import numpy as np
from sklearn.metrics.pairwise import chi2_kernel

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_utils import (
    build_bow_histogram,
    load_classifier,
    load_classes,
    load_kmeans,
    load_train_histograms,
    render_classification_result,
)


class BoWClassifyTool(ForensicsTool):
    """Set 7.3 — Classify image using BoW + chi-squared kernel SVM."""

    tool_id = "bow_classify"
    title = "Classify image (BoW)"
    category = "Set 7"
    description = (
        "Show the BoW histogram and classify the image using a "
        "chi-squared kernel SVM trained on Caltech-101 subset."
    )
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # Load all models (each cached after first call)
        try:
            kmeans  = load_kmeans()
            clf     = load_classifier()
            X_train = load_train_histograms()
            classes = load_classes()
        except FileNotFoundError as exc:
            messagebox.showerror("Models not found", str(exc), parent=parent)
            return None

        # -------------------------------------------------------------------
        # Step 1: Build BoW histogram (same as Set 7.2)
        # -------------------------------------------------------------------
        hist = build_bow_histogram(document.current, kmeans)   # shape (K,)

        if hist.sum() == 0:
            messagebox.showwarning(
                "No keypoints detected",
                "SIFT found no keypoints — cannot classify this image.",
                parent=parent,
            )
            return None

        # -------------------------------------------------------------------
        # Step 2: Compute chi-squared kernel row  K(query, X_train)
        #
        # chi²(h, g) = Σ_i  (h_i - g_i)² / (h_i + g_i)
        # K_chi²(h, g) = exp(-γ · chi²(h, g))
        #
        # Because SVC was trained with kernel='precomputed', prediction also
        # requires the precomputed kernel values against the support vectors
        # (sklearn handles this internally — we just supply K_test).
        # -------------------------------------------------------------------
        K_test = chi2_kernel(hist.reshape(1, -1), X_train)   # shape (1, N_train)

        # -------------------------------------------------------------------
        # Step 3: Predict
        # -------------------------------------------------------------------
        proba     = clf.predict_proba(K_test)[0]       # shape (n_classes,)
        top_idx   = int(np.argmax(proba))
        top_label = classes[top_idx]
        top_prob  = float(proba[top_idx])

        annotation = f"p({top_label}) = {top_prob:.2f}"

        # -------------------------------------------------------------------
        # Step 4: Render side-by-side classification view (image + histogram)
        # -------------------------------------------------------------------
        chart = render_classification_result(
            document.current,
            hist,
            top_label=top_label,
            top_prob=top_prob,
        )

        # Build per-class probability rows for the details panel
        prob_details = {
            f"p({cls})": f"{p:.4f}"
            for cls, p in zip(classes, proba)
        }

        return ToolResult(
            image=chart,
            message=annotation,
            details={
                "Operation": "BoW Classification",
                "Predicted class": top_label,
                "Confidence": f"{top_prob:.4f}",
                **prob_details,
                "Visual words (K)": kmeans.n_clusters,
                "Non-zero bins": int(np.count_nonzero(hist)),
                "Kernel": "Chi-squared (precomputed)",
            },
        )

