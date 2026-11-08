"""Shared Bag-of-Visual-Words helpers for Set 7.

Loaded models are cached at module level so they are only read from disk once
per application session.  All Set 7 tools import from here.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import chi2_kernel

from .descriptor_extractor import extract_descriptors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parents[1] / "models"
DATASET_DIR = Path(__file__).parents[1] / "images" / "Caltech_101_subset"
TEST_DIR = DATASET_DIR / "test"

# Module-level cache — populated lazily on first use
_cache: dict[str, Any] = {}


def _load(name: str) -> Any:
    """Load a pickled artefact from MODELS_DIR with module-level caching."""
    if name not in _cache:
        path = MODELS_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n\n"
                "Run the setup script first:\n"
                "  .venv\\Scripts\\python.exe forensics_app/scripts/build_bow.py"
            )
        with open(path, "rb") as f:
            _cache[name] = pickle.load(f)
    return _cache[name]


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_kmeans() -> KMeans:
    """Return the fitted KMeans codebook (K=30 visual words)."""
    return _load("kmeans_model.pkl")


def load_classifier():
    """Return the fitted SVM with chi-squared precomputed kernel."""
    return _load("classifier.pkl")


def load_train_histograms() -> np.ndarray:
    """Return training BoW histograms, shape (N_train, K)."""
    return _load("train_histograms.pkl")


def load_classes() -> list[str]:
    """Return ordered list of class names matching label indices."""
    return _load("label_classes.pkl")


def load_train_image_records() -> list[tuple[Path, str]]:
    """Return ordered list of (image_path, class_label) for training set.

    Matches exactly the 193 training images and order stored in train_histograms.pkl.
    """
    if "train_image_records" not in _cache:
        classes = load_classes()
        records: list[tuple[Path, str]] = []
        train_dir = DATASET_DIR / "train"
        for cls in classes:
            folder = train_dir / cls
            if folder.exists():
                for p in sorted(folder.iterdir()):
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                        records.append((p, cls))
        _cache["train_image_records"] = records
    return _cache["train_image_records"]


def load_visual_words() -> list[dict]:
    """Return list of 30 dicts for visual words vw 0 .. vw 29."""
    return _load("visual_words.pkl")


def load_visual_words_grid() -> Image.Image:
    """Return the overview 5x6 visual words grid image."""
    grid_path = MODELS_DIR / "visual_words_grid.png"
    if not grid_path.exists():
        raise FileNotFoundError(f"Visual words grid image not found at {grid_path}")
    return Image.open(grid_path).convert("RGB")


# ---------------------------------------------------------------------------
# Core BoW computation
# ---------------------------------------------------------------------------

def build_bow_histogram(image: Image.Image, kmeans: KMeans) -> np.ndarray:
    """Extract SIFT descriptors, assign to nearest centroid (Euclidean via
    KMeans.predict), and return an L1-normalised frequency histogram.

    Parameters
    ----------
    image  : PIL Image — converted to RGB internally.
    kmeans : Fitted KMeans codebook.

    Returns
    -------
    hist : np.ndarray, shape (K,), L1-normalised (sums to 1.0).
           All-zeros array if SIFT finds no keypoints.
    """
    K = kmeans.n_clusters
    _, descriptors, _ = extract_descriptors(image.convert("RGB"), method="SIFT")

    if len(descriptors) == 0:
        return np.zeros(K, dtype=np.float64)

    # kmeans.predict uses Euclidean distance to assign each descriptor to its
    # nearest visual word (cluster centroid).
    assignments = kmeans.predict(descriptors.astype(np.float64))

    hist, _ = np.histogram(assignments, bins=np.arange(K + 1))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total  # L1 normalise
    return hist


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def render_histogram(
    hist: np.ndarray,
    title: str = "Histogram of visual words",
    annotation: str | None = None,
) -> Image.Image:
    """Render a BoW frequency histogram as a Matplotlib bar chart."""
    K = len(hist)
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)

    ax.bar(
        np.arange(K),
        hist,
        color="#4c8cbf",
        edgecolor="#2b5c8f",
        linewidth=0.5,
        width=0.8,
    )
    ax.set_xlabel("Visual word index", fontsize=10)
    ax.set_ylabel("Normalized frequency", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_xticks(np.arange(0, K + 1, 5))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    if annotation:
        ax.text(
            0.98, 0.95,
            annotation,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=13, fontweight="bold",
            color="#d62728",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_single_image_histogram_with_top_words(
    image: Image.Image,
    hist: np.ndarray,
    visual_words_records: list[dict],
    image_name: str = "Image",
) -> tuple[Image.Image, list[tuple[int, int, float, dict]]]:
    """Render a single image alongside its BoW frequency histogram (matching Slide 3),
    with top 5 visual words highlighted in gold (without cluttering text above bars).

    Parameters
    ----------
    image                : PIL Image
    hist                 : np.ndarray shape (30,)
    visual_words_records : list of dicts for vw 0 .. vw 29
    image_name           : Label for the image

    Returns
    -------
    (composite_image, top5_info)
    top5_info is a list of tuples: (rank, vw_index, frequency, record_dict)
    """
    K = len(hist)
    ranked_indices = np.argsort(hist)[::-1]
    top5_indices = ranked_indices[:5]
    top5_info = []

    for rank, idx in enumerate(top5_indices, start=1):
        freq = float(hist[idx])
        rec = visual_words_records[idx] if idx < len(visual_words_records) else {}
        top5_info.append((rank, int(idx), freq, rec))

    fig, (ax_img, ax_hist) = plt.subplots(
        1, 2, figsize=(10, 4.2), dpi=100, gridspec_kw={"width_ratios": [1, 1.4]}
    )

    # Left: Image
    ax_img.imshow(image.convert("RGB"))
    ax_img.set_title(image_name, fontsize=11, fontweight="bold")
    ax_img.axis("off")

    # Right: Clean Histogram (Uniform bars matching Slide 3, no gold, no rank numbers)
    ax_hist.bar(
        np.arange(K),
        hist,
        color="#4c8cbf",
        edgecolor="#1f2937",
        linewidth=0.6,
        width=0.8,
    )
    ax_hist.set_xlabel("Visual word index", fontsize=10)
    ax_hist.set_ylabel("Normalized frequency", fontsize=10)
    ax_hist.set_title("Histogram of visual words", fontsize=11, fontweight="bold")
    ax_hist.set_xlim(-0.5, K - 0.5)
    ax_hist.set_xticks(np.arange(0, K + 1, 5))
    ax_hist.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    composite = Image.open(buf).convert("RGB")
    return composite, top5_info


def render_classification_result(
    image: Image.Image,
    hist: np.ndarray,
    top_label: str,
    top_prob: float,
) -> Image.Image:
    """Render side-by-side classification view (matching Slide 4 & 1).

    Left: The query image with title 'p(class) = prob'.
    Right: The BoW frequency histogram (K=30).
    """
    K = len(hist)
    fig, (ax_img, ax_hist) = plt.subplots(
        1, 2, figsize=(10, 4.2), dpi=100, gridspec_kw={"width_ratios": [1, 1.4]}
    )

    # Left: Query image
    ax_img.imshow(image.convert("RGB"))
    ax_img.set_title(f"p({top_label}) = {top_prob:.2f}", fontsize=13, fontweight="bold", pad=8)
    ax_img.axis("off")

    # Right: Histogram
    ax_hist.bar(
        np.arange(K),
        hist,
        color="#4c8cbf",
        edgecolor="#2b5c8f",
        linewidth=0.5,
        width=0.8,
    )
    ax_hist.set_xlabel("Visual word index", fontsize=10)
    ax_hist.set_ylabel("Normalized frequency", fontsize=10)
    ax_hist.set_title("Histogram of visual words", fontsize=11, fontweight="bold")
    ax_hist.set_xlim(-0.5, K - 0.5)
    ax_hist.set_xticks(np.arange(0, K + 1, 5))
    ax_hist.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Set 7 Extensions: Spatial Localization & CBIR
# ---------------------------------------------------------------------------

# Palette of distinct, vibrant colors for highlighting visual words
VW_COLORS: list[tuple[int, int, int]] = [
    (220, 38, 38),    # Red
    (37, 99, 235),    # Blue
    (22, 163, 74),    # Green
    (217, 119, 6),    # Amber / Gold
    (147, 51, 234),   # Purple
    (13, 148, 136),   # Teal
    (219, 39, 119),   # Pink
    (234, 88, 12),    # Orange
    (79, 70, 229),    # Indigo
    (101, 163, 13),   # Lime
]

VW_HEX_COLORS: list[str] = [
    "#dc2626", "#2563eb", "#16a34a", "#d97706", "#9333ea",
    "#0d9488", "#db2777", "#ea580c", "#4f46e5", "#65a30d",
]


def extract_keypoints_and_assignments(
    image: Image.Image,
    kmeans: KMeans,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract SIFT keypoints and assign each to its nearest visual word centroid.

    Parameters
    ----------
    image  : PIL Image
    kmeans : Fitted KMeans codebook

    Returns
    -------
    keypoints   : np.ndarray, shape (N, 2) with (row, col) coordinates
    descriptors : np.ndarray, shape (N, 128)
    assignments : np.ndarray, shape (N,) cluster indices 0..K-1
    """
    kps, descs, _ = extract_descriptors(image.convert("RGB"), method="SIFT")
    if len(descs) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 128), dtype=float), np.empty(0, dtype=int)
    assignments = kmeans.predict(descs.astype(np.float64))
    return kps, descs, assignments


def render_localization_overlay(
    image: Image.Image,
    keypoints: np.ndarray,
    assignments: np.ndarray,
    selected_words: list[int] | set[int] | None = None,
    grayscale_background: bool = True,
    radius: int = 5,
    palette: list[tuple[int, int, int]] | None = None,
) -> tuple[Image.Image, dict[int, dict]]:
    """Overlay color-coded keypoints onto image based on visual word assignment.

    Returns (overlay_image, word_stats).
    """
    if palette is None:
        palette = VW_COLORS

    if grayscale_background:
        base = image.convert("L").convert("RGB")
    else:
        base = image.convert("RGB").copy()

    draw = ImageDraw.Draw(base)
    word_stats: dict[int, dict] = {}

    if len(keypoints) == 0 or len(assignments) == 0:
        return base, word_stats

    # Determine unique words present
    unique_words, counts = np.unique(assignments, return_counts=True)
    count_map = dict(zip(unique_words, counts))

    if selected_words is None:
        # Default: Top 5 most frequent words
        ranked_words = sorted(count_map.keys(), key=lambda w: count_map[w], reverse=True)
        selected_words = ranked_words[:5]

    selected_set = set(selected_words)
    color_map: dict[int, tuple[int, int, int]] = {
        vw: palette[i % len(palette)] for i, vw in enumerate(selected_words)
    }

    # Draw non-selected keypoints as faint gray dots
    faint_color = (180, 180, 180)
    for idx, (row, col) in enumerate(keypoints):
        vw = assignments[idx]
        if vw not in selected_set:
            draw.ellipse(
                [(col - 2, row - 2), (col + 2, row + 2)],
                outline=faint_color,
                fill=faint_color,
            )

    # Draw selected keypoints in their assigned vibrant color with border
    for idx, (row, col) in enumerate(keypoints):
        vw = assignments[idx]
        if vw in selected_set:
            color = color_map[vw]
            # Outer dark ring for contrast
            draw.ellipse(
                [(col - radius - 1, row - radius - 1), (col + radius + 1, row + radius + 1)],
                outline=(0, 0, 0),
                width=1,
            )
            # Filled circle
            draw.ellipse(
                [(col - radius, row - radius), (col + radius, row + radius)],
                outline=color,
                fill=color,
                width=2,
            )

    for i, vw in enumerate(selected_words):
        word_stats[vw] = {
            "count": count_map.get(vw, 0),
            "color": color_map[vw],
            "hex": VW_HEX_COLORS[i % len(VW_HEX_COLORS)],
        }

    return base, word_stats


def render_localization_view(
    image: Image.Image,
    keypoints: np.ndarray,
    assignments: np.ndarray,
    top_words: list[int],
    visual_words_records: list[dict],
    image_name: str = "Image",
) -> Image.Image:
    """Render side-by-side localization panel: keypoint overlay + color legend with patches."""
    overlay_img, word_stats = render_localization_overlay(
        image,
        keypoints,
        assignments,
        selected_words=top_words,
        grayscale_background=True,
    )

    fig = plt.figure(figsize=(11, 5.2), dpi=100)
    # Left: Overlay Image
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(overlay_img)
    ax_img.set_title(
        f"Visual Word Localization ({image_name})\n"
        f"Total SIFT Keypoints: {len(keypoints)}",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_img.axis("off")

    # Right: Legend with mini patch cards
    n_words = len(top_words)
    if n_words > 0:
        sub_grid = fig.add_gridspec(n_words, 3, left=0.54, right=0.98, wspace=0.15, hspace=0.4)
        for row_idx, vw in enumerate(top_words):
            stats = word_stats.get(vw, {"count": 0, "hex": "#333333"})
            rec = visual_words_records[vw] if vw < len(visual_words_records) else {}

            # 1. Color badge & visual word name
            ax_name = fig.add_subplot(sub_grid[row_idx, 0])
            ax_name.text(
                0.05, 0.5,
                f"■ vw {vw}",
                fontsize=11,
                fontweight="bold",
                color=stats.get("hex", "#333333"),
                va="center",
            )
            ax_name.axis("off")

            # 2. Patch thumbnail
            ax_patch = fig.add_subplot(sub_grid[row_idx, 1])
            if rec and "patch" in rec:
                ax_patch.imshow(rec["patch"], cmap="gray")
            ax_patch.axis("off")

            # 3. Stats text
            ax_info = fig.add_subplot(sub_grid[row_idx, 2])
            pct = (stats["count"] / max(1, len(keypoints))) * 100
            ax_info.text(
                0.0, 0.5,
                f"{stats['count']} kps ({pct:.1f}%)",
                fontsize=10,
                va="center",
            )
            ax_info.axis("off")

    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def compute_histogram_similarities(
    query_hist: np.ndarray,
    train_hists: np.ndarray,
    metric: str = "chi2",
) -> np.ndarray:
    """Compute similarity scores between query BoW histogram and train histograms.

    Parameters
    ----------
    query_hist  : shape (K,)
    train_hists : shape (N_train, K)
    metric      : 'chi2' (Chi-squared kernel similarity) or 'intersection' (Histogram Intersection)

    Returns
    -------
    scores : np.ndarray, shape (N_train,), values in [0.0, 1.0] where 1.0 is max similarity
    """
    if metric == "chi2":
        scores = chi2_kernel(query_hist.reshape(1, -1), train_hists)[0]
    elif metric == "intersection":
        scores = np.sum(np.minimum(query_hist.reshape(1, -1), train_hists), axis=1)
    elif metric == "cosine":
        norm_q = np.linalg.norm(query_hist)
        norm_t = np.linalg.norm(train_hists, axis=1)
        denom = norm_q * norm_t
        denom[denom == 0] = 1e-9
        scores = np.dot(train_hists, query_hist) / denom
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return np.clip(scores, 0.0, 1.0)


def render_retrieval_composite(
    query_image: Image.Image,
    query_name: str,
    query_hist: np.ndarray,
    top_matches: list[dict],
    metric_name: str = "Chi-squared",
) -> Image.Image:
    """Render publication-quality CBIR retrieval view.

    Left: Query image with its BoW histogram.
    Right: Top retrieved database images with ranks, labels, and similarity scores.
    """
    n_matches = len(top_matches)
    fig = plt.figure(figsize=(3.5 + 2.2 * n_matches, 4.5), dpi=100)
    gs = fig.add_gridspec(2, 1 + n_matches, width_ratios=[1.4] + [1.0] * n_matches)

    # 1. Query image
    ax_q = fig.add_subplot(gs[0, 0])
    ax_q.imshow(query_image.convert("RGB"))
    ax_q.set_title(f"Query: {query_name[:15]}", fontsize=11, fontweight="bold")
    ax_q.axis("off")

    # 2. Query BoW histogram
    ax_qh = fig.add_subplot(gs[1, 0])
    ax_qh.bar(np.arange(len(query_hist)), query_hist, color="#4c8cbf", width=0.8)
    ax_qh.set_title("Query BoW Histogram", fontsize=9)
    ax_qh.set_xticks(np.arange(0, len(query_hist) + 1, 10))
    ax_qh.tick_params(axis="both", labelsize=8)

    # 3. Retrieved images
    for i, match in enumerate(top_matches):
        ax_m = fig.add_subplot(gs[:, i + 1])
        try:
            m_img = Image.open(match["path"]).convert("RGB")
            ax_m.imshow(m_img)
        except Exception:
            ax_m.text(0.5, 0.5, "Image\nnot found", ha="center", va="center")

        rank = match["rank"]
        sim_pct = match["similarity"] * 100
        cls_name = match["class"]
        ax_m.set_title(
            f"Rank #{rank}\n[{cls_name}]\nSim: {sim_pct:.1f}%",
            fontsize=10,
            fontweight="bold",
            color="#1d4ed8" if rank == 1 else "#374151",
            pad=4,
        )
        ax_m.axis("off")

    fig.suptitle(
        f"Content-Based Image Retrieval (CBIR) — Metric: {metric_name}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

