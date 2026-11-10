"""Shared Bag-of-Visual-Words helpers for Set 7.

Loaded models are cached at module level so they are only read from disk once
per application session.  All Set 7 tools import from here.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC

from .descriptor_extractor import extract_descriptors

# ---------------------------------------------------------------------------
# Paths & Active Model State
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parents[1] / "models"
DATASET_DIR = Path(__file__).parents[1] / "images" / "Caltech_101_subset"
TEST_DIR = DATASET_DIR / "test"

# Active model directory (defaults to forensics_app/models/)
_active_model_dir: Path = MODELS_DIR
_session_remember_choice: bool = False

# Module-level cache — populated lazily on first use
_cache: dict[str, Any] = {}


def get_default_model_dir() -> Path:
    """Return default models directory."""
    return MODELS_DIR


def get_active_model_dir() -> Path:
    """Return the currently active model directory."""
    return _active_model_dir


def set_active_model_dir(path: Path | str) -> None:
    """Set the active model directory and clear model cache."""
    global _active_model_dir
    _active_model_dir = Path(path).resolve()
    _cache.clear()


def validate_model_dir(path: Path | str) -> tuple[bool, str]:
    """Validate that a directory contains all required model bundle artefacts.

    Returns (is_valid, message).
    """
    p = Path(path).resolve()
    if not p.is_dir():
        return False, f"Directory does not exist: {p}"

    required_files = [
        "kmeans_model.pkl",
        "classifier.pkl",
        "train_histograms.pkl",
        "label_classes.pkl",
        "visual_words.pkl",
    ]
    missing = [f for f in required_files if not (p / f).exists()]
    if missing:
        return False, f"Missing required model files in {p.name}: {', '.join(missing)}"
    return True, "Valid codebook bundle"


def _load(name: str) -> Any:
    """Load a pickled artefact from the active model directory with fallback and caching."""
    if name not in _cache:
        path = _active_model_dir / name
        if not path.exists():
            # Fallback to default MODELS_DIR if present
            if (MODELS_DIR / name).exists():
                path = MODELS_DIR / name
            else:
                raise FileNotFoundError(
                    f"Model file not found: {name} in {_active_model_dir}\n\n"
                    "Please construct a codebook or ensure the model directory is valid."
                )
        with open(path, "rb") as f:
            _cache[name] = pickle.load(f)
    return _cache[name]


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_kmeans() -> KMeans:
    """Return the fitted KMeans codebook."""
    return _load("kmeans_model.pkl")


def load_classifier() -> SVC:
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

    If train_image_records.pkl is present in the active model directory, loads it;
    otherwise scans DATASET_DIR / 'train' for matching classes.
    """
    if "train_image_records" not in _cache:
        rec_path = _active_model_dir / "train_image_records.pkl"
        if rec_path.exists():
            with open(rec_path, "rb") as f:
                _cache["train_image_records"] = pickle.load(f)
        else:
            classes = load_classes()
            records: list[tuple[Path, str]] = []
            train_dir = DATASET_DIR / "train"
            for cls in classes:
                folder = train_dir / cls
                if folder.exists():
                    for p in sorted(folder.iterdir()):
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                            records.append((p, cls))
            _cache["train_image_records"] = records
    return _cache["train_image_records"]


def load_visual_words() -> list[dict]:
    """Return list of dicts for visual words vw 0 .. vw K-1."""
    return _load("visual_words.pkl")


def load_visual_words_grid(model_dir: Path | None = None) -> Image.Image:
    """Return the overview visual words grid image."""
    target_dir = Path(model_dir).resolve() if model_dir else _active_model_dir
    grid_path = target_dir / "visual_words_grid.png"
    if not grid_path.exists():
        if (MODELS_DIR / "visual_words_grid.png").exists():
            grid_path = MODELS_DIR / "visual_words_grid.png"
        else:
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


# ---------------------------------------------------------------------------
# Dynamic Codebook Construction & Training Bundle Engine
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_dataset_classes(dataset_dir: Path | str) -> dict[str, list[Path]]:
    """Scan dataset_dir for class subdirectories containing image files.

    Returns dict mapping class_name -> list of image Paths.
    """
    p = Path(dataset_dir).resolve()
    classes_map: dict[str, list[Path]] = {}
    if not p.is_dir():
        return classes_map

    for entry in sorted(p.iterdir()):
        if entry.is_dir() and entry.name not in ("train", "test", "__pycache__"):
            imgs = sorted(
                f for f in entry.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
            )
            if imgs:
                classes_map[entry.name] = imgs

    # If no top-level class dirs found, check if train/ exists
    if not classes_map and (p / "train").is_dir():
        for entry in sorted((p / "train").iterdir()):
            if entry.is_dir() and entry.name != "__pycache__":
                imgs = sorted(
                    f for f in entry.iterdir()
                    if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
                )
                if imgs:
                    classes_map[entry.name] = imgs
    return classes_map


def build_codebook_bundle(
    dataset_dir: Path | str,
    output_dir: Path | str,
    k: int,
    split_ratio: float = 0.80,
    seed: int = 42,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Construct a full Bag-of-Visual-Words codebook and train SVM classifier.

    Saves:
      - kmeans_model.pkl
      - classifier.pkl
      - train_histograms.pkl
      - label_classes.pkl
      - visual_words.pkl
      - visual_words_grid.png
      - train_image_records.pkl
      - codebook_info.json
    """
    def _notify(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    ds_path = Path(dataset_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    _notify(0.02, "Scanning dataset directories...")
    classes_map = find_dataset_classes(ds_path)
    if not classes_map:
        raise ValueError(
            f"No valid class subdirectories with images found in: {ds_path}\n"
            "Ensure the directory contains class folders (e.g. dataset/class_a/*.jpg)."
        )

    class_names = sorted(classes_map.keys())
    total_images = sum(len(imgs) for imgs in classes_map.values())

    # Range validation: 1 < k < total_images
    if k <= 1 or k >= total_images:
        raise ValueError(
            f"Invalid cluster count K={k}. Must be strictly greater than 1 "
            f"and less than total images ({total_images}). Valid range: 2 to {total_images - 1}."
        )

    # 1. Train / test split
    _notify(0.06, f"Splitting {total_images} images across {len(class_names)} classes...")
    train_records_meta: list[tuple[Path, int, str]] = []
    for cls_idx, cls_name in enumerate(class_names):
        imgs = list(classes_map[cls_name])
        rng = random.Random(seed)
        rng.shuffle(imgs)
        if 0.0 < split_ratio < 1.0:
            n_train = max(1, int(len(imgs) * split_ratio))
        else:
            n_train = len(imgs)
        for img_p in imgs[:n_train]:
            train_records_meta.append((img_p, cls_idx, cls_name))

    num_train = len(train_records_meta)
    _notify(0.10, f"Extracting SIFT descriptors from {num_train} train images...")

    # 2. Extract SIFT descriptors
    train_records_sift: list[tuple[Path, int, np.ndarray, np.ndarray]] = []
    all_desc_parts: list[np.ndarray] = []

    for idx, (img_path, label_idx, _cname) in enumerate(train_records_meta):
        try:
            img = Image.open(img_path).convert("RGB")
            kps, descs, _ = extract_descriptors(img, method="SIFT")
            descs_f64 = descs.astype(np.float64) if len(descs) > 0 else np.empty((0, 128))
        except Exception:
            kps = np.empty((0, 2), dtype=float)
            descs_f64 = np.empty((0, 128), dtype=np.float64)

        train_records_sift.append((img_path, label_idx, kps, descs_f64))
        if len(descs_f64) > 0:
            all_desc_parts.append(descs_f64)

        if idx % max(1, num_train // 10) == 0 or idx == num_train - 1:
            progress = 0.10 + 0.35 * ((idx + 1) / num_train)
            _notify(progress, f"SIFT extraction: {idx + 1}/{num_train} images done...")

    if not all_desc_parts:
        raise ValueError("SIFT could not extract any descriptors from the dataset images.")

    all_descs_stacked = np.vstack(all_desc_parts)
    if len(all_descs_stacked) < k:
        raise ValueError(
            f"Total descriptors ({len(all_descs_stacked)}) is less than K={k}. "
            "Please choose a smaller K or use higher contrast images."
        )

    # 3. Fit KMeans codebook
    _notify(0.50, f"Fitting KMeans (K={k}) on {len(all_descs_stacked):,} descriptors...")
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto", verbose=0)
    kmeans.fit(all_descs_stacked)

    # 4. Extract representative patches
    _notify(0.65, "Extracting canonical visual word patches...")
    best_patches = [{"dist": float("inf"), "img_path": None, "kp": None} for _ in range(k)]
    for img_path, _label_idx, kps, descs_f64 in train_records_sift:
        if len(descs_f64) == 0:
            continue
        for c_idx in range(k):
            dists = np.linalg.norm(descs_f64 - kmeans.cluster_centers_[c_idx], axis=1)
            min_kp_idx = int(np.argmin(dists))
            if dists[min_kp_idx] < best_patches[c_idx]["dist"]:
                best_patches[c_idx]["dist"] = float(dists[min_kp_idx])
                best_patches[c_idx]["img_path"] = img_path
                best_patches[c_idx]["kp"] = kps[min_kp_idx]

    patch_size = 48
    half_size = patch_size // 2
    visual_word_records: list[dict[str, Any]] = []

    for c_idx in range(k):
        rec = best_patches[c_idx]
        if rec["img_path"] is not None:
            src_img = Image.open(rec["img_path"]).convert("L")
            row, col = int(round(rec["kp"][0])), int(round(rec["kp"][1]))
            left = max(0, col - half_size)
            right = min(src_img.width, col + half_size)
            top = max(0, row - half_size)
            bottom = min(src_img.height, row + half_size)
            patch = src_img.crop((left, top, right, bottom))
            if patch.size != (patch_size, patch_size):
                patch = patch.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
        else:
            patch = Image.new("L", (patch_size, patch_size), color=128)

        visual_word_records.append({
            "index": c_idx,
            "name": f"vw {c_idx}",
            "patch": patch,
            "source_name": rec["img_path"].name if rec["img_path"] else "none",
            "source_class": rec["img_path"].parent.name if rec["img_path"] else "none",
            "distance": rec["dist"],
        })

    # 5. Render visual words grid image
    _notify(0.75, "Rendering visual words overview grid...")
    cols = min(6, k) if k <= 36 else 8
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6), dpi=100)
    ax_list = np.array(axes).flatten() if hasattr(axes, "flatten") else [axes]

    for idx, ax in enumerate(ax_list):
        if idx < k:
            ax.imshow(visual_word_records[idx]["patch"], cmap="gray")
            ax.set_title(f"vw {idx}", fontsize=9, pad=2)
        ax.axis("off")
    fig.suptitle(f"Visual Words Codebook (K={k})", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    grid_img_path = out_path / "visual_words_grid.png"
    fig.savefig(grid_img_path, bbox_inches="tight", dpi=110)
    plt.close(fig)

    # 6. Build L1-normalized train BoW histograms
    _notify(0.82, "Building L1-normalized histograms for training images...")
    train_hists: list[np.ndarray] = []
    train_labels: list[int] = []
    train_image_records: list[tuple[Path, str]] = []

    for img_path, label_idx, _kps, descs_f64 in train_records_sift:
        if len(descs_f64) == 0:
            hist = np.zeros(k, dtype=np.float64)
        else:
            assignments = kmeans.predict(descs_f64)
            hist, _ = np.histogram(assignments, bins=np.arange(k + 1))
            hist = hist.astype(np.float64)
            tot = hist.sum()
            if tot > 0:
                hist /= tot
        train_hists.append(hist)
        train_labels.append(label_idx)
        train_image_records.append((img_path, class_names[label_idx]))

    X_train = np.array(train_hists)
    y_train = np.array(train_labels)

    # 7. Train SVM classifier with chi-squared precomputed kernel
    _notify(0.90, "Training SVM classifier with Chi-squared kernel...")
    K_train = chi2_kernel(X_train, X_train)
    clf = SVC(kernel="precomputed", probability=True, random_state=seed)
    clf.fit(K_train, y_train)

    # 8. Save all bundle files
    _notify(0.95, "Saving codebook model bundle...")
    artefacts = {
        "kmeans_model.pkl": kmeans,
        "classifier.pkl": clf,
        "train_histograms.pkl": X_train,
        "label_classes.pkl": class_names,
        "visual_words.pkl": visual_word_records,
        "train_image_records.pkl": train_image_records,
    }
    for fname, obj in artefacts.items():
        with open(out_path / fname, "wb") as f:
            pickle.dump(obj, f)

    meta_info = {
        "dataset_dir": str(ds_path),
        "output_dir": str(out_path),
        "k": k,
        "classes": class_names,
        "num_classes": len(class_names),
        "num_train_images": num_train,
        "total_images": total_images,
        "total_descriptors": int(len(all_descs_stacked)),
    }
    with open(out_path / "codebook_info.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    # Set as active model directory in session
    set_active_model_dir(out_path)
    _notify(1.0, f"Codebook construction complete (K={k})! Saved to {out_path.name}")

    return meta_info


# ---------------------------------------------------------------------------
# Model Selection Dialog Helper
# ---------------------------------------------------------------------------

class CodebookSelectionDialog(tk.Toplevel):
    """Dialog allowing user to choose default codebook or custom codebook folder."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.title("Select BoW Codebook")
        self.geometry("540x300")
        self.minsize(480, 260)
        self.resizable(True, False)
        self.transient(parent)
        self.grab_set()

        self.result: Path | None = None
        self._remember_var = tk.BooleanVar(value=_session_remember_choice)
        self._mode_var = tk.StringVar(
            value="default" if _active_model_dir == MODELS_DIR else "custom"
        )
        self._custom_path_var = tk.StringVar(
            value=str(_active_model_dir) if _active_model_dir != MODELS_DIR else ""
        )

        self._build_ui()
        self.wait_window()

    def _build_ui(self) -> None:
        pad = 12
        frame = ttk.Frame(self, padding=pad)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Choose BoW Codebook for this operation:",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        # Radio 1: Default
        r1 = ttk.Radiobutton(
            frame,
            text=f"Default Pre-trained Codebook (Caltech-101 subset, K=30, 3 classes)",
            variable=self._mode_var,
            value="default",
            command=self._on_mode_change,
        )
        r1.pack(anchor="w", pady=4)

        # Radio 2: Custom
        r2 = ttk.Radiobutton(
            frame,
            text="Custom Codebook Folder:",
            variable=self._mode_var,
            value="custom",
            command=self._on_mode_change,
        )
        r2.pack(anchor="w", pady=(8, 2))

        custom_box = ttk.Frame(frame)
        custom_box.pack(fill="x", padx=20, pady=(0, 8))

        self.path_entry = ttk.Entry(custom_box, textvariable=self._custom_path_var)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_browse = ttk.Button(custom_box, text="Browse...", command=self._on_browse)
        self.btn_browse.pack(side="left")

        # Status indicator
        self.status_lbl = ttk.Label(frame, text="", font=("TkDefaultFont", 9))
        self.status_lbl.pack(anchor="w", pady=4)

        # Checkbox: Remember for session
        ttk.Checkbutton(
            frame,
            text="Don't ask again this session (use this codebook for future runs)",
            variable=self._remember_var,
        ).pack(anchor="w", pady=(6, 12))

        # Buttons
        btn_box = ttk.Frame(frame)
        btn_box.pack(side="bottom", fill="x")

        ttk.Button(btn_box, text="Proceed", command=self._on_proceed).pack(side="right", padx=(6, 0))
        ttk.Button(btn_box, text="Cancel", command=self.destroy).pack(side="right")

        self._on_mode_change()

    def _on_mode_change(self) -> None:
        is_custom = self._mode_var.get() == "custom"
        self.path_entry.configure(state="normal" if is_custom else "disabled")
        self.btn_browse.configure(state="normal" if is_custom else "disabled")
        if not is_custom:
            self.status_lbl.configure(
                text=f"Active: {MODELS_DIR.name} (Default)",
                foreground="#16a34a",
            )
        else:
            self._validate_custom_path()

    def _on_browse(self) -> None:
        folder = filedialog.askdirectory(
            parent=self,
            title="Select Custom Codebook Folder",
            initialdir=str(MODELS_DIR.parent),
        )
        if folder:
            self._custom_path_var.set(folder)
            self._validate_custom_path()

    def _validate_custom_path(self) -> bool:
        p_str = self._custom_path_var.get().strip()
        if not p_str:
            self.status_lbl.configure(text="Please select a codebook folder.", foreground="#b45309")
            return False
        valid, msg = validate_model_dir(p_str)
        if valid:
            self.status_lbl.configure(text=f"✓ {msg}", foreground="#16a34a")
            return True
        else:
            self.status_lbl.configure(text=f"✗ {msg}", foreground="#dc2626")
            return False

    def _on_proceed(self) -> None:
        global _session_remember_choice
        if self._mode_var.get() == "default":
            set_active_model_dir(MODELS_DIR)
            _session_remember_choice = self._remember_var.get()
            self.result = MODELS_DIR
            self.destroy()
        else:
            if not self._validate_custom_path():
                messagebox.showerror(
                    "Invalid Codebook",
                    "The selected folder does not contain a valid codebook bundle.",
                    parent=self,
                )
                return
            target = Path(self._custom_path_var.get().strip()).resolve()
            set_active_model_dir(target)
            _session_remember_choice = self._remember_var.get()
            self.result = target
            self.destroy()


def prompt_codebook_selection(
    parent: tk.Misc | None = None,
    force_prompt: bool = False,
) -> Path | None:
    """Prompt user to select or confirm codebook if needed."""
    global _session_remember_choice
    if not force_prompt and _session_remember_choice:
        return _active_model_dir

    if parent is None:
        return _active_model_dir

    try:
        if not parent.winfo_exists() or not parent.winfo_viewable():
            return _active_model_dir
    except Exception:
        return _active_model_dir

    dialog = CodebookSelectionDialog(parent)
    return dialog.result


