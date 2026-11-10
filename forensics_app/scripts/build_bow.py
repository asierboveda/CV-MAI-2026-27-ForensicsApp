"""Set 7 — Bag-of-Visual-Words setup script.

Performs ALL one-time setup in a single run:

  1. Split Caltech_101_subset → train/ (80 %) and test/ (20 %), seed = 42
  2. Extract SIFT descriptors from every train image
  3. Fit KMeans(K=30) → visual-word codebook
  4. Build L1-normalised BoW histograms for all train images
  5. Train SVM with precomputed chi-squared kernel
  6. Pickle everything to forensics_app/models/

Usage (from the project root):
    .venv\\Scripts\\python.exe forensics_app/scripts/build_bow.py

Outputs
-------
forensics_app/models/kmeans_model.pkl     – KMeans codebook (K=30)
forensics_app/models/classifier.pkl      – SVM (chi² precomputed kernel)
forensics_app/models/train_histograms.pkl – training BoW vectors (N_train, 30)
forensics_app/models/label_classes.pkl   – ['butterfly', 'dalmatian', 'kangaroo']
"""

from __future__ import annotations

import pickle
import random
import shutil
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure forensics_app is importable when running this script directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC

from forensics_app.tools.descriptor_extractor import extract_descriptors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).parents[1] / "images" / "Caltech_101_subset"
TRAIN_DIR   = DATASET_DIR / "train"
TEST_DIR    = DATASET_DIR / "test"
MODELS_DIR  = Path(__file__).parents[1] / "models"

K           = 30     # 10 × num_classes  (3 classes → 30 visual words)
SPLIT_RATIO = 0.80
SEED        = 42
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_files(folder: Path) -> list[Path]:
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS)


# ===========================================================================
# STEP 1 — Split dataset
# ===========================================================================
print("=" * 65)
print("STEP 1: Splitting dataset (80 / 20, seed=42) …")

class_dirs = sorted(
    d for d in DATASET_DIR.iterdir()
    if d.is_dir() and d.name not in ("train", "test")
)
classes = [d.name for d in class_dirs]
print(f"  Classes: {classes}")

for cls_dir in class_dirs:
    imgs = _image_files(cls_dir)
    random.seed(SEED)
    random.shuffle(imgs)
    n_train = int(len(imgs) * SPLIT_RATIO)
    splits  = [(TRAIN_DIR, imgs[:n_train]), (TEST_DIR, imgs[n_train:])]
    for dest_root, files in splits:
        out = dest_root / cls_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for src in files:
            tgt = out / src.name
            if not tgt.exists():
                shutil.copy2(src, tgt)
    print(f"  {cls_dir.name}: {n_train} train / {len(imgs) - n_train} test")

# ===========================================================================
# STEP 2 — Collect SIFT descriptors + keep track of train images
# ===========================================================================
print("\nSTEP 2: Extracting SIFT descriptors from train images …")

# We store (img_path, label_idx, kps, descriptors) so we can also find
# the representative patch for each visual word (Set 7.1).
train_records: list[tuple[Path, int, np.ndarray, np.ndarray]] = []
all_desc_parts: list[np.ndarray] = []

for label_idx, cls in enumerate(classes):
    img_files = _image_files(TRAIN_DIR / cls)
    for img_path in img_files:
        img = Image.open(img_path).convert("RGB")
        kps, descs, _ = extract_descriptors(img, method="SIFT")
        descs_f64 = descs.astype(np.float64) if len(descs) > 0 else np.empty((0, 128))
        train_records.append((img_path, label_idx, kps, descs_f64))
        if len(descs_f64) > 0:
            all_desc_parts.append(descs_f64)

n_total_descs = sum(len(d) for d in all_desc_parts)
print(f"  {len(train_records)} train images, {n_total_descs:,} SIFT descriptors collected")

# ===========================================================================
# STEP 3 — Fit KMeans codebook
#           KMeans uses Euclidean distance internally to assign each descriptor
#           to its nearest cluster centroid (visual word).
# ===========================================================================
print(f"\nSTEP 3: Fitting MiniBatchKMeans (K={K}) …")

all_descs_stacked = np.vstack(all_desc_parts)   # shape (N_total_descs, 128)
batch_size = min(1024, max(256, len(all_descs_stacked) // 5))
kmeans = MiniBatchKMeans(
    n_clusters=K,
    batch_size=batch_size,
    random_state=SEED,
    n_init="auto",
    verbose=0,
)
kmeans.fit(all_descs_stacked)
print(f"  Codebook fitted: {K} visual words from {len(all_descs_stacked):,} descriptors")

# ===========================================================================
# STEP 3b — Find representative patch for each visual word (Set 7.1)
# ===========================================================================
print("\nSTEP 3b: Extracting representative image patches for visual words …")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

best_patches = [
    {"dist": float("inf"), "img_path": None, "kp": None}
    for _ in range(K)
]

for img_path, label_idx, kps, descs_f64 in train_records:
    if len(descs_f64) == 0:
        continue
    for c_idx in range(K):
        dists = np.linalg.norm(descs_f64 - kmeans.cluster_centers_[c_idx], axis=1)
        min_kp_idx = int(np.argmin(dists))
        if dists[min_kp_idx] < best_patches[c_idx]["dist"]:
            best_patches[c_idx]["dist"] = float(dists[min_kp_idx])
            best_patches[c_idx]["img_path"] = img_path
            best_patches[c_idx]["kp"] = kps[min_kp_idx]

PATCH_SIZE = 48
HALF_SIZE = PATCH_SIZE // 2
visual_word_records = []

for c_idx in range(K):
    rec = best_patches[c_idx]
    src_img = Image.open(rec["img_path"]).convert("L")
    row, col = int(round(rec["kp"][0])), int(round(rec["kp"][1]))

    left = max(0, col - HALF_SIZE)
    right = min(src_img.width, col + HALF_SIZE)
    top = max(0, row - HALF_SIZE)
    bottom = min(src_img.height, row + HALF_SIZE)

    patch = src_img.crop((left, top, right, bottom))
    if patch.size != (PATCH_SIZE, PATCH_SIZE):
        patch = patch.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)

    visual_word_records.append({
        "index": c_idx,
        "name": f"vw {c_idx}",
        "patch": patch,
        "source_name": rec["img_path"].name,
        "source_class": rec["img_path"].parent.name,
        "kp": (row, col),
        "distance": rec["dist"],
    })

# Render 5x6 grid matching Slide 2
fig, axes = plt.subplots(5, 6, figsize=(10, 8), dpi=100)
for idx, ax in enumerate(axes.flat):
    if idx < K:
        ax.imshow(visual_word_records[idx]["patch"], cmap="gray")
        ax.set_title(f"vw {idx}", fontsize=10, pad=2)
    ax.axis("off")
fig.suptitle("Visual Words (Codebook Centroids)", fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout()
MODELS_DIR.mkdir(exist_ok=True)
grid_path = MODELS_DIR / "visual_words_grid.png"
fig.savefig(grid_path, bbox_inches="tight", dpi=120)
plt.close(fig)
print(f"  Visual words grid saved → {grid_path}")

# ===========================================================================
# STEP 4 — Build L1-normalised BoW histograms for all train images
# ===========================================================================
print("\nSTEP 4: Building BoW histograms for train images …")

train_hists:  list[np.ndarray] = []
train_labels: list[int]        = []

for img_path, label_idx, kps, descs_f64 in train_records:
    if len(descs_f64) == 0:
        hist = np.zeros(K, dtype=np.float64)
    else:
        # Assign each descriptor to its nearest visual word (Euclidean via KMeans)
        assignments = kmeans.predict(descs_f64)
        hist, _     = np.histogram(assignments, bins=np.arange(K + 1))
        hist        = hist.astype(np.float64)
        total       = hist.sum()
        if total > 0:
            hist /= total  # L1 normalise
    train_hists.append(hist)
    train_labels.append(label_idx)

X_train = np.array(train_hists)   # shape (N_train, K)
y_train = np.array(train_labels)  # shape (N_train,)
print(f"  X_train shape: {X_train.shape}")

# ===========================================================================
# STEP 5 — Train SVM with chi-squared kernel
#
# chi²(h, g) = Σ_i  (h_i - g_i)² / (h_i + g_i)
# K_chi²(h,g) = exp(-γ · chi²(h, g))
#
# We precompute the full (N_train × N_train) kernel matrix and pass it to
# SVC(kernel='precomputed').  At inference time the kernel row between the
# query and X_train is computed on-the-fly using chi2_kernel.
# ===========================================================================
print("\nSTEP 5: Training SVM with chi-squared kernel …")

K_train = chi2_kernel(X_train, X_train)   # shape (N_train, N_train)
clf = SVC(kernel="precomputed", probability=True, random_state=SEED)
clf.fit(K_train, y_train)
print(f"  SVM fitted. Support vectors per class: {clf.n_support_}")

# ===========================================================================
# STEP 6 — Pickle all artefacts
# ===========================================================================
print("\nSTEP 6: Saving models …")
MODELS_DIR.mkdir(exist_ok=True)

artefacts = {
    "kmeans_model.pkl":     kmeans,
    "classifier.pkl":       clf,
    "train_histograms.pkl": X_train,
    "label_classes.pkl":    classes,
    "visual_words.pkl":     visual_word_records,
}
for filename, obj in artefacts.items():
    out_path = MODELS_DIR / filename
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {out_path}")

print(f"\n{'='*65}")
print("All done! Models saved to:", MODELS_DIR)
print("Launch the app: .venv\\Scripts\\python.exe run.py")
print("=" * 65)

