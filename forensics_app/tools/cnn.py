"""Set 8 — CNN classification with a model passport, and feature projection."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

SET8_DIR = Path(__file__).resolve().parent.parent / "models" / "set8"


def _figure_to_image(figure: Figure) -> Image.Image:
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def _load_torch_first() -> None:
    """Load torch (and triton) before tensorflow/numba: the other order crashes the
    process later (SIGFPE in YOLO, or an LLVM abort in UMAP)."""
    try:
        import triton  # noqa: F401  (not available on every platform)
    except ImportError:
        pass
    import torch  # noqa: F401


@lru_cache(maxsize=2)
def _load_model(path: Path):
    _load_torch_first()
    import keras  # heavy import, only when a Set 8 tool runs

    return keras.models.load_model(path, compile=False)


def classify(image: Image.Image, model_path: Path) -> tuple[dict, np.ndarray]:
    """Return the model passport and the class probabilities for ``image``."""
    passport = json.loads(model_path.with_suffix(".json").read_text())
    width, height = passport["preprocessing"]["input_size"]
    x = np.asarray(image.convert("RGB").resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0
    return passport, _load_model(model_path).predict(x[None], verbose=0)[0]


def project_features(features: np.ndarray, method: str) -> np.ndarray:
    """Project feature vectors to 2D with t-SNE or UMAP (PCA-50 first to remove noise)."""
    from sklearn.decomposition import PCA

    reduced = PCA(n_components=min(50, *features.shape)).fit_transform(features)
    if method == "umap":
        _load_torch_first()
        import umap

        return umap.UMAP(n_components=2, random_state=0).fit_transform(reduced)
    from sklearn.manifold import TSNE

    return TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(reduced)


class CNNClassifyTool(ForensicsTool):
    tool_id = "cnn_classify"
    title = "CNN classification"
    category = "Set 8"
    description = "Load a trained CNN, show its passport and classify the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        filename = filedialog.askopenfilename(
            title="Select a trained model", parent=parent, initialdir=SET8_DIR,
            filetypes=[("Keras model", "*.keras")],
        )
        if not filename:
            return None
        passport, probs = classify(document.current, Path(filename))
        classes = passport["class_names"]
        top = np.argsort(probs)[::-1][:3]
        label, confidence = classes[top[0]], float(probs[top[0]])

        figure = Figure(figsize=(10, 4))
        ax_image, ax_history = figure.subplots(1, 2)
        ax_image.imshow(document.current.convert("RGB"))
        ax_image.set_title(f"{label} ({confidence:.2f})")
        ax_image.axis("off")
        history = passport["training_history"]
        for key in ("accuracy", "val_accuracy"):
            ax_history.plot(history[key], label=key)
        ax_history.set_title("Training history")
        ax_history.set_xlabel("epoch")
        ax_history.legend()
        figure.tight_layout()

        sizes = passport["sample_size"]
        return ToolResult(
            image=_figure_to_image(figure),
            message=f"{passport['model']}: {label} ({confidence:.2f})",
            details={
                "Model": passport["model"],
                "Architecture": passport["architecture"],
                "Dataset": passport["dataset"],
                "Samples (train/val/test)": f"{sizes['train']} / {sizes['validation']} / {sizes['test']}",
                "Classes": ", ".join(classes),
                "Preprocessing": f"resize {passport['preprocessing']['input_size']}, {passport['preprocessing']['scale']}",
                "Final val accuracy": f"{history['val_accuracy'][-1]:.4f}",
                **{f"#{rank + 1} {classes[i]}": f"{probs[i]:.4f}" for rank, i in enumerate(top)},
            },
        )


class FeatureProjectionTool(ForensicsTool):
    tool_id = "feature_projection"
    title = "Feature projection (t-SNE / UMAP)"
    category = "Set 8"
    description = "Project saved CNN feature vectors of the test set to 2D."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        filename = filedialog.askopenfilename(
            title="Select extracted features", parent=parent, initialdir=SET8_DIR,
            filetypes=[("Feature vectors", "*_features.npz")],
        )
        if not filename:
            return None
        method = simpledialog.askstring("Projection", "Method (tsne / umap):", parent=parent, initialvalue="tsne")
        if method is None:
            return None
        method = method.strip().lower()
        if method not in ("tsne", "umap"):
            raise ValueError("Method must be 'tsne' or 'umap'.")

        data = np.load(filename)
        points = project_features(data["features"], method)

        figure = Figure(figsize=(8, 8))
        ax = figure.subplots()
        for index, name in enumerate(data["class_names"]):
            mask = data["labels"] == index
            ax.scatter(points[mask, 0], points[mask, 1], s=8, label=str(name))
        ax.legend(markerscale=2)
        ax.set_title(f"{'t-SNE' if method == 'tsne' else 'UMAP'} — {Path(filename).stem}")
        figure.tight_layout()

        return ToolResult(
            image=_figure_to_image(figure),
            message=f"Projected {len(points)} feature vectors with {method.upper()}.",
            details={"Features": Path(filename).name, "Samples": len(points),
                     "Dimensions": data["features"].shape[1], "Method": method.upper()},
        )
