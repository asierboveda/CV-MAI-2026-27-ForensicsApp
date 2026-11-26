"""Set 10 — CLIP (ViT-B/32) zero-shot classification and text-to-image search."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_TEMPLATE = "a photo of a {}."
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SET10_IMAGES_DIR = Path(__file__).resolve().parent.parent / "images" / "set10"


def _figure_to_image(figure: Figure) -> Image.Image:
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def _as_tensor(out):
    """Handle both raw tensor outputs and BaseModelOutputWithPooling."""
    import torch

    return out if torch.is_tensor(out) else out.pooler_output


@lru_cache(maxsize=1)
def _load_clip(model_name: str = CLIP_MODEL_NAME):
    """Lazily load CLIP ViT-B/32 model and processor."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def embed_images(images: list[Image.Image], model_name: str = CLIP_MODEL_NAME) -> np.ndarray:
    """Return L2-normalized CLIP image embeddings of shape ``(N, D)``."""
    import torch
    import torch.nn.functional as F

    model, processor, device = _load_clip(model_name)
    rgb_images = [img.convert("RGB") for img in images]
    with torch.inference_mode():
        inputs = processor(images=rgb_images, return_tensors="pt").to(device)
        emb = _as_tensor(model.get_image_features(**inputs))
        return F.normalize(emb, dim=-1).cpu().numpy()


def embed_texts(texts: list[str], model_name: str = CLIP_MODEL_NAME) -> np.ndarray:
    """Return L2-normalized CLIP text embeddings of shape ``(C, D)``."""
    import torch
    import torch.nn.functional as F

    model, processor, device = _load_clip(model_name)
    with torch.inference_mode():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        emb = _as_tensor(model.get_text_features(**inputs))
        return F.normalize(emb, dim=-1).cpu().numpy()


def parse_labels(raw: str) -> list[str]:
    """Parse comma- or newline-separated class labels into a non-empty list."""
    parts = [part.strip() for chunk in raw.splitlines() for part in chunk.split(",")]
    return [part for part in parts if part]


def format_prompt(template: str, label: str) -> str:
    """Insert ``label`` into ``template`` (falls back to appending if ``{}`` is absent)."""
    clean_template = template.strip() or "{}"
    if "{}" in clean_template:
        return clean_template.format(label)
    return f"{clean_template} {label}".strip()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def zero_shot_classify(
    image: Image.Image,
    labels: list[str],
    template: str = DEFAULT_TEMPLATE,
) -> np.ndarray:
    """Compute CLIP zero-shot class probabilities for ``image`` over ``labels``."""
    if not labels:
        raise ValueError("Enter at least one class label.")

    img_emb = embed_images([image])  # (1, D)
    prompts = [format_prompt(template, label) for label in labels]
    txt_emb = embed_texts(prompts)  # (C, D)

    model, _, _ = _load_clip()
    scale = float(model.logit_scale.exp().item())
    logits = scale * (img_emb @ txt_emb.T)[0]
    return _softmax(logits)


def render_zero_shot_figure(
    image: Image.Image,
    labels: list[str],
    probs: np.ndarray,
    subtitle: str,
) -> Image.Image:
    """Render the input image alongside a sorted horizontal probability bar chart."""
    order = np.argsort(probs)
    fig_height = max(3.6, 0.45 * len(labels) + 1.5)
    figure = Figure(figsize=(10, fig_height))
    ax_img, ax_bar = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})

    ax_img.imshow(image.convert("RGB"))
    ax_img.axis("off")
    ax_img.set_title("Input image")

    ordered_labels = [labels[i] for i in order]
    ordered_probs = probs[order]
    ax_bar.barh(ordered_labels, ordered_probs, color="#3b6ea8")
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("probability")
    ax_bar.set_title(subtitle)
    for idx, val in enumerate(ordered_probs):
        ax_bar.text(min(float(val) + 0.02, 0.90), idx, f"{val:.2f}", va="center")

    figure.tight_layout()
    return _figure_to_image(figure)


@lru_cache(maxsize=4)
def _embed_folder(folder_str: str) -> tuple[tuple[Path, ...], np.ndarray]:
    """Load and cache CLIP embeddings for all images in ``folder_str``."""
    folder = Path(folder_str)
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")

    paths = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found in {folder.name}.")

    batch_size = 16
    valid_paths: list[Path] = []
    embeddings: list[np.ndarray] = []
    batch_imgs: list[Image.Image] = []
    batch_paths: list[Path] = []

    for path in paths:
        try:
            with Image.open(path) as img:
                batch_imgs.append(img.convert("RGB"))
                batch_paths.append(path)
        except OSError:
            continue

        if len(batch_imgs) == batch_size:
            embeddings.append(embed_images(batch_imgs))
            valid_paths.extend(batch_paths)
            batch_imgs.clear()
            batch_paths.clear()

    if batch_imgs:
        embeddings.append(embed_images(batch_imgs))
        valid_paths.extend(batch_paths)

    if not valid_paths:
        raise ValueError(f"Could not open any images in {folder.name}.")

    return tuple(valid_paths), np.vstack(embeddings)


def search_folder(folder: Path, query: str, top_k: int = 3) -> list[tuple[Path, float]]:
    """Rank images in ``folder`` by CLIP cosine similarity to ``query``."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Enter a search query.")
    if top_k < 1:
        raise ValueError("Top K must be at least 1.")

    paths, gallery = _embed_folder(str(folder.resolve()))
    q_emb = embed_texts([clean_query])  # (1, D)
    sims = (q_emb @ gallery.T)[0]
    k = min(top_k, len(paths))
    best = np.argsort(-sims)[:k]
    return [(paths[i], float(sims[i])) for i in best]


def render_search_figure(query: str, matches: list[tuple[Path, float]]) -> Image.Image:
    """Render top-K retrieved images in a horizontal grid with similarity scores."""
    k = len(matches)
    figure = Figure(figsize=(3.4 * k, 3.6))
    axes = figure.subplots(1, k)
    if k == 1:
        axes = [axes]

    for ax, (path, sim) in zip(axes, matches):
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.axis("off")
        ax.set_title(f"{path.name}\nsim={sim:.3f}", fontsize=9)

    figure.suptitle(f'Query: "{query}"')
    figure.tight_layout()
    return _figure_to_image(figure)


class _ZeroShotDialog(tk.Toplevel):
    """Modal dialog to collect class labels and prompt template."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.title("CLIP Zero-Shot Classification (ViT-B/32)")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: tuple[list[str], str] | None = None
        self.labels_var = tk.StringVar(value="cat, dog, tiger, rabbit")
        self.template_var = tk.StringVar(value=DEFAULT_TEMPLATE)

        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Class labels (comma-separated):").grid(row=0, column=0, sticky="w")
        labels_entry = ttk.Entry(frame, textvariable=self.labels_var, width=46)
        labels_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        labels_entry.focus_set()
        labels_entry.selection_range(0, "end")

        ttk.Label(frame, text="Prompt template (use {} for the label):").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.template_var, width=46).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(4, 12)
        )

        btns = ttk.Frame(frame)
        btns.grid(row=4, column=0, columnspan=2, sticky="e")
        ttk.Button(btns, text="Cancel", command=self._cancel).pack(side="right")
        ttk.Button(btns, text="Classify", command=self._submit).pack(side="right", padx=(0, 8))

        self.bind("<Return>", lambda _e: self._submit())
        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _submit(self) -> None:
        labels = parse_labels(self.labels_var.get())
        if not labels:
            messagebox.showwarning("Invalid input", "Enter at least one class label.", parent=self)
            return
        self.result = (labels, self.template_var.get())
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class _TextSearchDialog(tk.Toplevel):
    """Modal dialog to collect image folder, text query, and Top-K count."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.title("CLIP Text-to-Image Search (ViT-B/32)")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: tuple[Path, str, int] | None = None
        self.folder_var = tk.StringVar(value=str(SET10_IMAGES_DIR))
        self.query_var = tk.StringVar(value="something you drink in the morning")
        self.topk_var = tk.IntVar(value=3)

        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Image folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.folder_var, width=38).grid(row=1, column=0, sticky="ew", pady=(4, 10))
        ttk.Button(frame, text="Browse…", command=self._browse).grid(row=1, column=1, padx=(6, 0), pady=(4, 10))

        ttk.Label(frame, text="Search query:").grid(row=2, column=0, sticky="w")
        query_entry = ttk.Entry(frame, textvariable=self.query_var, width=46)
        query_entry.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        query_entry.focus_set()

        ttk.Label(frame, text="Top matches (K):").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.topk_var, width=6).grid(
            row=4, column=1, sticky="e", pady=(0, 12)
        )

        btns = ttk.Frame(frame)
        btns.grid(row=5, column=0, columnspan=2, sticky="e")
        ttk.Button(btns, text="Cancel", command=self._cancel).pack(side="right")
        ttk.Button(btns, text="Search", command=self._submit).pack(side="right", padx=(0, 8))

        self.bind("<Return>", lambda _e: self._submit())
        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _browse(self) -> None:
        initial = self.folder_var.get().strip() or str(SET10_IMAGES_DIR)
        folder = filedialog.askdirectory(title="Select image folder", initialdir=initial, parent=self)
        if folder:
            self.folder_var.set(folder)

    def _submit(self) -> None:
        folder_str = self.folder_var.get().strip()
        query = self.query_var.get().strip()
        if not folder_str:
            messagebox.showwarning("Invalid input", "Select an image folder.", parent=self)
            return
        if not query:
            messagebox.showwarning("Invalid input", "Enter a search query.", parent=self)
            return
        try:
            top_k = int(self.topk_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Invalid input", "Top K must be an integer.", parent=self)
            return
        self.result = (Path(folder_str), query, max(1, top_k))
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class CLIPZeroShotTool(ForensicsTool):
    tool_id = "clip_zero_shot"
    title = "CLIP zero-shot classification"
    category = "Set 10"
    description = "Classify the image against arbitrary text labels using CLIP ViT-B/32."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        dialog = _ZeroShotDialog(parent)
        parent.wait_window(dialog)
        if dialog.result is None:
            return None

        labels, template = dialog.result
        probs = zero_shot_classify(document.current, labels, template=template)

        subtitle = f'template: "{template}"'
        output = render_zero_shot_figure(document.current, labels, probs, subtitle)

        order = np.argsort(probs)[::-1]
        top_label = labels[order[0]]
        top_prob = float(probs[order[0]])

        details = {
            "Model": CLIP_MODEL_NAME,
            "Template": template,
            "Top prediction": f"{top_label} ({top_prob:.2f})",
            **{f"#{rank + 1} {labels[idx]}": f"{probs[idx]:.4f}" for rank, idx in enumerate(order)},
        }
        return ToolResult(
            image=output,
            message=f"CLIP ViT-B/32: {top_label} ({top_prob:.2f})",
            details=details,
        )


class CLIPTextSearchTool(ForensicsTool):
    tool_id = "clip_text_search"
    title = "CLIP text-to-image search"
    category = "Set 10"
    description = "Search an image folder by text query using CLIP ViT-B/32."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        dialog = _TextSearchDialog(parent)
        parent.wait_window(dialog)
        if dialog.result is None:
            return None

        folder, query, top_k = dialog.result
        matches = search_folder(folder, query, top_k=top_k)
        output = render_search_figure(query, matches)

        best_path, best_sim = matches[0]
        details = {
            "Model": CLIP_MODEL_NAME,
            "Folder": folder.name,
            "Query": query,
            **{f"#{rank + 1} {path.name}": f"sim={sim:.4f}" for rank, (path, sim) in enumerate(matches)},
        }
        return ToolResult(
            image=output,
            message=f'Top match for "{query}": {best_path.name} (sim={best_sim:.3f})',
            details=details,
        )
