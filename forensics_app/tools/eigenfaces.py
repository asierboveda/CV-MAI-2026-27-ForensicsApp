"""Display eigenfaces computed from the LFW dataset."""

from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
from tkinter import filedialog, simpledialog

from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from matplotlib.figure import Figure
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def load_faces(data_folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = data_folder / "lfw_people.npz"
    if not data_path.exists():
        with TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "lfw_home").symlink_to(data_folder, target_is_directory=True)
            lfw_people = fetch_lfw_people(data_home=temp_dir, min_faces_per_person=70, resize=0.4, download_if_missing=False)
        np.savez(data_path, images=lfw_people.images, target=lfw_people.target, target_names=lfw_people.target_names)
    with np.load(data_path) as data:
        return data["images"], data["target"], data["target_names"]


def save_model(pca, train_coefficients, test_coefficients, y_train, y_test, target_names, train_indices, test_indices, image_shape, path):
    np.savez(
        path,
        mean=pca.mean_,
        eigenfaces=pca.components_,
        train_coefficients=train_coefficients,
        test_coefficients=test_coefficients,
        explained_variance=pca.explained_variance_,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
        train_indices=train_indices,
        test_indices=test_indices,
        image_shape=image_shape,
    )


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Build a figure without relying on pyplot's process-wide backend."""
    figure = Figure(figsize=(1.8 * n_col, 2.4 * n_row), dpi=100)
    figure.subplots_adjust(bottom=0.03, left=0.03, right=0.97, top=0.94,
                           hspace=0.35, wspace=0.15)
    for i, image in enumerate(images):
        axis = figure.add_subplot(n_row, n_col, i + 1)
        # Normalize each signed component for display only; keep raw PCA values.
        axis.imshow(image.reshape((h, w)), cmap="gray")
        axis.set_title(titles[i], size=12)
        axis.set_xticks(())
        axis.set_yticks(())
    return figure


class EigenfacesTool(ForensicsTool):
    tool_id = "eigenfaces"
    title = "Eigenfaces"
    category = "Set 6"
    description = "Compute and visualize eigenfaces using PCA."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        default_folder = Path(__file__).resolve().parent.parent / "images" / "set6" / "eigenfaces"
        selected = filedialog.askdirectory(parent=parent, title="Select LFW folder", initialdir=str(default_folder))
        if not selected:
            return None
        data_folder = Path(selected)
        images, y, target_names = load_faces(data_folder)
        n_samples, h, w = images.shape
        X = images.reshape(n_samples, -1)
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X, y, np.arange(n_samples), test_size=0.25, random_state=42
        )
        max_components = min(X_train.shape)
        n_components = simpledialog.askinteger(
            self.title, "Number of PCA components:", parent=parent,
            initialvalue=min(100, max_components), minvalue=1, maxvalue=max_components,
        )
        if n_components is None:
            return None
        k = simpledialog.askinteger(
            self.title, "Number of eigenfaces to display:", parent=parent,
            initialvalue=min(12, n_components), minvalue=1, maxvalue=n_components,
        )
        if k is None:
            return None

        pca = PCA(n_components=n_components, svd_solver="randomized", whiten=False, random_state=42).fit(X_train)
        eigenfaces = pca.components_.reshape((n_components, h, w))
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        save_model(pca, X_train_pca, X_test_pca, y_train, y_test, target_names,
                   train_indices, test_indices, (h, w), data_folder / "eigenfaces_model.npz")

        columns = min(4, k)
        rows = (k + columns - 1) // columns
        eigenface_titles = [f"eigenface {i}" for i in range(k)]
        figure = plot_gallery(eigenfaces[:k], eigenface_titles, h, w, n_row=rows, n_col=columns)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        gallery = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")

        return ToolResult(
            image=gallery,
            message=f"Displayed {k} eigenfaces.",
            details={"Components": n_components, "Eigenfaces shown": k,
                     "Training images": X_train_pca.shape[0], "Test images": X_test_pca.shape[0],
                     "People": len(target_names), "Model": str(data_folder / "eigenfaces_model.npz"),
                     "Projection": "w = U.T @ (x - mean)", "Explained variance": f"{pca.explained_variance_ratio_.sum():.1%}"},
        )
