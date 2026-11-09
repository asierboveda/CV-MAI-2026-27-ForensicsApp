"""Recognize exported LFW test faces using the notebook's k-NN workflow."""
from pathlib import Path
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from .base import ForensicsTool, ToolResult
from .eigenfaces import plot_gallery


class FaceRecognitionTool(ForensicsTool):
    tool_id = "face_recognition"
    title = "Face recognition"
    category = "Set 6"
    description = "Recognize a test face with k-NN on saved eigenface coefficients."
    requires_image = False

    def run(self, parent, document):
        selected = filedialog.askdirectory(parent=parent, title="Select Eigenfaces model folder")
        if not selected:
            return None
        folder = Path(selected)
        instruction = "Run Eigenfaces first to regenerate the model and exported test images."
        if not all((folder / name).exists() for name in ("eigenfaces_model.npz", "lfw_people.npz", "test")):
            raise ValueError(instruction)
        try:
            with np.load(folder / "eigenfaces_model.npz") as archive:
                model = {key: archive[key] for key in archive.files}
            with np.load(folder / "lfw_people.npz") as data:
                images, labels, names = data['images'], data['target'], data['target_names']
            train, test = model['train_indices'], model['test_indices']
            shape = tuple(model['image_shape'])
            coefficients = model['train_coefficients']
            if (images.ndim != 3 or shape != images.shape[1:] or
                    train.ndim != 1 or test.ndim != 1 or
                    train.dtype.kind not in 'iu' or test.dtype.kind not in 'iu' or
                    len(train) < 3 or len(test) == 0 or
                    np.any(train < 0) or np.any(test < 0) or
                    np.any(train >= len(images)) or np.any(test >= len(images)) or
                    len(np.unique(np.concatenate((train, test)))) != len(train) + len(test) or
                    not np.array_equal(names, model['target_names']) or
                    not np.array_equal(labels[train], model['y_train']) or
                    not np.array_equal(labels[test], model['y_test'])):
                raise ValueError('Invalid dataset metadata')
            projections = (images[train].reshape(len(train), -1) - model['mean']) @ model['eigenfaces'].T
            if (bool(model.get('whiten', False)) or coefficients.shape != projections.shape or
                    not np.all(np.isfinite(coefficients)) or
                    not np.allclose(coefficients, projections, rtol=1e-4, atol=1e-4)):
                raise ValueError('Incompatible PCA coefficients')
        except (KeyError, ValueError, IndexError, OSError) as error:
            raise ValueError(instruction) from error
        selected = filedialog.askopenfilename(parent=parent, title="Select exported test face", initialdir=str(folder / 'test'), filetypes=[('PNG images', '*.png')])
        if not selected:
            return None
        paths = {(folder / 'test' / str(names[labels[i]]) / f'{i}.png').resolve(): int(i) for i in test}
        query_path = Path(selected).resolve()
        if query_path not in paths:
            raise ValueError('Select an image from the saved test split. ' + instruction)
        index = paths[query_path]
        with Image.open(query_path) as source:
            query = np.asarray(source.convert('L'), dtype=float) / 255
        expected = np.rint(np.clip(images[index], 0, 1) * 255) / 255
        if query.shape != shape or not np.allclose(query, expected, atol=1e-7):
            raise ValueError('Test image does not match the saved dataset. ' + instruction)
        value = simpledialog.askstring(self.title, 'Number of neighbors k (leave blank for GridSearch):', parent=parent)
        if value is None:
            return None
        details = {}
        if value.strip():
            try:
                k = int(value)
            except ValueError as error:
                raise ValueError('k must be a positive integer.') from error
            if not 1 <= k <= len(train):
                raise ValueError(f'k must be between 1 and {len(train)}.')
            classifier = KNeighborsClassifier(n_neighbors=k).fit(coefficients, model['y_train'])
        else:
            _, counts = np.unique(model['y_train'], return_counts=True)
            if counts.min() < 5 or len(train) - int(np.ceil(len(train) / 5)) < 11:
                raise ValueError('GridSearch requires five samples per class and at least 11 training samples in each fold.')
            search = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9, 11]}, cv=5)
            search.fit(coefficients, model['y_train'])
            classifier = search.best_estimator_
            details['Best CV accuracy'] = f'{search.best_score_:.2%}'
        weights = (query.ravel() - model['mean']) @ model['eigenfaces'].T
        prediction = int(classifier.predict(weights[None])[0])
        distances = np.linalg.norm(coefficients - weights, axis=1)
        nearest = np.argsort(distances, kind='stable')[:3]
        titles = [f'Query: {names[labels[index]]}']
        gallery = [query]
        for rank, neighbor in enumerate(nearest, 1):
            name = str(names[model['y_train'][neighbor]])
            gallery.append(images[train[neighbor]])
            titles.append(f'{name}\nd={distances[neighbor]:.4f}')
            details[f'Neighbor {rank}'] = f'{name}: {distances[neighbor]:.6f}'
        figure = plot_gallery(gallery, titles, *shape, n_row=1, n_col=4)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        output = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert('RGB')
        details.update({'True identity': str(names[labels[index]]), 'Proposed identity': str(names[prediction]), 'k': classifier.n_neighbors})
        return ToolResult(message=f'Proposed identity: {names[prediction]}', image=output, details=details)
