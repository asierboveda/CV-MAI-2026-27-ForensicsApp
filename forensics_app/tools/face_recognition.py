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
            projections = (images[train].reshape(len(train), -1) - model['mean']) @ model['eigenfaces'].T
            
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
        
        k = simpledialog.askinteger(
            self.title, 'Number of neighbors k (0 for GridSearch):',
            parent=parent, minvalue=0, maxvalue=len(train),
        )
        if k is None:
            return None
            
        details = {}
        if k > 0:
            classifier = KNeighborsClassifier(n_neighbors=k).fit(coefficients, model['y_train'])
        else:
            _, counts = np.unique(model['y_train'], return_counts=True)
            search = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9, 11]}, cv=5)
            search.fit(coefficients, model['y_train'])
            classifier = search.best_estimator_
            details['Best CV accuracy'] = f'{search.best_score_:.2%}'
        weights = (query.ravel() - model['mean']) @ model['eigenfaces'].T
        if model.get('whiten', False):
            # Match PCA.transform, including its guard for near-zero variance.
            scale = np.sqrt(model['explained_variance'])
            weights /= np.maximum(scale, np.finfo(scale.dtype).eps)
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
