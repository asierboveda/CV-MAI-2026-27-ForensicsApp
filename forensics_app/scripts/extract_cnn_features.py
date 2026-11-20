"""Set 8.3 — extract CNN feature vectors of the CIFAR-10 test set once.

Saves the penultimate-layer activations so the "Feature projection" tool
can load them instantly instead of running the CNN.

Usage (from the project root):
    .venv/bin/python forensics_app/scripts/extract_cnn_features.py [model_name] [n_samples]

Output: forensics_app/models/set8/<model_name>_features.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SET8_DIR = Path(__file__).resolve().parents[1] / "models" / "set8"


def main(model_name: str = "vgg16_cifar10", n_samples: int = 1000) -> Path:
    import keras
    from keras.datasets import cifar10

    passport = json.loads((SET8_DIR / f"{model_name}.json").read_text())
    model = keras.models.load_model(SET8_DIR / f"{model_name}.keras", compile=False)
    extractor = keras.Model(model.inputs, model.layers[-2].output)  # layer before the softmax

    _, (x_test, y_test) = cifar10.load_data()
    x = keras.ops.image.resize(x_test[:n_samples], passport["preprocessing"]["input_size"])
    features = extractor.predict(np.asarray(x) / 255.0, batch_size=64)

    out = SET8_DIR / f"{model_name}_features.npz"
    np.savez_compressed(out, features=features, labels=y_test[:n_samples].ravel(),
                        class_names=np.array(passport["class_names"]))
    print(f"Saved {features.shape} features to {out}")
    return out


if __name__ == "__main__":
    main(*(sys.argv[1:2]), *(int(n) for n in sys.argv[2:3]))
