from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.tools import build_tool_registry
from forensics_app.tools.clip import (
    CLIP_MODEL_NAME,
    format_prompt,
    parse_labels,
    render_search_figure,
    render_zero_shot_figure,
    search_folder,
    zero_shot_classify,
)


class _FakeScale:
    def exp(self):
        return SimpleNamespace(item=lambda: 100.0)


class CLIPToolTests(unittest.TestCase):
    def test_parse_labels_and_format_prompt(self) -> None:
        self.assertEqual(parse_labels(" cat, dog ,\n tiger , "), ["cat", "dog", "tiger"])
        self.assertEqual(parse_labels(" , \n "), [])
        self.assertEqual(format_prompt("a photo of a {}.", "cat"), "a photo of a cat.")
        self.assertEqual(format_prompt("a photo of", "cat"), "a photo of cat")
        self.assertEqual(format_prompt("", "cat"), "cat")

    def test_zero_shot_classify(self) -> None:
        img = Image.new("RGB", (32, 32), "red")
        fake_model = SimpleNamespace(logit_scale=_FakeScale())

        def fake_embed_images(_imgs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

        def fake_embed_texts(texts):
            out = []
            for t in texts:
                if "cat" in t:
                    out.append([1.0, 0.0])
                else:
                    out.append([0.0, 1.0])
            return np.array(out, dtype=np.float32)

        with (
            patch("forensics_app.tools.clip._load_clip", return_value=(fake_model, None, "cpu")),
            patch("forensics_app.tools.clip.embed_images", side_effect=fake_embed_images),
            patch("forensics_app.tools.clip.embed_texts", side_effect=fake_embed_texts),
        ):
            probs = zero_shot_classify(img, ["cat", "dog"], template="a photo of a {}.")
            self.assertEqual(probs.shape, (2,))
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
            self.assertGreater(probs[0], probs[1])

    def test_zero_shot_requires_labels(self) -> None:
        img = Image.new("RGB", (16, 16), "blue")
        with self.assertRaises(ValueError):
            zero_shot_classify(img, [])

    def test_render_zero_shot_figure(self) -> None:
        img = Image.new("RGB", (32, 32), "green")
        rendered = render_zero_shot_figure(img, ["a", "b"], np.array([0.2, 0.8]), 'template: "a photo of {}."')
        self.assertEqual(rendered.mode, "RGB")
        self.assertGreater(rendered.width, 100)

    def test_search_folder_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            Image.new("RGB", (16, 16), "red").save(folder / "a_red.png")
            Image.new("RGB", (16, 16), "blue").save(folder / "b_blue.png")
            Image.new("RGB", (16, 16), "green").save(folder / "c_green.png")

            def fake_embed_images(imgs):
                # Give b_blue highest similarity to [0, 1]
                return np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)[: len(imgs)]

            with (
                patch("forensics_app.tools.clip.embed_images", side_effect=fake_embed_images),
                patch("forensics_app.tools.clip.embed_texts", return_value=np.array([[0.0, 1.0]], dtype=np.float32)),
            ):
                matches = search_folder(folder, "something blue", top_k=2)

            self.assertEqual(len(matches), 2)
            self.assertEqual(matches[0][0].name, "b_blue.png")
            self.assertGreater(matches[0][1], matches[1][1])

            grid = render_search_figure("something blue", matches)
            self.assertEqual(grid.mode, "RGB")

    def test_registry_contains_set10_clip_tools(self) -> None:
        registry = build_tool_registry()
        categories = dict(registry.categories())
        self.assertIn("Set 10", categories)
        ids = [tool.tool_id for tool in categories["Set 10"]]
        self.assertEqual(ids, ["clip_zero_shot", "clip_text_search"])
        self.assertEqual(CLIP_MODEL_NAME, "openai/clip-vit-base-patch32")


if __name__ == "__main__":
    unittest.main()
