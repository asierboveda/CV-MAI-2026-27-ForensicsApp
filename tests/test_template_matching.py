import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.template_matching import (
    TemplateMatchingTool,
    match_template_rotated,
    rotate_template,
    superimpose_detection,
)


class TemplateMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        rng = np.random.RandomState(42)
        arr = rng.randint(40, 220, size=(50, 50), dtype=np.uint8)
        self.image = Image.fromarray(arr, mode="L")
        self.document.current = self.image
        self.template = Image.fromarray(arr[20:30, 20:30], mode="L")

    def test_registered(self) -> None:
        tools = [t for t in build_tool_registry().all() if t.tool_id == "template_matching"]
        self.assertEqual(len(tools), 1, "template_matching must be registered")
        tool = tools[0]
        self.assertIsInstance(tool, TemplateMatchingTool)
        self.assertEqual(tool.category, "Set 4")

    def test_rotate_template(self) -> None:
        rot0 = rotate_template(self.template, 0.0)
        self.assertEqual(rot0.shape, (10, 10))
        rot45 = rotate_template(self.template, 45.0)
        self.assertEqual(rot45.shape, (10, 10))

    def test_match_template_rotated_finds_match(self) -> None:
        corr, tmpl_rot, (py, px), pval = match_template_rotated(
            self.image, self.template, angle=0.0, pad_input=True
        )
        self.assertEqual(corr.shape, (50, 50))
        self.assertAlmostEqual(pval, 1.0, places=3)
        self.assertTrue(19 <= py <= 30)
        self.assertTrue(19 <= px <= 30)

    def test_superimpose_detection_styles(self) -> None:
        corr = np.zeros((50, 50), dtype=float)
        corr[25, 25] = 1.0
        for style in (1, 2, 3, 4):
            with self.subTest(style=style):
                out_img, peaks, count = superimpose_detection(
                    self.image,
                    template_size=(10, 10),
                    corr_map=corr,
                    threshold=0.8,
                    style=style,
                )
                self.assertIsInstance(out_img, Image.Image)
                self.assertEqual(out_img.size, (50, 50))
                self.assertEqual(count, 1)
                self.assertEqual(peaks, [(25, 25)])

    def test_tool_run_cancelled(self) -> None:
        tool = TemplateMatchingTool()
        with patch("forensics_app.tools.template_matching.TemplateMatchingDialog") as mock_dialog:
            mock_dialog.return_value.result = None
            res = tool.run(None, self.document)
            self.assertIsNone(res)

    def test_tool_run_applied(self) -> None:
        tool = TemplateMatchingTool()
        with patch("forensics_app.tools.template_matching.TemplateMatchingDialog") as mock_dialog, \
             patch("forensics_app.tools.template_matching.CorrelationPlotWindow") as mock_window, \
             patch("PIL.Image.open") as mock_open:
            mock_open.return_value.__enter__.return_value = self.template
            mock_dialog.return_value.result = {
                "template_path": "fake_template.png",
                "angle": 15.0,
                "threshold": 0.75,
                "style_idx": 3,
                "style_name": "Darken everything except template detections",
                "show_3d_plot": False,
            }
            res = tool.run(None, self.document)
            self.assertIsNotNone(res)
            self.assertIsNone(res.image)  # Preserves working document image
            self.assertIn("Rotation angle", res.details)
            self.assertEqual(res.details["Rotation angle"], "15°")
            mock_window.assert_called_once()

    def test_dialog_slider_callbacks(self) -> None:
        from forensics_app.tools.template_matching import TemplateMatchingDialog
        dialog = TemplateMatchingDialog.__new__(TemplateMatchingDialog)
        # Calling slider callbacks before entry attributes exist must not raise AttributeError
        dialog._on_angle_slide("45.0")
        dialog._on_thresh_slide("0.85")


if __name__ == "__main__":
    unittest.main()
