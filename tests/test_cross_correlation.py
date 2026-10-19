import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.cross_correlation import (
    NormalizedCrossCorrelationTool,
    compute_cross_correlation,
    correlation_to_image,
    create_comparison_image,
)


class CrossCorrelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        rng = np.random.RandomState(42)
        arr = rng.randint(40, 220, size=(40, 40), dtype=np.uint8)
        self.image = Image.fromarray(arr, mode="L")
        self.document.current = self.image

        # Template is the 10x10 cropped patch from (15, 10)
        self.template = Image.fromarray(arr[15:25, 10:20], mode="L")

    def test_registered(self) -> None:
        tools = [t for t in build_tool_registry().all() if t.tool_id == "cross_correlation"]
        self.assertEqual(len(tools), 1, "cross_correlation must be registered")
        tool = tools[0]
        self.assertIsInstance(tool, NormalizedCrossCorrelationTool)
        self.assertEqual(tool.category, "Set 4")

    def test_compute_cross_correlation_finds_peak(self) -> None:
        corr, (py, px), pval = compute_cross_correlation(self.image, self.template, pad_input=True)
        self.assertEqual(corr.shape, (40, 40))
        # With pad_input=True, peak is at the center of the patch (around y=19/20, x=14/15)
        self.assertAlmostEqual(pval, 1.0, places=3)
        self.assertTrue(14 <= py <= 25)
        self.assertTrue(9 <= px <= 20)

    def test_compute_cross_correlation_rejects_oversized_template(self) -> None:
        oversized = Image.new("L", (50, 50), 255)
        with self.assertRaises(ValueError):
            compute_cross_correlation(self.image, oversized)

    def test_correlation_to_image_grayscale_and_colormap(self) -> None:
        corr = np.linspace(-1, 1, 100).reshape((10, 10))
        img_gray = correlation_to_image(corr, colormap="gray")
        self.assertEqual(img_gray.size, (10, 10))
        self.assertEqual(img_gray.mode, "L")

        img_color = correlation_to_image(corr, colormap="viridis")
        self.assertEqual(img_color.size, (10, 10))
        self.assertEqual(img_color.mode, "RGB")

    def test_create_comparison_image(self) -> None:
        corr = np.zeros((40, 40), dtype=float)
        corr[20, 15] = 1.0
        composite = create_comparison_image(
            self.image, self.template, corr, (20, 15), threshold=0.8
        )
        self.assertIsInstance(composite, Image.Image)
        self.assertEqual(composite.mode, "RGB")

    def test_tool_run_cancelled(self) -> None:
        tool = NormalizedCrossCorrelationTool()
        with patch("forensics_app.tools.cross_correlation.CrossCorrelationDialog") as mock_dialog:
            mock_dialog.return_value.result = None
            res = tool.run(None, self.document)
            self.assertIsNone(res)

    def test_tool_run_applied(self) -> None:
        tool = NormalizedCrossCorrelationTool()
        with patch("forensics_app.tools.cross_correlation.CrossCorrelationDialog") as mock_dialog, \
             patch("forensics_app.tools.cross_correlation.CorrelationPlotWindow") as mock_window, \
             patch("PIL.Image.open") as mock_open:
            mock_open.return_value.__enter__.return_value = self.template
            mock_dialog.return_value.result = {
                "template_path": "fake_path.png",
                "pad_input": True,
                "threshold": 0.8,
                "show_3d_plot": True,
                "show_overview": True,
                "output_mode": "Correlation Map (Normalized Grayscale)",
            }
            res = tool.run(None, self.document)
            self.assertIsNotNone(res)
            self.assertIsNone(res.image)
            self.assertIn("Peak correlation", res.details)
            mock_window.assert_called_once()


if __name__ == "__main__":
    unittest.main()
