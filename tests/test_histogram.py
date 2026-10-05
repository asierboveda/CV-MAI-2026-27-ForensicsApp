import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.tools.histogram import compute_histograms
from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class HistogramTests(unittest.TestCase):
    def test_registered_tool_is_analysis_only(self):
        tool = next(t for t in build_tool_registry().all() if t.tool_id == "histogram")
        document = ImageDocument()
        document.current = Image.new("RGB", (2, 1), (200, 50, 20))
        before = document.current.tobytes()
        with patch("forensics_app.tools.histogram.HistogramWindow") as window:
            result = tool.run(None, document)
        window.assert_called_once()
        self.assertIsNone(result.image)
        self.assertEqual(document.current.tobytes(), before)
        self.assertFalse(document.can_undo)

    def test_rgb_counts_each_channel(self):
        image = Image.new("RGB", (2, 1), (200, 50, 20))
        image.putpixel((1, 0), (0, 50, 255))
        histograms = compute_histograms(image)
        self.assertEqual(set(histograms), {"R", "G", "B"})
        self.assertEqual(histograms["R"][200], 1)
        self.assertEqual(histograms["R"][0], 1)
        self.assertEqual(histograms["G"][50], 2)
        self.assertEqual(histograms["B"][255], 1)
        for counts in histograms.values():
            self.assertEqual(len(counts), 256)
            self.assertEqual(sum(counts), 2)
        self.assertEqual(image.getpixel((0, 0)), (200, 50, 20))

    def test_grayscale_has_one_curve(self):
        image = Image.new("L", (2, 1), 75)
        counts = compute_histograms(image)
        self.assertEqual(list(counts), ["Intensity"])
        self.assertEqual(counts["Intensity"][75], 2)

    def test_alpha_does_not_become_a_color_curve(self):
        counts = compute_histograms(Image.new("RGBA", (1, 1), (20, 30, 40, 0)))
        self.assertEqual(set(counts), {"R", "G", "B"})
        self.assertEqual(counts["R"][20], 1)

    def test_palette_image_is_converted_to_rgb(self):
        image = Image.new("P", (1, 1))
        image.putpalette([255, 0, 0] + [0] * 765)
        counts = compute_histograms(image)
        self.assertEqual(counts["R"][255], 1)

    def test_binary_image_is_grayscale(self):
        counts = compute_histograms(Image.new("1", (1, 1), 1))
        self.assertEqual(counts["Intensity"][255], 1)
