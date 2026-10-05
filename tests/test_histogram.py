import unittest
import numpy as np
from PIL import Image

from forensics_app.tools.histogram import (
    compute_channel_histogram,
    plot_skimage_histogram,
)


class HistogramTests(unittest.TestCase):
    def setUp(self) -> None:
        self.channel = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
        self.rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = 50
        self.rgb[:, :, 1] = 100
        self.rgb[:, :, 2] = 150

    def test_compute_channel_histogram(self) -> None:
        hist, centers = compute_channel_histogram(self.channel)
        self.assertEqual(len(hist), 256)
        self.assertEqual(hist[0], 1)
        self.assertEqual(hist[50], 1)
        self.assertEqual(hist[255], 1)
        self.assertEqual(hist[1], 0)

    def test_plot_skimage_histogram_rgb(self) -> None:
        plot_img = plot_skimage_histogram(self.rgb, show_r=True, show_g=True, show_b=True, title="Test Hist")
        self.assertIsInstance(plot_img, Image.Image)
        self.assertGreater(plot_img.width, 100)
        self.assertGreater(plot_img.height, 100)

    def test_plot_skimage_histogram_partial_channels(self) -> None:
        plot_img = plot_skimage_histogram(self.rgb, show_r=True, show_g=False, show_b=True, title="Red and Blue")
        self.assertIsInstance(plot_img, Image.Image)

    def test_plot_skimage_histogram_grayscale(self) -> None:
        gray = self.rgb[:, :, 0]
        plot_img = plot_skimage_histogram(gray, title="Grayscale Hist")
        self.assertIsInstance(plot_img, Image.Image)


if __name__ == "__main__":
    unittest.main()

