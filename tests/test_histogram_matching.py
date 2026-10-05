import unittest
import numpy as np
from PIL import Image

from forensics_app.tools.histogram_matching import (
    match_image_histograms,
    create_matching_comparison_figure,
)


class HistogramMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = np.random.randint(50, 100, (30, 30), dtype=np.uint8)
        self.reference = np.random.randint(150, 200, (30, 30), dtype=np.uint8)

    def test_match_histograms_grayscale(self) -> None:
        matched = match_image_histograms(self.source, self.reference)
        self.assertEqual(matched.shape, self.source.shape)
        self.assertGreater(matched.mean(), self.source.mean())

    def test_create_matching_comparison_figure(self) -> None:
        matched = match_image_histograms(self.source, self.reference)
        fig_img = create_matching_comparison_figure(self.source, self.reference, matched)
        self.assertIsInstance(fig_img, Image.Image)


if __name__ == "__main__":
    unittest.main()
