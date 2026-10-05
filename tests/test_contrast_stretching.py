import unittest
import numpy as np
from PIL import Image

from forensics_app.tools.contrast_stretching import (
    simulate_low_contrast,
    apply_percentile_contrast_stretching,
    apply_histogram_equalization,
    apply_adaptive_equalization,
    stretch_contrast,
)


class ContrastStretchingTests(unittest.TestCase):
    def setUp(self) -> None:
        # Low contrast synthetic array (range 100 to 150)
        self.gray_low = np.random.randint(100, 150, (30, 30), dtype=np.uint8)
        self.rgb_low = np.random.randint(100, 150, (30, 30, 3), dtype=np.uint8)

    def test_simulate_low_contrast(self) -> None:
        full_range = np.arange(256, dtype=np.uint8).reshape(16, 16)
        low = simulate_low_contrast(full_range)
        self.assertEqual(low.shape, (16, 16))
        self.assertGreaterEqual(low.min(), 50)
        self.assertLessEqual(low.max(), 200)

    def test_percentile_stretching_gray(self) -> None:
        res = apply_percentile_contrast_stretching(self.gray_low)
        self.assertEqual(res.shape, (30, 30))
        self.assertEqual(res.min(), 0)
        self.assertEqual(res.max(), 255)

    def test_equalize_hist_gray(self) -> None:
        res = apply_histogram_equalization(self.gray_low)
        self.assertEqual(res.shape, (30, 30))
        self.assertGreaterEqual(res.max(), 200)

    def test_percentile_stretching_rgb(self) -> None:
        res = apply_percentile_contrast_stretching(self.rgb_low)
        self.assertEqual(res.shape, (30, 30, 3))
        self.assertEqual(res.min(), 0)
        self.assertEqual(res.max(), 255)

    def test_clahe_rgb(self) -> None:
        res = apply_adaptive_equalization(self.rgb_low, clip_limit=0.03)
        self.assertEqual(res.shape, (30, 30, 3))


if __name__ == "__main__":
    unittest.main()

