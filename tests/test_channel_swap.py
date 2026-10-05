import unittest
import numpy as np
from PIL import Image

from forensics_app.tools.channel_swap import (
    swap_channels,
    create_swap_comparison,
)


class ChannelSwapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = 200  # Red
        self.rgb[:, :, 1] = 100  # Green
        self.rgb[:, :, 2] = 50   # Blue

    def test_swap_red_and_blue(self) -> None:
        swapped = swap_channels(self.rgb, 0, 2)
        self.assertTrue(np.all(swapped[:, :, 0] == 50))
        self.assertTrue(np.all(swapped[:, :, 1] == 100))
        self.assertTrue(np.all(swapped[:, :, 2] == 200))

    def test_swap_same_channel_identity(self) -> None:
        swapped = swap_channels(self.rgb, 1, 1)
        self.assertTrue(np.array_equal(swapped, self.rgb))

    def test_swap_grayscale_fallback(self) -> None:
        gray = np.full((10, 10), 128, dtype=np.uint8)
        res = swap_channels(gray, 0, 2)
        self.assertTrue(np.array_equal(res, gray))

    def test_create_swap_comparison(self) -> None:
        swapped = swap_channels(self.rgb, 0, 2)
        comp_img = create_swap_comparison(self.rgb, swapped)
        self.assertIsInstance(comp_img, Image.Image)


if __name__ == "__main__":
    unittest.main()
