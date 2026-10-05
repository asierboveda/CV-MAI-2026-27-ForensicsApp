import unittest
import numpy as np
from PIL import Image

from forensics_app.tools.channel_split import (
    split_rgb_channels,
    combine_rgb_channels,
    create_3channel_figure,
    extract_single_channel,
    create_channel_comparison_grid,
)


class ChannelSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = 255  # Red
        self.rgb[:, :, 1] = 128  # Green
        self.rgb[:, :, 2] = 64   # Blue

    def test_split_rgb_channels(self) -> None:
        r, g, b = split_rgb_channels(self.rgb)
        self.assertEqual(r.shape, (20, 30))
        self.assertEqual(g.shape, (20, 30))
        self.assertEqual(b.shape, (20, 30))
        self.assertTrue(np.all(r == 255))
        self.assertTrue(np.all(g == 128))
        self.assertTrue(np.all(b == 64))

    def test_combine_rgb_channels(self) -> None:
        r, g, b = split_rgb_channels(self.rgb)
        combined = combine_rgb_channels(r, g, b)
        self.assertEqual(combined.shape, (20, 30, 3))
        self.assertTrue(np.array_equal(combined, self.rgb))

    def test_create_3channel_figure(self) -> None:
        r, g, b = split_rgb_channels(self.rgb)
        fig_img = create_3channel_figure(r, g, b, title="Test")
        self.assertIsInstance(fig_img, Image.Image)
        self.assertGreater(fig_img.width, 100)

    def test_extract_single_channel_grayscale(self) -> None:
        r_gray = extract_single_channel(self.rgb, 0, colorized=False)
        self.assertEqual(r_gray.shape, (20, 30))
        self.assertTrue(np.all(r_gray == 255))

    def test_extract_single_channel_colorized(self) -> None:
        g_col = extract_single_channel(self.rgb, 1, colorized=True)
        self.assertEqual(g_col.shape, (20, 30, 3))
        self.assertTrue(np.all(g_col[:, :, 0] == 0))
        self.assertTrue(np.all(g_col[:, :, 1] == 128))
        self.assertTrue(np.all(g_col[:, :, 2] == 0))

    def test_create_channel_comparison_grid(self) -> None:
        grid_img = create_channel_comparison_grid(self.rgb, title="Test")
        self.assertIsInstance(grid_img, Image.Image)
        self.assertGreater(grid_img.width, 100)
        self.assertGreater(grid_img.height, 50)


if __name__ == "__main__":
    unittest.main()
