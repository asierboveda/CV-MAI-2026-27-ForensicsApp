import unittest

from PIL import Image

from forensics_app.tools.masking import compose_masked


class MaskingTests(unittest.TestCase):
    def setUp(self):
        self.base = Image.new("RGB", (2, 1), (100, 100, 100))
        self.coat = Image.new("RGB", (2, 1), "black")
        self.coat.putpixel((1, 0), (200, 150, 0))

    def test_black_background_is_not_copied(self):
        result, mask = compose_masked(self.base, self.coat, 10)
        self.assertEqual(mask.tobytes(), bytes([0, 255]))
        self.assertEqual(result.getpixel((0, 0)), (100, 100, 100))
        self.assertEqual(result.getpixel((1, 0)), (200, 150, 0))
        self.assertEqual(self.base.getpixel((1, 0)), (100, 100, 100))

    def test_texture_only_fills_selected_pixels(self):
        texture = Image.new("RGB", (1, 1), (255, 0, 0))
        result, mask = compose_masked(self.base, self.coat, 10, texture)
        self.assertEqual(result.getpixel((0, 0)), (100, 100, 100))
        self.assertEqual(result.getpixel((1, 0)), (255, 0, 0))

    def test_threshold_boundary_and_empty_mask(self):
        self.coat = Image.new("RGB", (2, 1), (10, 0, 0))
        result, mask = compose_masked(self.base, self.coat, 10)
        self.assertIsNone(mask.getbbox())
        self.assertEqual(result.tobytes(), self.base.tobytes())

    def test_transparent_overlay_pixels_are_excluded(self):
        coat = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
        result, mask = compose_masked(self.base, coat, 10)
        self.assertIsNone(mask.getbbox())

    def test_preserves_base_alpha_outside_mask(self):
        base = Image.new("RGBA", (2, 1), (100, 100, 100, 128))
        result, mask = compose_masked(base, self.coat, 10)
        self.assertEqual(result.getpixel((0, 0)), (100, 100, 100, 128))

    def test_rejects_mismatched_images(self):
        with self.assertRaisesRegex(ValueError, "same size"):
            compose_masked(self.base, Image.new("RGB", (1, 1)), 10)
