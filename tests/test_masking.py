import unittest
import numpy as np
from PIL import Image

from forensics_app.tools.masking import (
    extract_binary_mask,
    apply_binary_threshold_mask,
    composite_texture_or_overlay,
    create_masking_comparison_strip,
    create_coins_mask_comparison,
)


class MaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = np.zeros((40, 40, 3), dtype=np.uint8)
        # Create a central square mask
        self.mask_img = Image.new("RGB", (40, 40), (0, 0, 0))
        for y in range(10, 30):
            for x in range(10, 30):
                self.mask_img.putpixel((x, y), (255, 255, 0))

    def test_extract_binary_mask(self) -> None:
        mask = extract_binary_mask(self.mask_img, threshold=135)
        self.assertEqual(mask.shape, (40, 40))
        self.assertEqual(np.sum(mask), 20 * 20)
        self.assertTrue(mask[15, 15])
        self.assertFalse(mask[5, 5])

    def test_apply_binary_threshold_mask(self) -> None:
        sample = np.array([[50, 150], [200, 100]], dtype=np.uint8)
        mask, masked1, masked2 = apply_binary_threshold_mask(sample, threshold=135)
        self.assertTrue(mask[0, 1])
        self.assertTrue(mask[1, 0])
        self.assertFalse(mask[0, 0])
        self.assertEqual(masked1[0, 1], 150)
        self.assertEqual(masked1[0, 0], 0)
        self.assertEqual(masked2[0, 0], 50)
        self.assertEqual(masked2[0, 1], 0)

    def test_threshold_cutoff_behavior(self) -> None:
        test_img = Image.new("L", (10, 10), 130)
        test_img.putpixel((5, 5), 140)
        mask = extract_binary_mask(test_img, threshold=135)
        self.assertEqual(np.sum(mask), 1)
        self.assertTrue(mask[5, 5])
        self.assertFalse(mask[0, 0])

    def test_composite_direct_overlay(self) -> None:
        mask_np = np.array(self.mask_img)
        res = composite_texture_or_overlay(self.base, mask_np, texture_np=None, mode="overlay")
        self.assertEqual(res.shape, (40, 40, 3))
        # Masked region should now be yellow
        self.assertTrue(np.all(res[15, 15] == [255, 255, 0]))
        # Non-masked region stays black
        self.assertTrue(np.all(res[5, 5] == [0, 0, 0]))

    def test_composite_tiled_texture(self) -> None:
        mask_np = np.array(self.mask_img)
        texture = np.full((10, 10, 3), 180, dtype=np.uint8)
        res = composite_texture_or_overlay(
            self.base, mask_np, texture_np=texture, mode="tiled", preserve_shading=False
        )
        self.assertEqual(res.shape, (40, 40, 3))
        self.assertTrue(np.all(res[15, 15] == [180, 180, 180]))

    def test_create_masking_comparison_strip(self) -> None:
        base_pil = Image.fromarray(self.base)
        coat_pil = self.mask_img
        strip = create_masking_comparison_strip(coat_pil, base_pil, coat_pil, coat_pil)
        self.assertIsInstance(strip, Image.Image)

    def test_create_coins_mask_comparison(self) -> None:
        sample = np.full((20, 20), 150, dtype=np.uint8)
        mask = sample > 135
        m1 = sample * mask
        m2 = np.zeros_like(sample)
        m2[~mask] = sample[~mask]
        comp = create_coins_mask_comparison(sample, mask, m1, m2)
        self.assertIsInstance(comp, Image.Image)


if __name__ == "__main__":
    unittest.main()
