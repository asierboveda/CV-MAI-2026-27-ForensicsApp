"""Regression tests for image modes, feature extraction, and dialog boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.blur import BlurTool
from forensics_app.tools.descriptor_extractor import extract_descriptors
from forensics_app.tools.feature_matching import FeatureMatchingDialog, match_features_between
from forensics_app.tools.folder_match import FolderMatchingTool
from forensics_app.tools.hog import HOGTool
from forensics_app.tools.person_detection import PersonDetectionTool
from forensics_app.tools.skimage_filters import SkimageFiltersTool


class ReviewRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.image = Image.fromarray(
            np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        )
        self.document = ImageDocument()
        self.document.current = self.image

    def test_person_detection_compares_compatible_descriptors(self):
        template = self.folder / "template.png"
        self.image.save(template)
        with patch(
            "forensics_app.tools.person_detection.filedialog.askopenfilename",
            return_value=str(template),
        ):
            result = PersonDetectionTool().run(None, self.document)
        self.assertIsNotNone(result)
        self.assertEqual(result.image.getpixel((0, 0)), (0, 255, 255))
        np.testing.assert_array_equal(self.document.current, self.image)

    def test_person_detection_cancel_returns_none(self):
        with patch(
            "forensics_app.tools.person_detection.filedialog.askopenfilename",
            return_value="",
        ):
            self.assertIsNone(PersonDetectionTool().run(None, self.document))

    def test_sift_extractor_accepts_an_image_without_features(self):
        blank = Image.new("RGB", (64, 64))
        points, descriptors, visual = extract_descriptors(blank, method="SIFT")
        self.assertEqual(points.shape, (0, 2))
        self.assertEqual(descriptors.shape, (0, 128))
        self.assertEqual(visual.size, blank.size)

    def test_matching_dialog_rejects_nonpositive_sift_counts(self):
        image_path = self.folder / "image.png"
        self.image.save(image_path)
        for octaves, scales in ((0, 3), (-1, 3), (8, 0), (8, -1)):
            with self.subTest(octaves=octaves, scales=scales):
                dialog = Mock()
                dialog.path_entry.get.return_value = str(image_path)
                dialog.max_ratio_entry.get.return_value = "0.8"
                dialog.method_combo.get.return_value = "SIFT"
                dialog.n_octaves_entry.get.return_value = str(octaves)
                dialog.n_scales_entry.get.return_value = str(scales)
                dialog.sigma_min_entry.get.return_value = "1.6"
                with patch("forensics_app.tools.feature_matching.messagebox.showerror"):
                    self.assertFalse(FeatureMatchingDialog.validate(dialog))

    def test_sift_matching_reports_a_tiny_image_as_invalid(self):
        with self.assertRaisesRegex(ValueError, "at least 6"):
            match_features_between(self.image, Image.new("RGB", (1, 1)))

    def test_folder_search_skips_a_tiny_image_and_continues(self):
        Image.new("RGB", (1, 1)).save(self.folder / "a_tiny.png")
        self.image.save(self.folder / "b_good.png")
        with patch(
            "forensics_app.tools.folder_match.filedialog.askdirectory",
            return_value=str(self.folder),
        ):
            result = FolderMatchingTool().run(None, self.document)
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Best image"], "b_good.png")
        self.assertEqual(result.details["Images compared"], 1)

    def test_hog_rejects_blocks_that_do_not_fit(self):
        for size, parameters in (
            ((8, 8), [9, 3, 3, 3, 3]),
            ((64, 64), [9, 3, 3, 32, 32]),
        ):
            with self.subTest(size=size):
                self.document.current = Image.new("RGB", size)
                with (
                    patch("forensics_app.tools.hog.simpledialog.askinteger", side_effect=parameters),
                    patch("tkinter.messagebox.showerror"),
                ):
                    self.assertIsNone(HOGTool().run(None, self.document))
                self.assertEqual(self.document.current.size, size)

    def test_blur_accepts_binary_and_palette_images(self):
        binary = Image.new("1", (8, 8), 1)
        palette = Image.new("P", (8, 8), 1)
        palette.putpalette([0, 0, 0, 20, 140, 200] + [0] * 762)
        for image, expected in ((binary, (255, 255, 255)), (palette, (20, 140, 200))):
            with self.subTest(mode=image.mode):
                self.document.current = image
                with patch("forensics_app.tools.blur.simpledialog.askfloat", return_value=1.0):
                    result = BlurTool().run(None, self.document)
                self.assertEqual(result.image.convert("RGB").getpixel((0, 0)), expected)
                self.assertEqual(self.document.current.mode, image.mode)

    def test_skimage_gaussian_preserves_binary_white_intensity(self):
        self.document.current = Image.new("1", (8, 8), 1)
        with patch("forensics_app.tools.skimage_filters.SkimageFilterDialog") as dialog:
            dialog.return_value.result = ("Gaussian", 1.0, None)
            result = SkimageFiltersTool().run(None, self.document)
        self.assertEqual(result.image.getextrema(), (255, 255))


if __name__ == "__main__":
    unittest.main()
