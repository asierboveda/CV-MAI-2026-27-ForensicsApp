import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.feature_matching import match_features_between


class FeatureMatchingTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        arr1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # arr2 is shifted version with some overlap
        arr2 = np.roll(arr1, shift=5, axis=(0, 1))

        self.image1 = Image.fromarray(arr1)
        self.image2 = Image.fromarray(arr2)

        self.document = ImageDocument()
        self.document.current = self.image1

        # Save temporary image for dialog file browsing test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "test2.png"
        self.image2.save(self.temp_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def tool(self):
        tools = [t for t in build_tool_registry().all() if t.tool_id == "feature_matching"]
        self.assertEqual(len(tools), 1, "Feature matching tool must be registered")
        return tools[0]

    def test_orb_feature_matching(self):
        kp1, kp2, matches, visual = match_features_between(
            self.image1,
            self.image2,
            method="ORB",
            cross_check=True,
            max_ratio=0.8,
            metric="hamming",
            n_keypoints=100,
        )
        self.assertIsInstance(matches, np.ndarray)
        self.assertGreater(len(kp1), 0)
        self.assertGreater(len(kp2), 0)
        self.assertIsInstance(visual, Image.Image)

    def test_sift_feature_matching(self):
        kp1, kp2, matches, visual = match_features_between(
            self.image1,
            self.image2,
            method="SIFT",
            cross_check=True,
            max_ratio=0.8,
            metric="euclidean",
            n_octaves=4,
            n_scales=3,
            sigma_min=1.6,
        )
        self.assertIsInstance(matches, np.ndarray)
        self.assertGreater(len(kp1), 0)
        self.assertGreater(len(kp2), 0)
        self.assertIsInstance(visual, Image.Image)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            match_features_between(self.image1, self.image2, method="UNKNOWN")

    def test_tool_run_orb(self):
        tool = self.tool()
        cfg = {
            "second_path": str(self.temp_file),
            "method": "ORB",
            "cross_check": True,
            "max_ratio": 0.8,
            "metric": "hamming",
            "n_keypoints": 100,
            "n_octaves": 8,
            "n_scales": 3,
            "sigma_min": 1.6,
        }
        with patch("forensics_app.tools.feature_matching.FeatureMatchingDialog") as mock_dialog:
            mock_dialog.return_value.result = cfg
            result = tool.run(None, self.document)

        self.assertIsNotNone(result)
        self.assertIn("Matched", result.message)
        self.assertEqual(result.details["Method"], "ORB")
        self.assertEqual(result.details["Second image"], "test2.png")
        self.assertEqual(result.details["Metric"], "hamming")
        self.assertTrue(result.details["Cross check"])

    def test_tool_run_sift(self):
        tool = self.tool()
        cfg = {
            "second_path": str(self.temp_file),
            "method": "SIFT",
            "cross_check": False,
            "max_ratio": 0.75,
            "metric": "euclidean",
            "n_keypoints": 100,
            "n_octaves": 4,
            "n_scales": 2,
            "sigma_min": 1.4,
        }
        with patch("forensics_app.tools.feature_matching.FeatureMatchingDialog") as mock_dialog:
            mock_dialog.return_value.result = cfg
            result = tool.run(None, self.document)

        self.assertIsNotNone(result)
        self.assertEqual(result.details["Method"], "SIFT")
        self.assertEqual(result.details["Metric"], "euclidean")
        self.assertFalse(result.details["Cross check"])
        self.assertEqual(result.details["Max ratio"], 0.75)

    def test_cancel_returns_none(self):
        tool = self.tool()
        with patch("forensics_app.tools.feature_matching.FeatureMatchingDialog") as mock_dialog:
            mock_dialog.return_value.result = None
            result = tool.run(None, self.document)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
