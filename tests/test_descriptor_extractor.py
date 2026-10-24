import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.descriptor_extractor import extract_descriptors


class DescriptorExtractorTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        arr = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        self.image = Image.fromarray(arr)
        self.document = ImageDocument()
        self.document.current = self.image

    def tool(self):
        tools = [t for t in build_tool_registry().all() if t.tool_id == "descriptor_extractor"]
        self.assertEqual(len(tools), 1, "Descriptor extractor tool must be registered")
        return tools[0]

    def test_orb_extraction(self):
        kp, desc, visual = extract_descriptors(self.image, method="ORB", n_keypoints=50)
        self.assertIsInstance(kp, np.ndarray)
        self.assertGreater(len(kp), 0)
        self.assertEqual(len(kp), len(desc))
        self.assertEqual(visual.size, self.image.size)
        self.assertEqual(visual.mode, "RGB")

    def test_sift_extraction(self):
        kp, desc, visual = extract_descriptors(
            self.image, method="SIFT", n_octaves=4, n_scales=3, sigma_min=1.6
        )
        self.assertIsInstance(kp, np.ndarray)
        self.assertGreater(len(kp), 0)
        self.assertEqual(len(kp), len(desc))
        self.assertEqual(visual.size, self.image.size)
        self.assertEqual(visual.mode, "RGB")

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            extract_descriptors(self.image, method="UNKNOWN")

    def test_tool_run_orb(self):
        tool = self.tool()
        cfg = {
            "method": "ORB",
            "n_keypoints": 60,
            "n_octaves": 8,
            "n_scales": 3,
            "sigma_min": 1.6,
        }
        with patch("forensics_app.tools.descriptor_extractor.DescriptorExtractorDialog") as mock_dialog:
            mock_dialog.return_value.result = cfg
            result = tool.run(None, self.document)

        self.assertIsNotNone(result)
        self.assertEqual(result.image.size, self.image.size)
        self.assertEqual(result.details["Method"], "ORB")
        self.assertEqual(result.details["Max keypoints (n_keypoints)"], 60)

    def test_tool_run_sift(self):
        tool = self.tool()
        cfg = {
            "method": "SIFT",
            "n_keypoints": 200,
            "n_octaves": 4,
            "n_scales": 2,
            "sigma_min": 1.4,
        }
        with patch("forensics_app.tools.descriptor_extractor.DescriptorExtractorDialog") as mock_dialog:
            mock_dialog.return_value.result = cfg
            result = tool.run(None, self.document)

        self.assertIsNotNone(result)
        self.assertEqual(result.details["Method"], "SIFT")
        self.assertEqual(result.details["Octaves (n_octaves)"], 4)
        self.assertEqual(result.details["Scales per octave (n_scales)"], 2)
        self.assertEqual(result.details["Initial sigma (sigma_min)"], 1.4)

    def test_cancel_returns_none(self):
        tool = self.tool()
        with patch("forensics_app.tools.descriptor_extractor.DescriptorExtractorDialog") as mock_dialog:
            mock_dialog.return_value.result = None
            result = tool.run(None, self.document)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
