import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.corner_detection import detect_corners


class CornerDetectionTests(unittest.TestCase):
    def setUp(self):
        # Create a synthetic image with high contrast checkerboard pattern
        arr = np.zeros((60, 60, 3), dtype=np.uint8)
        arr[10:30, 10:30] = 255
        arr[30:50, 30:50] = 255
        self.image = Image.fromarray(arr)
        self.document = ImageDocument()
        self.document.current = self.image

    def tool(self):
        tools = [t for t in build_tool_registry().all() if t.tool_id == "corner_detection"]
        self.assertEqual(len(tools), 1, "Corner detection tool must be registered")
        return tools[0]

    def test_fast_corner_detection(self):
        points, response, visual = detect_corners(
            self.image,
            method="FAST",
            peak_threshold_rel=0.1,
            min_distance=2,
            fast_threshold=0.1,
            fast_n=9,
        )
        self.assertIsInstance(points, np.ndarray)
        self.assertGreater(len(points), 0)
        self.assertEqual(visual.size, self.image.size)
        self.assertEqual(visual.mode, "RGB")

    def test_harris_corner_detection(self):
        points, response, visual = detect_corners(
            self.image,
            method="Harris",
            peak_threshold_rel=0.05,
            min_distance=2,
            harris_k=0.05,
            harris_sigma=1.0,
        )
        self.assertIsInstance(points, np.ndarray)
        self.assertGreater(len(points), 0)
        self.assertEqual(visual.size, self.image.size)
        self.assertEqual(visual.mode, "RGB")

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            detect_corners(self.image, method="UNKNOWN")

    def test_tool_run_fast(self):
        tool = self.tool()
        cfg = {
            "method": "FAST",
            "peak_threshold_rel": 0.1,
            "min_distance": 3,
            "fast_threshold": 0.15,
            "fast_n": 9,
            "harris_k": 0.05,
            "harris_sigma": 1.0,
        }
        with patch("forensics_app.tools.corner_detection.CornerDetectionDialog") as mock_dialog:
            mock_dialog.return_value.result = cfg
            result = tool.run(None, self.document)

        self.assertIsNotNone(result)
        self.assertEqual(result.image.size, self.image.size)
        self.assertIn("corners", result.message.lower())
        self.assertEqual(result.details["Method"], "FAST")
        self.assertEqual(result.details["Peak threshold (rel)"], 0.1)

    def test_tool_run_harris(self):
        tool = self.tool()
        cfg = {
            "method": "Harris",
            "peak_threshold_rel": 0.05,
            "min_distance": 3,
            "fast_threshold": 0.15,
            "fast_n": 12,
            "harris_k": 0.04,
            "harris_sigma": 1.5,
        }
        with patch("forensics_app.tools.corner_detection.CornerDetectionDialog") as mock_dialog:
            mock_dialog.return_value.result = cfg
            result = tool.run(None, self.document)

        self.assertIsNotNone(result)
        self.assertEqual(result.details["Method"], "Harris")
        self.assertEqual(result.details["Harris k"], 0.04)
        self.assertEqual(result.details["Harris sigma"], 1.5)

    def test_cancel_returns_none(self):
        tool = self.tool()
        with patch("forensics_app.tools.corner_detection.CornerDetectionDialog") as mock_dialog:
            mock_dialog.return_value.result = None
            result = tool.run(None, self.document)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
