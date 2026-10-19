import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.hog import (
    HOGTool,
    compute_hog,
    create_hog_comparison_image,
)


class HOGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        # Synthetic RGB image with vertical and horizontal stripes
        arr = np.zeros((36, 36, 3), dtype=np.uint8)
        arr[:, ::4] = [255, 0, 0]
        arr[::4, :] = [0, 255, 0]
        self.image = Image.fromarray(arr, mode="RGB")
        self.document.current = self.image

    def test_registered(self) -> None:
        tools = [t for t in build_tool_registry().all() if t.tool_id == "hog_visualization"]
        self.assertEqual(len(tools), 1, "hog_visualization must be registered")
        tool = tools[0]
        self.assertIsInstance(tool, HOGTool)
        self.assertEqual(tool.category, "Set 4")

    def test_compute_hog_basic(self) -> None:
        fd, raw_hog, vis_img = compute_hog(
            self.image,
            orientations=9,
            pixels_per_cell=(6, 6),
            cells_per_block=(2, 2),
        )
        self.assertGreater(len(fd), 0)
        self.assertEqual(vis_img.size, (36, 36))
        self.assertEqual(vis_img.mode, "L")

    def test_compute_hog_orientations(self) -> None:
        for orient in (2, 4, 9):
            with self.subTest(orientations=orient):
                fd, raw_hog, vis_img = compute_hog(
                    self.image,
                    orientations=orient,
                    pixels_per_cell=(6, 6),
                    cells_per_block=(2, 2),
                )
                self.assertGreater(len(fd), 0)
                self.assertEqual(vis_img.size, (36, 36))

    def test_create_hog_comparison_image(self) -> None:
        composite = create_hog_comparison_image(
            self.image,
            orientations_list=(2, 4, 9),
            pixels_per_cell=(6, 6),
            cells_per_block=(2, 2),
        )
        self.assertIsInstance(composite, Image.Image)
        self.assertEqual(composite.mode, "RGB")

    def test_tool_run_cancelled(self) -> None:
        tool = HOGTool()
        with patch("forensics_app.tools.hog.HOGDialog") as mock_dialog:
            mock_dialog.return_value.result = None
            res = tool.run(None, self.document)
            self.assertIsNone(res)

    def test_tool_run_applied(self) -> None:
        tool = HOGTool()
        with patch("forensics_app.tools.hog.HOGDialog") as mock_dialog, \
             patch("forensics_app.tools.hog.HOGComparisonWindow") as mock_window:
            mock_dialog.return_value.result = {
                "orientations": 9,
                "pixels_per_cell": (6, 6),
                "cells_per_block": (2, 2),
            }
            res = tool.run(None, self.document)
            self.assertIsNotNone(res)
            self.assertIsNone(res.image)  # Preserves working document image
            self.assertIn("Descriptor length", res.details)
            self.assertIn("Orientations", res.details)
            self.assertEqual(res.details["Orientations"], 9)
            mock_window.assert_called_once()


if __name__ == "__main__":
    unittest.main()
