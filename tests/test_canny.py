import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class CannyTests(unittest.TestCase):
    def tool(self):
        tools = [tool for tool in build_tool_registry().all() if tool.tool_id == "canny"]
        self.assertEqual(len(tools), 1, "Canny must be registered")
        return tools[0]

    def run_image(self, image, sigma=1.0):
        document = ImageDocument()
        document.original = image.copy()
        document.current = image
        with patch("tkinter.simpledialog.askfloat", return_value=sigma):
            result = self.tool().run(None, document)
        return result, document

    def test_detects_edges_and_sigma_controls_smoothing(self):
        image = Image.new("L", (32, 32))
        image.putpixel((16, 16), 255)
        original = image.tobytes()
        result, document = self.run_image(image, sigma=0.5)
        edges = np.array(result.image)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, image.size)
        self.assertEqual(set(np.unique(edges)), {0, 255})
        self.assertEqual(np.count_nonzero(edges), 8)
        self.assertEqual(edges[16, 16], 0)
        self.assertEqual(document.current.tobytes(), original)
        self.assertEqual(document.original.tobytes(), original)
        smoothed, _ = self.run_image(image, sigma=3.0)
        self.assertEqual(smoothed.image.getextrema(), (0, 0))

    def test_constant_and_tiny_images_have_no_edges(self):
        for mode, size, color in [("RGB", (8, 8), "white"), ("L", (1, 1), 255)]:
            with self.subTest(mode=mode, size=size):
                result, _ = self.run_image(Image.new(mode, size, color), sigma=0.0)
                self.assertEqual(result.image.getextrema(), (0, 0))

    def test_cancel_leaves_document_unchanged(self):
        image = Image.new("RGB", (8, 8), "red")
        result, document = self.run_image(image, sigma=None)
        self.assertIsNone(result)
        self.assertEqual(document.current.tobytes(), image.tobytes())
        self.assertFalse(document.can_undo)

    def test_invalid_sigma_is_reprompted(self):
        document = ImageDocument()
        document.current = Image.new("L", (8, 8))
        with patch("tkinter.simpledialog.askfloat", side_effect=[float("nan"), float("inf"), -1.0, 1.0]), \
                patch("tkinter.messagebox.showerror"):
            result = self.tool().run(None, document)
        self.assertEqual(result.image.getextrema(), (0, 0))
        self.assertEqual(result.details["Sigma"], 1.0)
