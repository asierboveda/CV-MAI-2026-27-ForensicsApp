import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class EdgeOverlayTests(unittest.TestCase):
    def tool(self):
        tools = [tool for tool in build_tool_registry().all() if tool.tool_id == "edge_overlay"]
        self.assertEqual(len(tools), 1, "Edge overlay must be registered")
        return tools[0]

    def setUp(self):
        self.document = ImageDocument()
        self.document.original = Image.new("RGB", (32, 32))
        self.document.original.putpixel((16, 16), (255, 255, 255))
        self.document.current = Image.new("L", (4, 4))

    def test_overlays_only_edges_on_original_not_current(self):
        original = self.document.original.tobytes()
        current = self.document.current.tobytes()
        expected = self.document.original.copy()
        for point in [(15, 15), (16, 15), (17, 15), (15, 16),
                      (17, 16), (15, 17), (16, 17), (17, 17)]:
            expected.putpixel(point, (255, 0, 0))
        with patch("tkinter.simpledialog.askfloat", return_value=0.5):
            result = self.tool().run(None, self.document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, (32, 32))
        self.assertEqual(result.image.tobytes(), expected.tobytes())
        self.assertEqual(self.document.original.tobytes(), original)
        self.assertEqual(self.document.current.tobytes(), current)

    def test_uniform_grayscale_original_is_preserved_in_rgb(self):
        self.document.original = Image.new("L", (8, 8), 100)
        with patch("tkinter.simpledialog.askfloat", return_value=0.0):
            result = self.tool().run(None, self.document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.getextrema(), ((100, 100),) * 3)

    def test_cancel_leaves_document_unchanged(self):
        original = self.document.original.tobytes()
        current = self.document.current.tobytes()
        with patch("tkinter.simpledialog.askfloat", return_value=None):
            result = self.tool().run(None, self.document)
        self.assertIsNone(result)
        self.assertEqual(self.document.original.tobytes(), original)
        self.assertEqual(self.document.current.tobytes(), current)
        self.assertFalse(self.document.can_undo)

    def test_invalid_sigma_is_reprompted(self):
        with patch("tkinter.simpledialog.askfloat", side_effect=[float("nan"), -1.0, float("inf"), 0.5]), \
                patch("tkinter.messagebox.showerror"):
            result = self.tool().run(None, self.document)
        self.assertEqual(result.image.getpixel((16, 15)), (255, 0, 0))
        self.assertEqual(result.details["Sigma"], 0.5)
