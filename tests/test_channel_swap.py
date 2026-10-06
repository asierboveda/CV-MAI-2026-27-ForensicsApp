import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class ChannelSwapTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (2, 1), (200, 50, 20))

    def tool(self):
        tools = [tool for tool in build_tool_registry().all()
                 if tool.tool_id == "channel_swap"]
        self.assertEqual(len(tools), 1, "Channel swap must be registered")
        return tools[0]

    def test_reorders_channels_without_mutating_input(self):
        tool = self.tool()
        cases = {
            "RGB": (200, 50, 20), "RBG": (200, 20, 50),
            "GRB": (50, 200, 20), "GBR": (50, 20, 200),
            "BRG": (20, 200, 50), " bgr ": (20, 50, 200),
        }
        for order, expected in cases.items():
            with self.subTest(order=order):
                with patch("tkinter.simpledialog.askstring", return_value=order):
                    result = tool.run(None, self.document)
                self.assertEqual(result.image.mode, "RGB")
                self.assertEqual(result.image.size, (2, 1))
                self.assertEqual(result.image.getpixel((0, 0)), expected)
                self.assertEqual(self.document.current.getpixel((0, 0)), (200, 50, 20))

    def test_preserves_alpha(self):
        tool = self.tool()
        self.document.current = Image.new("RGBA", (1, 1), (200, 50, 20, 128))
        with patch("tkinter.simpledialog.askstring", return_value="BGR"):
            result = tool.run(None, self.document)
        self.assertEqual(result.image.mode, "RGBA")
        self.assertEqual(result.image.getpixel((0, 0)), (20, 50, 200, 128))

    def test_cancel(self):
        tool = self.tool()
        with patch("tkinter.simpledialog.askstring", return_value=None):
            self.assertIsNone(tool.run(None, self.document))

    def test_rejects_invalid_orders(self):
        tool = self.tool()
        for order in ["", "RRB", "RGBB", "XYZ"]:
            with self.subTest(order=order):
                with patch("tkinter.simpledialog.askstring", return_value=order):
                    with self.assertRaises(ValueError):
                        tool.run(None, self.document)

    def test_grayscale_input_is_supported(self):
        tool = self.tool()
        self.document.current = Image.new("L", (1, 1), 75)
        with patch("tkinter.simpledialog.askstring", return_value="BGR"):
            result = tool.run(None, self.document)
        self.assertEqual(result.image.getpixel((0, 0)), (75, 75, 75))

    def test_can_undo_swap(self):
        tool = self.tool()
        with patch("tkinter.simpledialog.askstring", return_value="BGR"):
            result = tool.run(None, self.document)
        self.document.apply(result.image)
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.current.getpixel((0, 0)), (200, 50, 20))
