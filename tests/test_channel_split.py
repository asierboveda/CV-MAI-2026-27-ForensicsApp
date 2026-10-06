import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class ChannelSplitTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (2, 1), (200, 50, 20))

    def tool(self):
        tools = [tool for tool in build_tool_registry().all()
                 if tool.tool_id == "channel_split"]
        self.assertEqual(len(tools), 1, "Channel split must be registered")
        return tools[0]

    def test_extracts_each_channel_without_changing_document(self):
        tool = self.tool()
        for channel, expected in [("R", 200), (" g ", 50), ("B", 20)]:
            with self.subTest(channel=channel):
                with patch("tkinter.simpledialog.askstring", return_value=channel):
                    result = tool.run(None, self.document)
                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, (2, 1))
                self.assertEqual(result.image.getpixel((0, 0)), expected)
                self.assertEqual(self.document.current.getpixel((0, 0)), (200, 50, 20))

    def test_cancel_returns_no_result(self):
        tool = self.tool()
        with patch("tkinter.simpledialog.askstring", return_value=None):
            self.assertIsNone(tool.run(None, self.document))

    def test_invalid_channel_is_rejected(self):
        tool = self.tool()
        with patch("tkinter.simpledialog.askstring", return_value="X"):
            with self.assertRaisesRegex(ValueError, "R, G, or B"):
                tool.run(None, self.document)

    def test_rgba_input_is_supported(self):
        tool = self.tool()
        self.document.current = Image.new("RGBA", (1, 1), (200, 50, 20, 128))
        with patch("tkinter.simpledialog.askstring", return_value="B"):
            result = tool.run(None, self.document)
        self.assertEqual(result.image.getpixel((0, 0)), 20)

    def test_result_can_be_undone(self):
        tool = self.tool()
        with patch("tkinter.simpledialog.askstring", return_value="R"):
            result = tool.run(None, self.document)
        self.document.apply(result.image)
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.current.getpixel((0, 0)), (200, 50, 20))
