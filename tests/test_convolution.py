import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class ConvolutionTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (10, 10), (100, 100, 100))

    def tool(self):
        tools = [t for t in build_tool_registry().all() if t.tool_id == "convolution"]
        self.assertEqual(len(tools), 1, "Convolution must be registered")
        return tools[0]

    def test_2d_convolution(self):
        tool = self.tool()
        with patch("forensics_app.tools.convolution.ConvolutionDialog") as mock_dialog:
            mock_dialog.return_value.result = (3, "2D")
            result = tool.run(None, self.document)
        self.assertIsNotNone(result)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, (10, 10))

    def test_horizontal_and_vertical_convolution(self):
        tool = self.tool()
        for direction in ["Horizontal (1D)", "Vertical (1D)"]:
            with self.subTest(direction=direction):
                with patch("forensics_app.tools.convolution.ConvolutionDialog") as mock_dialog:
                    mock_dialog.return_value.result = (5, direction)
                    result = tool.run(None, self.document)
                self.assertIsNotNone(result)
                self.assertEqual(result.image.size, (10, 10))

    def test_cancel_returns_none(self):
        tool = self.tool()
        with patch("forensics_app.tools.convolution.ConvolutionDialog") as mock_dialog:
            mock_dialog.return_value.result = None
            result = tool.run(None, self.document)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
