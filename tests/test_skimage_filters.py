import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class SkimageFiltersTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (10, 10), (120, 120, 120))

    def tool(self):
        tools = [t for t in build_tool_registry().all() if t.tool_id == "skimage_filters"]
        self.assertEqual(len(tools), 1, "Skimage filters must be registered")
        return tools[0]

    def test_filters(self):
        tool = self.tool()
        filter_cases = [
            ("Gaussian", 2.0, None),
            ("Sobel", 0.0, "Both (Magnitude)"),
            ("Sobel", 0.0, "Horizontal"),
            ("Sobel", 0.0, "Vertical"),
            ("Prewitt", 0.0, "Both (Magnitude)"),
            ("Scharr", 0.0, "Both (Magnitude)"),
            ("Otsu Threshold", 0.0, None),
        ]
        for name, param, axis in filter_cases:
            with self.subTest(filter=name, axis=axis):
                with patch("forensics_app.tools.skimage_filters.SkimageFilterDialog") as mock_dialog:
                    mock_dialog.return_value.result = (name, param, axis)
                    result = tool.run(None, self.document)
                self.assertIsNotNone(result)
                self.assertEqual(result.image.size, (10, 10))

    def test_cancel_returns_none(self):
        tool = self.tool()
        with patch("forensics_app.tools.skimage_filters.SkimageFilterDialog") as mock_dialog:
            mock_dialog.return_value.result = None
            result = tool.run(None, self.document)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
