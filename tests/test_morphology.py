import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class MorphologyTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (30, 30), (50, 50, 50))

    def tool(self):
        tools = [t for t in build_tool_registry().all() if t.tool_id == "morphology_elements"]
        self.assertEqual(len(tools), 1, "Morphology tool must be registered")
        return tools[0]

    def test_all_shapes_black_background(self):
        tool = self.tool()
        for shape in ["Disk", "Diamond", "Square", "Star"]:
            with self.subTest(shape=shape):
                with patch("forensics_app.tools.morphology.MorphologyDialog") as mock_dialog:
                    mock_dialog.return_value.result = (shape, 5, "Black background")
                    result = tool.run(None, self.document)
                self.assertIsNotNone(result)
                self.assertEqual(result.image.size, (30, 30))

    def test_current_image_background(self):
        tool = self.tool()
        with patch("forensics_app.tools.morphology.MorphologyDialog") as mock_dialog:
            mock_dialog.return_value.result = ("Disk", 5, "Current image (grayscale)")
            result = tool.run(None, self.document)
        self.assertIsNotNone(result)
        self.assertEqual(result.image.size, (30, 30))

    def test_cancel_returns_none(self):
        tool = self.tool()
        with patch("forensics_app.tools.morphology.MorphologyDialog") as mock_dialog:
            mock_dialog.return_value.result = None
            result = tool.run(None, self.document)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
