import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class ContrastStretchTests(unittest.TestCase):
    def tool(self):
        tools = [tool for tool in build_tool_registry().all()
                 if tool.tool_id == "contrast_stretch"]
        self.assertEqual(len(tools), 1, "Contrast stretching must be registered")
        return tools[0]

    def run_image(self, image):
        document = ImageDocument()
        document.current = image
        with patch("tkinter.simpledialog.askfloat", return_value=0.0):
            return self.tool().run(None, document), document

    def test_percentile_cutoff_ignores_isolated_extremes(self):
        image = Image.new("L", (100, 1))
        image.putdata([0] + [50] * 49 + [150] * 49 + [255])
        document = ImageDocument()
        document.current = image
        with patch("tkinter.simpledialog.askfloat", return_value=1.0):
            result = self.tool().run(None, document)
        self.assertEqual(result.image.getpixel((1, 0)), 0)
        self.assertEqual(result.image.getpixel((50, 0)), 255)

    def test_cancel_returns_no_result(self):
        document = ImageDocument()
        document.current = Image.new("L", (1, 1), 75)
        with patch("tkinter.simpledialog.askfloat", return_value=None):
            self.assertIsNone(self.tool().run(None, document))

    def test_grayscale_expands_range_and_preserves_input(self):
        image = Image.new("L", (3, 1))
        image.putdata([50, 100, 150])
        result, document = self.run_image(image)
        self.assertEqual(result.image.tobytes(), bytes([0, 128, 255]))
        self.assertEqual(image.tobytes(), bytes([50, 100, 150]))
        document.apply(result.image)
        self.assertTrue(document.undo())
        self.assertEqual(document.current.tobytes(), image.tobytes())

    def test_uniform_image_is_unchanged(self):
        image = Image.new("L", (1, 1), 75)
        result, _ = self.run_image(image)
        self.assertEqual(result.image.getpixel((0, 0)), 75)

    def test_rgb_channels_stretch_independently(self):
        image = Image.new("RGB", (2, 1))
        image.putdata([(10, 50, 80), (20, 50, 180)])
        result, _ = self.run_image(image)
        self.assertEqual(result.image.getpixel((0, 0)), (0, 50, 0))
        self.assertEqual(result.image.getpixel((1, 0)), (255, 50, 255))

    def test_alpha_is_preserved(self):
        image = Image.new("RGBA", (2, 1))
        image.putdata([(10, 20, 30, 40), (50, 60, 70, 80)])
        result, _ = self.run_image(image)
        self.assertEqual(result.image.getchannel("A").tobytes(), bytes([40, 80]))

    def test_grayscale_alpha_is_preserved(self):
        image = Image.new("LA", (2, 1))
        image.putdata([(50, 10), (150, 200)])
        result, _ = self.run_image(image)
        self.assertEqual(result.image.mode, "LA")
        self.assertEqual(result.image.getpixel((0, 0)), (0, 10))
        self.assertEqual(result.image.getpixel((1, 0)), (255, 200))

    def test_full_range_is_unchanged(self):
        image = Image.new("L", (3, 1))
        image.putdata([0, 100, 255])
        result, _ = self.run_image(image)
        self.assertEqual(result.image.tobytes(), image.tobytes())
