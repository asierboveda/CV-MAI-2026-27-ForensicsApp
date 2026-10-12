import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry


class SharpenTests(unittest.TestCase):
    def tool(self):
        tools = [tool for tool in build_tool_registry().all() if tool.tool_id == "sharpen"]
        self.assertEqual(len(tools), 1, "Sharpen must be registered")
        return tools[0]

    def run_image(self, image, amount=1.0):
        document = ImageDocument()
        document.original = image.copy()
        document.current = image
        with patch("tkinter.simpledialog.askfloat", return_value=amount):
            result = self.tool().run(None, document)
        return result, document

    def test_zero_amount_preserves_grayscale_and_rgb(self):
        for mode in ["L", "RGB"]:
            with self.subTest(mode=mode):
                image = Image.new("L", (6, 1))
                image.putdata([80, 80, 80, 120, 120, 120])
                image = image.convert(mode)
                result, _ = self.run_image(image, amount=0.0)
                self.assertEqual(result.image.mode, mode)
                self.assertEqual(result.image.tobytes(), image.tobytes())

    def test_positive_amount_increases_boundary_contrast_without_mutation(self):
        image = Image.new("L", (6, 1))
        image.putdata([80, 80, 80, 120, 120, 120])
        original = image.tobytes()
        result, document = self.run_image(image)
        self.assertLess(result.image.getpixel((2, 0)), 80)
        self.assertGreater(result.image.getpixel((3, 0)), 120)
        self.assertEqual(result.image.size, image.size)
        self.assertEqual(document.current.tobytes(), original)
        self.assertEqual(document.original.tobytes(), original)

    def test_rgb_channels_are_processed_independently(self):
        red = Image.new("L", (6, 1))
        red.putdata([80, 80, 80, 120, 120, 120])
        image = Image.merge("RGB", (red, Image.new("L", red.size, 20), Image.new("L", red.size, 200)))
        result, _ = self.run_image(image)
        r, g, b = result.image.getpixel((2, 0))
        self.assertLess(r, 80)
        self.assertEqual((g, b), (20, 200))

    def test_output_is_clipped_before_uint8_conversion(self):
        image = Image.new("L", (6, 1))
        image.putdata([50, 50, 50, 200, 200, 200])
        result, _ = self.run_image(image, amount=10.0)
        self.assertEqual(result.image.getpixel((2, 0)), 0)
        self.assertEqual(result.image.getpixel((3, 0)), 255)

    def test_cancel_leaves_document_unchanged(self):
        image = Image.new("RGB", (8, 8), "red")
        result, document = self.run_image(image, amount=None)
        self.assertIsNone(result)
        self.assertEqual(document.current.tobytes(), image.tobytes())
        self.assertFalse(document.can_undo)

    def test_invalid_amount_is_reprompted(self):
        document = ImageDocument()
        document.current = Image.new("L", (8, 8), 100)
        with patch("tkinter.simpledialog.askfloat", side_effect=[float("nan"), float("inf"), -1.0, 0.0]), \
                patch("tkinter.messagebox.showerror"):
            result = self.tool().run(None, document)
        self.assertEqual(result.image.getextrema(), (100, 100))
        self.assertEqual(result.details["Amount"], 0.0)
