"""Display 256-bin intensity histograms without changing the working image."""

import tkinter as tk
from tkinter import ttk

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def compute_histograms(image: Image.Image) -> dict[str, list[int]]:
    """Count pixels at each 8-bit intensity; alpha is not a color channel.

    All pixels are counted, including transparent pixels' stored RGB values.
    Non-8-bit grayscale modes are converted to L for this 0–255 visualization.
    """
    if image.mode in {"1", "L", "LA", "I", "F"} or image.mode.startswith("I;16"):
        return {"Intensity": image.convert("L").histogram()}
    red, green, blue = image.convert("RGB").split()
    return {"R": red.histogram(), "G": green.histogram(), "B": blue.histogram()}


class HistogramWindow(tk.Toplevel):
    def __init__(self, parent: tk.Misc, histograms: dict[str, list[int]]) -> None:
        super().__init__(parent)
        self.title("Histogram visualization")
        self.geometry("760x480")
        self.minsize(480, 320)
        self.histograms = histograms
        ttk.Label(self, text="Pixel counts by intensity (0–255)", padding=8).pack()
        self.canvas = tk.Canvas(self, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        names = "  |  ".join(histograms)
        ttk.Label(
            self, text=f"Channels: {names}. All pixels counted; alpha ignored.", padding=8,
        ).pack()
        self.canvas.bind("<Configure>", self._draw)

    def _draw(self, _event=None) -> None:
        canvas = self.canvas
        canvas.delete("all")
        left, top = 75, 30
        right, bottom = max(left + 1, canvas.winfo_width() - 25), max(top + 1, canvas.winfo_height() - 55)
        peak = max(1, max(max(counts) for counts in self.histograms.values()))
        for step in range(5):
            y = bottom - (bottom - top) * step / 4
            canvas.create_line(left, y, right, y, fill="#e5e7eb")
            canvas.create_text(left - 8, y, text=f"{peak * step / 4:,.0f}", anchor="e")
        canvas.create_line(left, top, left, bottom, right, bottom, fill="#333333")
        for value in [0, 64, 128, 192, 255]:
            x = left + (right - left) * value / 255
            canvas.create_line(x, bottom, x, bottom + 5)
            canvas.create_text(x, bottom + 16, text=str(value))
        canvas.create_text((left + right) / 2, bottom + 38, text="Intensity")
        canvas.create_text(18, (top + bottom) / 2, text="Pixels", angle=90)
        colors = {"R": "#dc2626", "G": "#15803d", "B": "#2563eb", "Intensity": "#374151"}
        for index, (name, counts) in enumerate(self.histograms.items()):
            points = []
            for value, count in enumerate(counts):
                points.extend((left + (right - left) * value / 255,
                               bottom - (bottom - top) * count / peak))
            canvas.create_line(*points, fill=colors[name], width=2)
            canvas.create_text(left + index * 100, 12, text=name, fill=colors[name], anchor="w")


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram visualization"
    category = "Set 2"
    description = "Plot RGB or grayscale intensity counts without changing the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        histograms = compute_histograms(document.current)
        HistogramWindow(parent, histograms)
        details = {"Operation": "Histogram", "Bins": 256,
                   "Pixels": document.current.width * document.current.height}
        for name, counts in histograms.items():
            details[f"{name} peak intensity"] = max(range(256), key=counts.__getitem__)
        return ToolResult(
            message="Histogram opened in a separate window.", details=details,
        )
