"""Linear min/max contrast stretching for 8-bit intensity channels."""

import tkinter as tk
from tkinter import simpledialog

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def channel_limits(channel: Image.Image, cutoff: float) -> tuple[int, int]:
    """Find intensity bounds after ignoring cutoff percent at each tail."""
    counts = channel.histogram()
    discard = int(sum(counts) * cutoff / 100)
    cumulative = 0
    low = 0
    for value, count in enumerate(counts):
        cumulative += count
        if cumulative > discard:
            low = value
            break
    cumulative = 0
    high = 255
    for value in range(255, -1, -1):
        cumulative += counts[value]
        if cumulative > discard:
            high = value
            break
    return low, high


def stretch_channel(channel: Image.Image, cutoff: float = 0) -> Image.Image:
    """Map minimum to 0 and maximum to 255; leave constant channels intact."""
    low, high = channel_limits(channel, cutoff)
    if low >= high:
        return channel.copy()
    lookup = [max(0, min(255, round((value - low) * 255 / (high - low))))
              for value in range(256)]
    return channel.point(lookup)


class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast stretching"
    category = "Set 2"
    description = "Expand intensities to 0–255, optionally ignoring extreme pixels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        cutoff = simpledialog.askfloat(
            "Contrast stretching",
            "Percentage to ignore at EACH end of the histogram:\n"
            "0 = basic min/max; 1 = ignore the lowest and highest 1%.",
            parent=parent, initialvalue=1.0, minvalue=0.0, maxvalue=49.0,
        )
        if cutoff is None:
            return None
        source = document.current
        if source.mode in {"1", "L", "LA", "I", "F"} or source.mode.startswith("I;16"):
            working = source.convert("LA" if source.mode == "LA" else "L")
        else:
            has_alpha = "A" in source.getbands() or "transparency" in source.info
            working = source.convert("RGBA" if has_alpha else "RGB")

        channels = []
        details = {"Operation": "Contrast stretching", "Method": "Per-channel linear stretch",
                   "Cutoff per tail": f"{cutoff:g}%",
                   "Input mode": source.mode, "Output mode": working.mode}
        for name, channel in zip(working.getbands(), working.split()):
            if name == "A":
                channels.append(channel.copy())
                continue
            low, high = channel.getextrema()
            lower, upper = channel_limits(channel, cutoff)
            channels.append(stretch_channel(channel, cutoff))
            details[f"{name} input range"] = f"{low}–{high}"
            details[f"{name} stretch bounds"] = f"{lower}–{upper}"
            details[f"{name} output range"] = "0–255" if lower < upper else "Unchanged (collapsed bounds)"
        output = Image.merge(working.mode, tuple(channels))
        return ToolResult(
            image=output,
            message=f"Applied contrast stretching with {cutoff:g}% cutoff per tail.",
            details=details,
        )
