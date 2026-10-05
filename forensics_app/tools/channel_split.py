"""Extract one RGB channel as a grayscale image."""

import tkinter as tk
from tkinter import simpledialog

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel split"
    category = "Set 2"
    description = "Show an RGB channel as grayscale. Reset before comparing another channel."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        channel = simpledialog.askstring(
            "Channel split",
            "Choose R (red), G (green), or B (blue).\n"
            "Use Reset before comparing another channel.",
            parent=parent,
            initialvalue="R",
        )
        if channel is None:
            return None
        channel = channel.strip().upper()
        if channel not in {"R", "G", "B"}:
            raise ValueError("Choose R, G, or B.")

        assert document.current is not None
        # Pillow returns one grayscale image for each RGB component.
        red, green, blue = document.current.convert("RGB").split()
        output = {"R": red, "G": green, "B": blue}[channel]
        return ToolResult(
            image=output,
            message=f"Showing channel {channel}. Use Reset before comparing another channel.",
            details={
                "Operation": "Channel split",
                "Channel": channel,
                "Input mode": document.current.mode,
                "Output mode": output.mode,
            },
        )
