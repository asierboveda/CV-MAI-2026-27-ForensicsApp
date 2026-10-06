"""Recombine RGB channels in a user-selected order."""

import tkinter as tk
from tkinter import simpledialog

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Channel swap"
    category = "Set 2"
    description = "Reorder RGB channels; BGR swaps red and blue."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        order = simpledialog.askstring(
            "Channel swap",
            "Enter the source channels for the output R, G, B positions.\n"
            "Options: RGB, RBG, GRB, GBR, BRG, BGR.\n"
            "Example: BGR swaps red and blue.",
            parent=parent,
            initialvalue="BGR",
        )
        if order is None:
            return None
        order = order.strip().upper()
        if len(order) != 3 or set(order) != {"R", "G", "B"}:
            raise ValueError("Enter R, G, and B exactly once, for example BGR.")

        assert document.current is not None
        source = document.current
        red, green, blue = source.convert("RGB").split()
        channels = {"R": red, "G": green, "B": blue}
        # Output is still an RGB image; the input channels change positions.
        output = Image.merge("RGB", tuple(channels[name] for name in order))
        if "A" in source.getbands() or "transparency" in source.info:
            output.putalpha(source.convert("RGBA").getchannel("A"))

        return ToolResult(
            image=output,
            message=f"Reordered channels: RGB → {order}.",
            details={
                "Operation": "Channel swap",
                "Channel order": order,
                "Output R from": order[0],
                "Output G from": order[1],
                "Output B from": order[2],
                "Output mode": output.mode,
            },
        )
