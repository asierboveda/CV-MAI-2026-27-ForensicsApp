"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretching import (
    LowContrastSimTool,
    ContrastStretchingPercentileTool,
    HistogramEqualizationTool,
    AdaptiveEqualizationTool,
)
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .histogram_matching import HistogramMatchingTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            HistogramTool(),
            HistogramMatchingTool(),
            LowContrastSimTool(),
            ContrastStretchingPercentileTool(),
            HistogramEqualizationTool(),
            AdaptiveEqualizationTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]

