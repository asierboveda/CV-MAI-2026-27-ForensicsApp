"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .blur import BlurTool
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .masking import MaskingTool
from .histogram import HistogramTool
from .contrast_stretch import ContrastStretchTool
from .convolution import ConvolutionTool
from .skimage_filters import SkimageFiltersTool
from .morphology import MorphologyTool
from .canny import CannyTool
from .edge_overlay import EdgeOverlayTool
from .sharpen import SharpenTool
from .normalized_cross_corr import Norm_cross_corr      
from .hog import HOGTool
from .person_detection import PersonDetectionTool

def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            BlurTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastStretchTool(),
            ConvolutionTool(),
            SkimageFiltersTool(),
            MorphologyTool(),
            CannyTool(),
            EdgeOverlayTool(),
            SharpenTool(),
            Norm_cross_corr(),
            HOGTool(),
            PersonDetectionTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
