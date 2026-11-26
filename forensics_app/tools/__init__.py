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
from .corner_detection import CornerDetectionTool
from .descriptor_extractor import DescriptorExtractorTool
from .feature_matching import FeatureMatchingTool
from .folder_match import FolderMatchingTool
from .bow_builder import CodebookBuilderTool
from .haar import HaarTool
from .yunet import YuNetTool
from .eigenfaces import EigenfacesTool
from .face_recognition import FaceRecognitionTool
from .bow_visual_words import VisualWordsTool
from .bow_histogram import BoWHistogramTool
from .bow_classify import BoWClassifyTool
from .bow_retrieval import BoWRetrievalTool
from .cnn import CNNClassifyTool, FeatureProjectionTool
from .yolo import YOLODetectionTool
from .clip import CLIPZeroShotTool, CLIPTextSearchTool


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
            CornerDetectionTool(),
            DescriptorExtractorTool(),
            FeatureMatchingTool(),
            FolderMatchingTool(),
            # Set 6
            HaarTool(),
            YuNetTool(),
            EigenfacesTool(),
            FaceRecognitionTool(),
            # Set 7 — Bag of Visual Words
            CodebookBuilderTool(),
            BoWHistogramTool(),
            BoWClassifyTool(),
            BoWRetrievalTool(),
            # Set 8 — CNN classification
            CNNClassifyTool(),
            FeatureProjectionTool(),
            # Set 9 — YOLO detection
            YOLODetectionTool(),
            # Set 10 — CLIP
            CLIPZeroShotTool(),
            CLIPTextSearchTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
