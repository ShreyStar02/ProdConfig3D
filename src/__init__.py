"""
src package initialization
"""
from .config import AppConfig, SegmentationMethod
from .step1.segment_classifier import SegmentClassifier, classify_all_segments
from .step1.segmentation import MeshSegment, MeshSegmentationPipeline
from .step1.zeroshot_segmentation import ZeroShotMaxAccuracySegmenter

__version__ = "1.0.0"
__all__ = [
    "AppConfig",
    "SegmentationMethod",
    "SegmentClassifier",
    "classify_all_segments",
    "MeshSegment",
    "MeshSegmentationPipeline",
    "ZeroShotMaxAccuracySegmenter"
]
