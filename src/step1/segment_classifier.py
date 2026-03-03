"""
Segment classification helpers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from .segmentation import MeshSegment

logger = logging.getLogger(__name__)


class SegmentClassifier:
    """Placeholder classifier - keeps existing labels or maps to AI criteria."""

    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold

    def classify(self, segments: List[MeshSegment], ai_criteria: Optional[Dict[str, Any]] = None) -> List[MeshSegment]:
        labels = _criteria_labels(ai_criteria)
        if not labels:
            return segments

        label_idx = 0
        for seg in segments:
            if seg.label.startswith("segment_") or seg.label == "unlabeled":
                target_label = labels[label_idx % len(labels)]
                seg.label = target_label
                seg.mesh_data.name = target_label
                label_idx += 1
        return segments


def classify_all_segments(
    segments: List[MeshSegment],
    product_description: str,
    ai_criteria: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.25,
    nim_config: Optional[Any] = None
) -> List[MeshSegment]:
    """
    Basic classification wrapper used by the pipeline.

    This keeps current labels unless AI criteria provides explicit labels.
    """
    classifier = SegmentClassifier(confidence_threshold=confidence_threshold)
    return classifier.classify(segments, ai_criteria)


def _criteria_labels(ai_criteria: Optional[Dict[str, Any]]) -> List[str]:
    if not ai_criteria:
        return []

    labels: List[str] = []
    for part in ai_criteria.get("parts", []):
        if isinstance(part, dict):
            label = part.get("part_name") or part.get("sam_prompt") or part.get("name")
            if label:
                labels.append(str(label))
        elif isinstance(part, str):
            labels.append(part)
    return labels
