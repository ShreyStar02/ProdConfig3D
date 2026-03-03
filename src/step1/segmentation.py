"""
Segmentation utilities and geometric fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque
import logging

import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

from ..config import SegmentationConfig
from .mesh_processor import MeshData, merge_meshes

logger = logging.getLogger(__name__)


@dataclass
class MeshSegment:
    """Container for a segmented mesh part."""
    segment_id: int
    label: str
    mesh_data: MeshData
    confidence: float = 0.5
    properties: Dict[str, Any] = field(default_factory=dict)


class MeshSegmentationPipeline:
    """Basic geometric segmentation with conservative defaults."""

    def __init__(self, config: SegmentationConfig):
        self.config = config

    def process(self, mesh_data: MeshData, custom_parts: Optional[List[str]] = None) -> List[MeshSegment]:
        if not TRIMESH_AVAILABLE:
            logger.warning("trimesh not available - returning single segment")
            return [self._make_segment(mesh_data, 0, custom_parts, 0)]

        mesh = mesh_data.to_trimesh()
        components = mesh.split(only_watertight=False)
        if not components:
            return [self._make_segment(mesh_data, 0, custom_parts, 0)]

        segments: List[MeshSegment] = []
        for idx, comp in enumerate(components):
            sub_mesh = MeshData.from_trimesh(comp, name=f"segment_{idx}")
            seg = self._make_segment(sub_mesh, idx, custom_parts, idx)
            segments.append(seg)

        return segments

    def _make_segment(
        self,
        mesh_data: MeshData,
        segment_id: int,
        custom_parts: Optional[List[str]],
        index: int
    ) -> MeshSegment:
        label = f"segment_{segment_id}"
        if custom_parts and index < len(custom_parts):
            label = custom_parts[index]

        properties = {
            "face_count": mesh_data.num_faces,
        }
        if TRIMESH_AVAILABLE:
            try:
                tri = mesh_data.to_trimesh()
                properties["area"] = float(tri.area)
                properties["bounds"] = tri.bounds.tolist()
                properties["center"] = tri.bounds.mean(axis=0).tolist()
            except Exception:
                pass

        return MeshSegment(
            segment_id=segment_id,
            label=label,
            mesh_data=mesh_data,
            confidence=0.5,
            properties=properties
        )


def merge_segments_by_label(segments: List[MeshSegment]) -> List[MeshSegment]:
    """Merge segments sharing the same label into single MeshSegments."""
    if not segments:
        return []

    grouped: Dict[str, List[MeshSegment]] = {}
    for seg in segments:
        grouped.setdefault(seg.label, []).append(seg)

    merged: List[MeshSegment] = []
    for idx, (label, segs) in enumerate(grouped.items()):
        if len(segs) == 1:
            merged.append(segs[0])
            continue

        meshes = [s.mesh_data for s in segs]
        merged_mesh = merge_meshes(meshes)
        merged_seg = MeshSegment(
            segment_id=idx,
            label=label,
            mesh_data=merged_mesh,
            confidence=max(s.confidence for s in segs),
            properties={
                "face_count": merged_mesh.num_faces,
                "merged_from": [s.segment_id for s in segs]
            }
        )
        merged.append(merged_seg)

    return merged


class AIGuidedSegmenter:
    """Simple AI-guided segmentation using geometric hints."""

    def __init__(self, config: SegmentationConfig):
        self.config = config

    def segment_with_criteria(self, mesh_data: MeshData, criteria: Dict[str, Any]) -> List[MeshSegment]:
        parts = criteria.get("parts", []) if criteria else []
        if not parts:
            return [MeshSegment(0, "main_body", mesh_data)]

        features = self._compute_face_features(mesh_data)
        if not features:
            return [MeshSegment(0, "main_body", mesh_data)]

        norm_coords = features["norm_coords"]
        normals = features["normals"]
        radial_dirs = features["radial_dirs"]
        area_norm = features["area_norm"]
        height_axis = features["height_axis"]
        length_axis = features["length_axis"]
        width_axis = features["width_axis"]
        length_sign = self._infer_length_sign(parts, features)

        part_names: List[str] = []
        score_stack: List[np.ndarray] = []

        for part in parts:
            if isinstance(part, dict):
                part_name = part.get("part_name") or part.get("name") or "part"
                part_names.append(part_name)

                scores = np.ones(mesh_data.num_faces, dtype=np.float32)

                height_range = part.get("height_range")
                if height_range and isinstance(height_range, (list, tuple)) and len(height_range) == 2:
                    lo, hi = float(height_range[0]), float(height_range[1])
                    scores *= self._range_score(norm_coords[:, height_axis], lo, hi)

                position = str(part.get("position", "")).strip().lower()
                if position and position not in ("any", "all_around"):
                    position_score = self._position_score(
                        position,
                        norm_coords[:, length_axis],
                        norm_coords[:, width_axis],
                        norm_coords[:, height_axis],
                        length_sign
                    )
                    if position_score is not None:
                        scores *= position_score

                normal_direction = str(part.get("normal_direction", "")).strip().lower()
                if normal_direction and normal_direction != "any":
                    normal_score = self._normal_score(
                        normal_direction,
                        normals,
                        radial_dirs,
                        height_axis,
                        length_axis,
                        width_axis,
                        length_sign
                    )
                    if normal_score is not None:
                        scores *= normal_score

                special_features = part.get("special_features") or []
                if isinstance(special_features, str):
                    special_features = [special_features]
                special_features = [str(f).strip().lower() for f in special_features if f]

                if "thin_elements" in special_features:
                    scores *= self._thin_score(area_norm)

                relative_size = str(part.get("relative_size", "")).strip().lower()
                size_cap = self._relative_size_cap(relative_size)
                if size_cap is not None:
                    scores = self._apply_size_cap(scores, size_cap)

                score_stack.append(scores)
            elif isinstance(part, str):
                part_names.append(part)
                score_stack.append(np.ones(mesh_data.num_faces, dtype=np.float32))

        if not part_names:
            return [MeshSegment(0, "main_body", mesh_data)]

        scores = np.vstack(score_stack)
        best_idx = np.argmax(scores, axis=0)
        labels = np.array([part_names[i] for i in best_idx], dtype=object)

        if self.config.enable_boundary_refinement:
            labels = _smooth_labels_by_connectivity(
                mesh_data,
                labels,
                self.config.min_segment_faces,
                self.config.boundary_smoothing_iterations
            )

        return _segments_from_labels(mesh_data, labels)

    def _face_centers(self, mesh_data: MeshData) -> np.ndarray:
        faces = mesh_data.faces
        verts = mesh_data.vertices
        return verts[faces].mean(axis=1)

    def _compute_face_features(self, mesh_data: MeshData) -> Optional[Dict[str, Any]]:
        if mesh_data.num_faces == 0:
            return None

        faces = mesh_data.faces
        verts = mesh_data.vertices
        centers = verts[faces].mean(axis=1)

        bounds = np.vstack([verts.min(axis=0), verts.max(axis=0)])
        extents = bounds[1] - bounds[0]
        safe_extents = np.where(extents > 1e-6, extents, 1e-6)
        norm_coords = (centers - bounds[0]) / safe_extents

        normals = mesh_data.face_normals
        if normals is None or len(normals) != mesh_data.num_faces:
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            normals = np.cross(v1 - v0, v2 - v0)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.clip(norms, 1e-8, None)
        else:
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]

        tri_normals = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(tri_normals, axis=1)
        normal_alignment = np.sum(np.abs(normals) * areas[:, None], axis=0)
        height_axis = int(np.argmax(normal_alignment))
        remaining_axes = [axis for axis in range(3) if axis != height_axis]
        if extents[remaining_axes[0]] >= extents[remaining_axes[1]]:
            length_axis, width_axis = remaining_axes[0], remaining_axes[1]
        else:
            length_axis, width_axis = remaining_axes[1], remaining_axes[0]

        center = bounds.mean(axis=0)
        radial = centers - center
        radial_norms = np.linalg.norm(radial, axis=1, keepdims=True)
        radial_dirs = radial / np.clip(radial_norms, 1e-8, None)

        return {
            "centers": centers,
            "normals": normals,
            "norm_coords": norm_coords,
            "radial_dirs": radial_dirs,
            "face_areas": areas,
            "area_norm": self._normalize_feature(areas),
            "height_axis": height_axis,
            "length_axis": length_axis,
            "width_axis": width_axis
        }

    def _range_score(self, coords: np.ndarray, lo: float, hi: float, softness: float = 0.12) -> np.ndarray:
        if hi < lo:
            lo, hi = hi, lo
        dist = np.zeros_like(coords)
        dist = np.where(coords < lo, lo - coords, dist)
        dist = np.where(coords > hi, coords - hi, dist)
        return np.exp(-np.square(dist / max(softness, 1e-3)))

    def _infer_length_sign(self, parts: List[Any], features: Dict[str, Any]) -> float:
        norm_coords = features["norm_coords"]
        normals = features["normals"]
        radial_dirs = features["radial_dirs"]
        height_axis = features["height_axis"]
        length_axis = features["length_axis"]
        width_axis = features["width_axis"]

        def score_for_sign(sign: float) -> float:
            total = 0.0
            weight = 0
            for part in parts:
                if not isinstance(part, dict):
                    continue

                height_range = part.get("height_range")
                if height_range and isinstance(height_range, (list, tuple)) and len(height_range) == 2:
                    lo, hi = float(height_range[0]), float(height_range[1])
                    mask = (norm_coords[:, height_axis] >= lo) & (norm_coords[:, height_axis] <= hi)
                else:
                    mask = np.ones(norm_coords.shape[0], dtype=bool)

                if not np.any(mask):
                    continue

                position = str(part.get("position", "")).strip().lower()
                if position and position not in ("any", "all_around"):
                    pos_score = self._position_score(
                        position,
                        norm_coords[:, length_axis],
                        norm_coords[:, width_axis],
                        norm_coords[:, height_axis],
                        sign
                    )
                    if pos_score is not None:
                        total += float(np.mean(pos_score[mask]))
                        weight += 1

                normal_direction = str(part.get("normal_direction", "")).strip().lower()
                if normal_direction and normal_direction != "any":
                    norm_score = self._normal_score(
                        normal_direction,
                        normals,
                        radial_dirs,
                        height_axis,
                        length_axis,
                        width_axis,
                        sign
                    )
                    if norm_score is not None:
                        total += float(np.mean(norm_score[mask]))
                        weight += 1

            return total / max(weight, 1)

        score_pos = score_for_sign(1.0)
        score_neg = score_for_sign(-1.0)
        return 1.0 if score_pos >= score_neg else -1.0

    def _normalize_feature(self, values: np.ndarray) -> np.ndarray:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmax - vmin < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return (values - vmin) / (vmax - vmin)

    def _thin_score(self, area_norm: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - area_norm, 0.1, 1.0)

    def _relative_size_cap(self, relative_size: str) -> Optional[float]:
        size_map = {
            "tiny": 0.03,
            "small": 0.08,
            "medium": 0.2,
            "large": 0.35,
            "dominant": 0.6
        }
        return size_map.get(relative_size)

    def _apply_size_cap(self, scores: np.ndarray, target_frac: float, floor: float = 0.02) -> np.ndarray:
        if target_frac <= 0.0 or target_frac >= 1.0:
            return scores
        threshold = float(np.quantile(scores, 1.0 - target_frac))
        capped = np.where(scores >= threshold, scores, scores * floor)
        return capped

    def _position_score(
        self,
        position: str,
        length_coords: np.ndarray,
        width_coords: np.ndarray,
        height_coords: np.ndarray,
        length_sign: float
    ) -> Optional[np.ndarray]:
        tokens = [t for t in position.replace("_", " ").split(" ") if t]
        if not tokens:
            return None

        if length_sign < 0:
            length_coords = 1.0 - length_coords

        score = np.ones_like(length_coords, dtype=np.float32)
        for token in tokens:
            if token == "front":
                score *= self._range_score(length_coords, 0.66, 1.0)
            elif token == "back":
                score *= self._range_score(length_coords, 0.0, 0.34)
            elif token == "left":
                score *= self._range_score(width_coords, 0.0, 0.34)
            elif token == "right":
                score *= self._range_score(width_coords, 0.66, 1.0)
            elif token == "top":
                score *= self._range_score(height_coords, 0.66, 1.0)
            elif token == "bottom":
                score *= self._range_score(height_coords, 0.0, 0.34)
            elif token == "center":
                score *= self._range_score(length_coords, 0.33, 0.67)
                score *= self._range_score(width_coords, 0.33, 0.67)
        return score

    def _normal_score(
        self,
        normal_direction: str,
        normals: np.ndarray,
        radial_dirs: np.ndarray,
        height_axis: int,
        length_axis: int,
        width_axis: int,
        length_sign: float
    ) -> Optional[np.ndarray]:
        if normal_direction == "up":
            return np.clip(normals[:, height_axis], 0.0, 1.0)
        if normal_direction == "down":
            return np.clip(-normals[:, height_axis], 0.0, 1.0)
        if normal_direction == "outward":
            return np.clip(np.sum(normals * radial_dirs, axis=1), 0.0, 1.0)
        if normal_direction == "inward":
            return np.clip(-np.sum(normals * radial_dirs, axis=1), 0.0, 1.0)
        if normal_direction == "forward":
            return np.clip(normals[:, length_axis] * length_sign, 0.0, 1.0)
        if normal_direction == "backward":
            return np.clip(-normals[:, length_axis] * length_sign, 0.0, 1.0)
        if normal_direction == "right":
            return np.clip(normals[:, width_axis], 0.0, 1.0)
        if normal_direction == "left":
            return np.clip(-normals[:, width_axis], 0.0, 1.0)
        return None


def compute_geometric_cluster_labels(mesh_data: MeshData, target_segments: Optional[int]) -> np.ndarray:
    """Compute geometric cluster labels using KMeans on face features."""
    if not TRIMESH_AVAILABLE or not target_segments or target_segments < 2:
        return np.zeros(mesh_data.num_faces, dtype=np.int64)

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except Exception:
        logger.warning("scikit-learn not available - returning single cluster")
        return np.zeros(mesh_data.num_faces, dtype=np.int64)

    mesh = mesh_data.to_trimesh()
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals

    try:
        curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh, mesh.vertices, radius=mesh.scale / 50
        )
        face_curvature = np.mean(curvature[mesh.faces], axis=1, keepdims=True)
    except Exception:
        face_curvature = np.zeros((mesh_data.num_faces, 1))

    features = np.hstack([face_centers, face_normals, face_curvature])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=target_segments, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(features)
    return labels.astype(np.int64)


def _segments_from_labels(mesh_data: MeshData, labels: np.ndarray) -> List[MeshSegment]:
    segments: List[MeshSegment] = []
    unique_labels = list(dict.fromkeys(labels.tolist()))

    if not TRIMESH_AVAILABLE:
        return [MeshSegment(0, unique_labels[0], mesh_data)]

    base_mesh = mesh_data.to_trimesh()
    for idx, label in enumerate(unique_labels):
        face_indices = np.where(labels == label)[0]
        if len(face_indices) == 0:
            continue

        submesh = base_mesh.submesh([face_indices], append=True, repair=False)
        sub_data = MeshData.from_trimesh(submesh, name=label)
        segments.append(MeshSegment(
            segment_id=idx,
            label=label,
            mesh_data=sub_data,
            confidence=0.6,
            properties={"face_count": sub_data.num_faces}
        ))

    return segments


def _smooth_labels_by_connectivity(
    mesh_data: MeshData,
    labels: np.ndarray,
    min_component_faces: int,
    iterations: int
) -> np.ndarray:
    if min_component_faces <= 1:
        return labels

    mesh = mesh_data.to_trimesh()
    adjacency = mesh.face_adjacency
    if adjacency is None or len(adjacency) == 0:
        return labels

    neighbors: List[List[int]] = [[] for _ in range(mesh_data.num_faces)]
    for a, b in adjacency:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))

    smoothed = labels.copy()

    for _ in range(max(1, iterations)):
        visited = np.zeros(mesh_data.num_faces, dtype=bool)
        for face_idx in range(mesh_data.num_faces):
            if visited[face_idx]:
                continue

            label = smoothed[face_idx]
            queue = deque([face_idx])
            component = []
            visited[face_idx] = True

            while queue:
                current = queue.popleft()
                component.append(current)
                for nbr in neighbors[current]:
                    if visited[nbr]:
                        continue
                    if smoothed[nbr] != label:
                        continue
                    visited[nbr] = True
                    queue.append(nbr)

            neighbor_counts: Dict[str, int] = {}
            boundary_edges = 0
            total_edges = 0
            for face_id in component:
                for nbr in neighbors[face_id]:
                    total_edges += 1
                    nbr_label = smoothed[nbr]
                    if nbr_label == label:
                        continue
                    boundary_edges += 1
                    neighbor_counts[nbr_label] = neighbor_counts.get(nbr_label, 0) + 1

            if not neighbor_counts:
                continue

            boundary_ratio = boundary_edges / max(total_edges, 1)
            needs_relabel = len(component) < min_component_faces or boundary_ratio >= 0.6
            if not needs_relabel:
                continue

            new_label = max(neighbor_counts, key=neighbor_counts.get)
            for face_id in component:
                smoothed[face_id] = new_label

    return smoothed
