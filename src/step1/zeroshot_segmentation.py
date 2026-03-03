"""
Zero-shot segmentation using SAM3 text prompts on rendered views.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import tempfile
from collections import deque

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False

from PIL import Image

from .mesh_processor import MeshData
from .segmentation import MeshSegment, compute_geometric_cluster_labels

logger = logging.getLogger(__name__)

# Compatibility flags for older demo script expectations
SAM1_AVAILABLE = False
SAM2_AVAILABLE = False
XATLAS_AVAILABLE = False

@dataclass
class ZeroShotConfig:
    image_size: int = 640
    conf: float = 0.25
    retry_conf: float = 0.1
    half: bool = True
    retry_enabled: bool = True
    model_path: str = str(Path("./models/sam3.pt"))
    views: Optional[List[Tuple[float, float]]] = None
    min_segment_faces: int = 50
    min_small_segment_faces: int = 12
    min_face_confidence: float = 0.08
    min_visible_views: int = 3
    min_component_faces: int = 60
    cluster_dominance: float = 0.6
    cluster_min_faces: int = 80
    target_num_parts: Optional[int] = None
    auto_mask_enabled: bool = True
    auto_mask_min_area_ratio: float = 0.002
    auto_mask_max_per_view: int = 12
    auto_mask_weight: float = 0.35
    clip_model: str = "ViT-B/32"
    clip_threshold: float = 0.25
    geometry_weight: float = 0.75
    geometry_height_weight: float = 0.45
    geometry_position_weight: float = 0.35
    geometry_normal_weight: float = 0.20
    geometry_softness: float = 0.08

    def resolved_views(self) -> List[Tuple[float, float]]:
        if self.views:
            return self.views
        return [
            (0.0, 25.0),
            (30.0, 25.0),
            (60.0, 25.0),
            (90.0, 25.0),
            (120.0, 25.0),
            (150.0, 25.0),
            (180.0, 25.0),
            (210.0, 25.0),
            (240.0, 25.0),
            (270.0, 25.0),
            (300.0, 25.0),
            (330.0, 25.0),
            (0.0, -25.0),
            (60.0, -25.0),
            (120.0, -25.0),
            (180.0, -25.0),
            (240.0, -25.0),
            (300.0, -25.0),
            (0.0, 55.0),
            (180.0, -55.0)
        ]


class ZeroShotMaxAccuracySegmenter:
    """Segmenter that uses SAM3 PCS on rendered views and projects masks to faces."""

    def __init__(self, config: Optional[ZeroShotConfig] = None):
        self.config = config or ZeroShotConfig()

    def segment(self, mesh_data: MeshData, ai_criteria: Optional[Dict[str, Any]] = None) -> List[MeshSegment]:
        if not SAM3_AVAILABLE:
            raise ImportError("SAM3SemanticPredictor not available. Install ultralytics >= 8.3.237.")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SAM3.")
        if not TRIMESH_AVAILABLE or not PYRENDER_AVAILABLE:
            raise ImportError("trimesh and pyrender are required for SAM3 mesh rendering.")

        prompt_map = _build_prompt_map(ai_criteria)
        prompts = list(prompt_map.keys())
        if not prompts:
            prompts = ["main body"]

        auto_prompts = _select_auto_prompts(ai_criteria, prompt_map, prompts)

        target_parts = _extract_part_names(ai_criteria)
        target_num_parts = self.config.target_num_parts or len(target_parts) or None
        geom_clusters = compute_geometric_cluster_labels(mesh_data, target_num_parts)

        label_to_prompt = _label_to_primary_prompt(prompt_map, prompts)
        clip_assets = _prepare_clip_assets(
            self.config,
            prompt_map,
            ai_criteria
        )

        views = self._render_views(mesh_data)
        face_scores = {prompt: np.zeros(mesh_data.num_faces, dtype=np.float64) for prompt in prompts}
        face_coverage = np.zeros(mesh_data.num_faces, dtype=np.int64)
        prompt_has_mask = {prompt: False for prompt in prompts}
        temp_paths: List[str] = []

        for view in views:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                Image.fromarray(view["rgb"]).save(tmp.name)
                view["image_path"] = tmp.name
                temp_paths.append(tmp.name)

        overrides = dict(
            conf=self.config.conf,
            task="segment",
            mode="predict",
            model=self.config.model_path,
            half=self.config.half
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)

        try:
            for view in views:
                face_ids = view["face_ids"]
                visible = view["visible"]

                _accumulate_coverage(face_coverage, face_ids, visible)
                predictor.set_image(view["image_path"])

                if clip_assets is not None and self.config.auto_mask_weight > 0:
                    auto_masks = _predict_auto_masks(predictor, auto_prompts)
                    if auto_masks:
                        auto_masks = _select_masks_by_area(
                            np.stack(auto_masks, axis=0),
                            self.config.auto_mask_min_area_ratio,
                            self.config.auto_mask_max_per_view
                        )
                        for mask in auto_masks:
                            label = _clip_assign_mask(
                                view["rgb"],
                                mask,
                                clip_assets
                            )
                            if label is None:
                                continue
                            prompt_key = label_to_prompt.get(label)
                            if not prompt_key:
                                continue
                            _accumulate_face_scores(
                                face_scores[prompt_key],
                                face_ids,
                                mask,
                                visible,
                                weight=self.config.auto_mask_weight
                            )

                for prompt in prompts:
                    results = predictor(text=[prompt])
                    masks = _extract_masks(results)
                    if masks is None or masks.size == 0:
                        continue

                    prompt_has_mask[prompt] = True
                    for mask in masks:
                        _accumulate_face_scores(face_scores[prompt], face_ids, mask, visible)

            if self.config.retry_enabled:
                retry_prompts = [p for p, found in prompt_has_mask.items() if not found]
                if retry_prompts:
                    retry_overrides = dict(
                        conf=self.config.retry_conf,
                        task="segment",
                        mode="predict",
                        model=self.config.model_path,
                        half=self.config.half
                    )
                    retry_predictor = SAM3SemanticPredictor(overrides=retry_overrides)

                    for view in views:
                        face_ids = view["face_ids"]
                        visible = view["visible"]
                        retry_predictor.set_image(view["image_path"])

                        for prompt in retry_prompts:
                            results = retry_predictor(text=[prompt])
                            masks = _extract_masks(results)
                            if masks is None or masks.size == 0:
                                continue

                            prompt_has_mask[prompt] = True
                            for mask in masks:
                                _accumulate_face_scores(face_scores[prompt], face_ids, mask, visible)
        finally:
            for path in temp_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass

        prompt_thresholds = _build_prompt_thresholds(
            ai_criteria,
            prompt_map,
            self.config.min_face_confidence,
            self.config.min_visible_views
        )
        _apply_geometry_gating(
            face_scores,
            mesh_data,
            ai_criteria,
            prompt_map,
            self.config
        )
        labels = _assign_labels(
            mesh_data.num_faces,
            prompts,
            face_scores,
            face_coverage,
            self.config.min_face_confidence,
            self.config.min_visible_views,
            prompt_thresholds
        )
        labels = _map_labels(labels, prompt_map)
        labels = _merge_with_geometry(
            labels,
            geom_clusters,
            target_parts,
            self.config.cluster_dominance,
            self.config.cluster_min_faces
        )
        labels = _smooth_labels_by_connectivity(mesh_data, labels, self.config.min_component_faces)
        segments = _segments_from_labels(
            mesh_data,
            labels,
            self.config.min_segment_faces,
            self.config.min_small_segment_faces,
            target_parts
        )
        return segments

    def _render_views(self, mesh_data: MeshData) -> List[Dict[str, Any]]:
        mesh = mesh_data.to_trimesh().copy()
        if hasattr(mesh, "visual"):
            mesh.visual.vertex_colors = np.tile(
                np.array([200, 200, 200, 255], dtype=np.uint8),
                (mesh.vertices.shape[0], 1)
            )
        mesh_face_id = _face_id_mesh(mesh)

        size = self.config.image_size
        renderer = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size)

        rgb_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        id_mesh = pyrender.Mesh.from_trimesh(mesh_face_id, smooth=False)

        bounds = mesh.bounds
        center = bounds.mean(axis=0)
        extents = bounds[1] - bounds[0]
        radius = float(np.max(extents)) * 1.8

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        views = []
        for azimuth, elevation in self.config.resolved_views():
            cam_pose = _look_at_pose(center, radius, azimuth, elevation)

            scene_rgb = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.8, 0.8, 0.8])
            scene_rgb.add(rgb_mesh)
            scene_rgb.add(camera, pose=cam_pose)
            scene_rgb.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0), pose=cam_pose)

            rgb, depth = renderer.render(scene_rgb, flags=pyrender.RenderFlags.RGBA)
            rgb = rgb[:, :, :3]

            scene_id = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
            scene_id.add(id_mesh)
            scene_id.add(camera, pose=cam_pose)

            flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.RGBA
            face_id_rgb, depth_id = renderer.render(scene_id, flags=flags)
            face_ids = _decode_face_ids(face_id_rgb[:, :, :3])
            visible = _visible_mask(depth_id)

            views.append({"rgb": rgb, "face_ids": face_ids, "visible": visible})

        renderer.delete()
        return views


def segment_any_product(mesh_data: MeshData, ai_criteria: Optional[Dict[str, Any]] = None) -> List[MeshSegment]:
    """Convenience wrapper used by example scripts."""
    segmenter = ZeroShotMaxAccuracySegmenter()
    return segmenter.segment(mesh_data, ai_criteria)


def _build_prompt_map(ai_criteria: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not ai_criteria:
        return {}

    prompt_map: Dict[str, str] = {}
    parts = ai_criteria.get("parts", [])
    product_type = str(ai_criteria.get("product_type", "")).strip()

    for part in parts:
        if isinstance(part, dict):
            canonical = part.get("part_name") or part.get("sam_prompt") or part.get("name")
            base = part.get("sam_prompt") or canonical
            if canonical and base:
                for variant in _expand_prompt(str(base), product_type):
                    prompt_map.setdefault(variant, str(canonical))
        elif isinstance(part, str):
            for variant in _expand_prompt(part, product_type):
                prompt_map.setdefault(variant, part)

    return _dedupe_prompt_map(prompt_map)


def _expand_prompt(prompt: str, product_type: str) -> List[str]:
    raw = prompt.strip()
    if not raw:
        return []

    variants = {raw}
    spaced = raw.replace("_", " ").strip()
    if spaced:
        variants.add(spaced)

    if product_type:
        for variant in list(variants):
            if variant.startswith(product_type + " "):
                variants.add(variant[len(product_type) + 1:])
            if variant.startswith(product_type + "_"):
                variants.add(variant[len(product_type) + 1:])

    if spaced:
        tokens = [t for t in spaced.split(" ") if t]
        if len(tokens) >= 2:
            variants.add(" ".join(tokens[-2:]))
        if len(tokens) >= 1:
            variants.add(tokens[-1])

    return [v for v in variants if v]


def _label_to_primary_prompt(prompt_map: Dict[str, str], prompts: List[str]) -> Dict[str, str]:
    if not prompt_map:
        return {p: p for p in prompts}

    label_to_prompt: Dict[str, str] = {}
    for prompt, label in prompt_map.items():
        if label not in label_to_prompt:
            label_to_prompt[label] = prompt

    for prompt in prompts:
        label_to_prompt.setdefault(prompt, prompt)

    return label_to_prompt


def _prepare_clip_assets(
    config: ZeroShotConfig,
    prompt_map: Dict[str, str],
    ai_criteria: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not config.auto_mask_enabled:
        return None
    if not CLIP_AVAILABLE or not TORCH_AVAILABLE:
        logger.warning("CLIP not available - skipping auto mask filtering")
        return None

    labels = _extract_part_names(ai_criteria)
    if not labels:
        labels = list(dict.fromkeys(prompt_map.values()))
    labels = [l for l in labels if l]
    if not labels:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(config.clip_model, device=device)
    model.eval()

    text_prompts = [l.replace("_", " ") for l in labels]
    with torch.no_grad():
        tokens = clip.tokenize(text_prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return {
        "model": model,
        "preprocess": preprocess,
        "text_features": text_features,
        "labels": labels,
        "device": device,
        "threshold": config.clip_threshold
    }


def _predict_auto_masks(predictor, prompts: List[str]) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    if not prompts:
        return masks

    for prompt in prompts:
        try:
            results = predictor(text=[prompt])
            prompt_masks = _extract_masks(results)
            if prompt_masks is None or prompt_masks.size == 0:
                continue
            for mask in prompt_masks:
                masks.append(mask)
        except Exception:
            logger.debug("SAM3 auto mask prediction failed", exc_info=True)

    return masks


def _select_masks_by_area(
    masks: np.ndarray,
    min_area_ratio: float,
    max_masks: int
) -> List[np.ndarray]:
    if masks.ndim != 3:
        return []

    total_pixels = float(masks.shape[1] * masks.shape[2])
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    min_area = max(total_pixels * min_area_ratio, 1.0)
    keep = [i for i, area in enumerate(areas) if area >= min_area]
    if not keep:
        return []

    keep = sorted(keep, key=lambda i: areas[i], reverse=True)
    if max_masks > 0:
        keep = keep[:max_masks]

    return [masks[i] for i in keep]


def _clip_assign_mask(
    image: np.ndarray,
    mask: np.ndarray,
    clip_assets: Dict[str, Any]
) -> Optional[str]:
    if mask.shape != image.shape[:2]:
        mask = _resize_mask(mask, image.shape[:2])

    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    if (y_max - y_min) < 2 or (x_max - x_min) < 2:
        return None

    crop = image[y_min:y_max + 1, x_min:x_max + 1].copy()
    crop_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    crop[~crop_mask] = 0

    pil = Image.fromarray(crop)
    preprocess = clip_assets["preprocess"]
    model = clip_assets["model"]
    device = clip_assets["device"]
    text_features = clip_assets["text_features"]
    threshold = clip_assets["threshold"]

    with torch.no_grad():
        image_input = preprocess(pil).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = (image_features @ text_features.T).squeeze(0)
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())

    if best_score < threshold:
        return None

    return clip_assets["labels"][best_idx]


def _dedupe_prompt_map(prompt_map: Dict[str, str]) -> Dict[str, str]:
    seen = set()
    ordered: Dict[str, str] = {}
    for prompt, label in prompt_map.items():
        key = prompt.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered[prompt] = label
    return ordered


def _extract_masks(results) -> Optional[np.ndarray]:
    if results is None:
        return None

    res0 = results[0] if isinstance(results, list) and results else results
    masks = getattr(res0, "masks", None)
    if masks is None or masks.data is None:
        return None

    data = masks.data
    if TORCH_AVAILABLE and hasattr(data, "detach"):
        data = data.detach().cpu().numpy()

    if data.ndim == 2:
        data = data[None, ...]

    return data.astype(bool)


def _accumulate_face_scores(
    scores: np.ndarray,
    face_ids: np.ndarray,
    mask: np.ndarray,
    visible: np.ndarray,
    weight: float = 1.0
) -> None:
    if mask.shape != face_ids.shape:
        mask = _resize_mask(mask, face_ids.shape)

    combined = mask & visible
    ids = face_ids[combined]
    valid = ids >= 0
    ids = ids[valid]
    if ids.size:
        ids = ids[ids < scores.size]
    if ids.size == 0:
        return

    counts = np.bincount(ids, minlength=scores.size)
    scores[:len(counts)] += counts * weight


def _accumulate_coverage(coverage: np.ndarray, face_ids: np.ndarray, visible: np.ndarray) -> None:
    ids = face_ids[visible]
    valid = ids >= 0
    ids = ids[valid]
    if ids.size:
        ids = ids[ids < coverage.size]
    if ids.size == 0:
        return

    counts = np.bincount(ids, minlength=coverage.size)
    coverage[:len(counts)] += counts


def _resize_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = pil.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return np.array(resized) > 0


def _assign_labels(
    num_faces: int,
    prompts: List[str],
    face_scores: Dict[str, np.ndarray],
    face_coverage: np.ndarray,
    min_confidence: float,
    min_visible_views: int,
    prompt_thresholds: Optional[Dict[str, Tuple[float, int]]] = None
) -> np.ndarray:
    labels = np.array(["unlabeled"] * num_faces, dtype=object)
    if not prompts:
        labels[:] = "main_body"
        return labels

    score_stack = np.stack([face_scores[p] for p in prompts], axis=0)
    coverage = np.maximum(face_coverage, 1)
    normalized = score_stack / coverage
    best_idx = np.argmax(normalized, axis=0)
    best_score = np.max(normalized, axis=0)

    for face_idx in range(num_faces):
        best_prompt = prompts[int(best_idx[face_idx])]
        if prompt_thresholds and best_prompt in prompt_thresholds:
            conf_threshold, view_threshold = prompt_thresholds[best_prompt]
        else:
            conf_threshold, view_threshold = min_confidence, min_visible_views

        if face_coverage[face_idx] >= view_threshold and best_score[face_idx] >= conf_threshold:
            labels[face_idx] = best_prompt

    return labels


def _map_labels(labels: np.ndarray, prompt_map: Dict[str, str]) -> np.ndarray:
    if not prompt_map:
        return labels

    mapped = labels.copy()
    for idx, label in enumerate(mapped):
        mapped[idx] = prompt_map.get(label, label)
    return mapped


def _build_prompt_thresholds(
    ai_criteria: Optional[Dict[str, Any]],
    prompt_map: Dict[str, str],
    default_conf: float,
    default_views: int
) -> Dict[str, Tuple[float, int]]:
    thresholds = {prompt: (default_conf, default_views) for prompt in prompt_map}
    if not ai_criteria:
        return thresholds

    parts = ai_criteria.get("parts", [])
    if not parts:
        return thresholds

    label_sizes: Dict[str, str] = {}
    label_specials: Dict[str, List[str]] = {}
    for part in parts:
        if isinstance(part, dict):
            name = part.get("part_name") or part.get("sam_prompt") or part.get("name")
            if name:
                size = str(part.get("relative_size", "")).strip().lower()
                label_sizes[str(name)] = size
                specials = part.get("special_features") or []
                if isinstance(specials, (list, tuple)):
                    label_specials[str(name)] = [str(s).strip().lower() for s in specials if s]

    label_to_prompts: Dict[str, List[str]] = {}
    for prompt, label in prompt_map.items():
        label_to_prompts.setdefault(label, []).append(prompt)

    for label, size in label_sizes.items():
        specials = label_specials.get(label, [])
        is_fine_detail = size in ("tiny", "small") or any(
            feat in ("thin_elements", "disconnected_parts") for feat in specials
        )
        if not is_fine_detail:
            continue

        conf = min(default_conf, 0.03)
        views = max(1, default_views - 2)
        for prompt in label_to_prompts.get(label, []):
            thresholds[prompt] = (conf, views)

    return thresholds


def _merge_with_geometry(
    labels: np.ndarray,
    geom_clusters: np.ndarray,
    part_names: List[str],
    cluster_dominance: float,
    min_cluster_faces: int
) -> np.ndarray:
    if geom_clusters is None or geom_clusters.size == 0:
        return labels

    merged = labels.copy()
    cluster_ids = np.unique(geom_clusters)
    cluster_sizes = {cid: int(np.sum(geom_clusters == cid)) for cid in cluster_ids}
    cluster_order = sorted(cluster_ids, key=lambda cid: cluster_sizes[cid], reverse=True)

    label_counts: Dict[str, int] = {}
    for lab in merged.tolist():
        if lab == "unlabeled":
            continue
        label_counts[lab] = label_counts.get(lab, 0) + 1

    used = {lab for lab, count in label_counts.items() if count >= min_cluster_faces}
    remaining = [p for p in part_names if p not in used]

    for cid in cluster_order:
        mask = geom_clusters == cid
        if np.sum(mask) < min_cluster_faces:
            continue

        sam_labels = merged[mask]
        sam_labels = [l for l in sam_labels if l != "unlabeled"]
        assigned = None
        if sam_labels:
            counts: Dict[str, int] = {}
            for lab in sam_labels:
                counts[lab] = counts.get(lab, 0) + 1
            best = max(counts, key=counts.get)
            dominance = counts[best] / max(len(sam_labels), 1)
            if dominance >= cluster_dominance:
                assigned = best
                used.add(best)

        if assigned is None and remaining:
            assigned = remaining.pop(0)

        if assigned is not None:
            merged[mask] = assigned

    if "unlabeled" in merged:
        unlabeled_mask = merged == "unlabeled"
        fallback = part_names[0] if part_names else "main_body"
        merged[unlabeled_mask] = fallback

    return merged


def _smooth_labels_by_connectivity(
    mesh_data: MeshData,
    labels: np.ndarray,
    min_component_faces: int
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
    face_centers = mesh.triangles_center
    label_counts: Dict[str, int] = {}
    for lab in smoothed.tolist():
        label_counts[lab] = label_counts.get(lab, 0) + 1
    dominant_label = max(label_counts, key=label_counts.get) if label_counts else None
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
            if len(component) < min_component_faces:
                component_idx = np.array(component, dtype=int)
                candidate_mask = smoothed != label
                if np.any(candidate_mask):
                    comp_center = face_centers[component_idx].mean(axis=0)
                    candidate_indices = np.flatnonzero(candidate_mask)
                    candidate_centers = face_centers[candidate_indices]
                    nearest_idx = int(np.argmin(np.linalg.norm(candidate_centers - comp_center, axis=1)))
                    new_label = smoothed[candidate_indices[nearest_idx]]
                else:
                    new_label = dominant_label

                if new_label is not None:
                    for face_id in component:
                        smoothed[face_id] = new_label
            continue

        boundary_ratio = boundary_edges / max(total_edges, 1)
        needs_relabel = len(component) < min_component_faces or boundary_ratio >= 0.6
        if not needs_relabel:
            continue

        new_label = max(neighbor_counts, key=neighbor_counts.get)
        for face_id in component:
            smoothed[face_id] = new_label

    return smoothed


def _extract_part_names(ai_criteria: Optional[Dict[str, Any]]) -> List[str]:
    if not ai_criteria:
        return []
    names = []
    for part in ai_criteria.get("parts", []):
        if isinstance(part, dict):
            name = part.get("part_name") or part.get("sam_prompt") or part.get("name")
            if name:
                names.append(str(name))
        elif isinstance(part, str):
            names.append(part)
    return names


def _select_auto_prompts(
    ai_criteria: Optional[Dict[str, Any]],
    prompt_map: Dict[str, str],
    prompts: List[str]
) -> List[str]:
    if not ai_criteria or not prompts:
        return prompts

    raw_prompts: List[str] = []
    for part in ai_criteria.get("parts", []):
        if isinstance(part, dict):
            prompt = part.get("sam_prompt") or part.get("part_name") or part.get("name")
            if prompt:
                raw_prompts.append(str(prompt).strip())
        elif isinstance(part, str):
            raw_prompts.append(part.strip())

    if not raw_prompts:
        return prompts

    prompt_set = set(prompts)
    product_type = str(ai_criteria.get("product_type", "")).strip()
    selected: List[str] = []
    for raw in raw_prompts:
        if not raw:
            continue
        if raw in prompt_set:
            selected.append(raw)
            continue
        for variant in _expand_prompt(raw, product_type):
            if variant in prompt_set:
                selected.append(variant)
                break

    return list(dict.fromkeys(selected)) or prompts


def _segments_from_labels(
    mesh_data: MeshData,
    labels: np.ndarray,
    min_faces: int,
    min_small_faces: int,
    target_parts: List[str]
) -> List[MeshSegment]:
    mesh = mesh_data.to_trimesh()
    segments: List[MeshSegment] = []

    unique_labels = list(dict.fromkeys(labels.tolist()))
    for idx, label in enumerate(unique_labels):
        face_indices = np.where(labels == label)[0]
        face_count = len(face_indices)
        if face_count < min_faces:
            if label not in target_parts or face_count < min_small_faces:
                continue

        submesh = mesh.submesh([face_indices], append=True, repair=False)
        sub_data = MeshData.from_trimesh(submesh, name=label)
        segments.append(MeshSegment(
            segment_id=idx,
            label=label,
            mesh_data=sub_data,
            confidence=0.7,
            properties={"face_count": sub_data.num_faces}
        ))

    if not segments:
        segments.append(MeshSegment(0, "main_body", mesh_data))

    return segments


def _apply_geometry_gating(
    face_scores: Dict[str, np.ndarray],
    mesh_data: MeshData,
    ai_criteria: Optional[Dict[str, Any]],
    prompt_map: Dict[str, str],
    config: ZeroShotConfig
) -> None:
    if not ai_criteria or config.geometry_weight <= 0:
        return

    criteria_by_label = _criteria_by_label(ai_criteria)
    if not criteria_by_label:
        return

    features = _compute_face_features(mesh_data)
    if features is None:
        return

    for prompt, scores in face_scores.items():
        label = prompt_map.get(prompt, prompt)
        criteria = criteria_by_label.get(label)
        if not criteria:
            continue

        geometry_score = _geometry_score_for_part(criteria, features, config)
        if geometry_score is None:
            continue
        if geometry_score.shape[0] != scores.shape[0]:
            logger.warning(
                "Skipping geometry gating for '%s' due to face count mismatch (scores=%d, geometry=%d)",
                label,
                scores.shape[0],
                geometry_score.shape[0]
            )
            continue

        scores *= (1.0 - config.geometry_weight + config.geometry_weight * geometry_score)


def _criteria_by_label(ai_criteria: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    criteria_map: Dict[str, Dict[str, Any]] = {}
    for part in ai_criteria.get("parts", []):
        if not isinstance(part, dict):
            continue
        name = part.get("part_name") or part.get("sam_prompt") or part.get("name")
        if name:
            criteria_map[str(name)] = part
    return criteria_map


def _compute_face_features(mesh_data: MeshData) -> Optional[Dict[str, Any]]:
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
        "height_axis": height_axis,
        "length_axis": length_axis,
        "width_axis": width_axis
    }


def _geometry_score_for_part(
    criteria: Dict[str, Any],
    features: Dict[str, Any],
    config: ZeroShotConfig
) -> Optional[np.ndarray]:
    weights = []
    components = []

    norm_coords = features["norm_coords"]
    height_axis = features["height_axis"]
    length_axis = features["length_axis"]
    width_axis = features["width_axis"]

    height_range = criteria.get("height_range")
    if isinstance(height_range, (list, tuple)) and len(height_range) == 2:
        lo, hi = float(height_range[0]), float(height_range[1])
        height_score = _range_score(norm_coords[:, height_axis], lo, hi, config.geometry_softness)
        weights.append(config.geometry_height_weight)
        components.append(height_score)

    position = str(criteria.get("position", "")).strip().lower()
    if position and position not in ("any", "all_around"):
        position_score = _position_score(
            position,
            norm_coords[:, length_axis],
            norm_coords[:, width_axis],
            norm_coords[:, height_axis],
            config.geometry_softness
        )
        if position_score is not None:
            weights.append(config.geometry_position_weight)
            components.append(position_score)

    normal_direction = str(criteria.get("normal_direction", "")).strip().lower()
    if normal_direction and normal_direction != "any":
        normal_score = _normal_score(normal_direction, features)
        if normal_score is not None:
            weights.append(config.geometry_normal_weight)
            components.append(normal_score)

    if not components:
        return None

    total_weight = sum(weights) if weights else 0.0
    if total_weight <= 0:
        return None

    combined = np.zeros_like(components[0])
    for weight, component in zip(weights, components):
        combined += weight * component

    return combined / total_weight


def _range_score(values: np.ndarray, lo: float, hi: float, softness: float) -> np.ndarray:
    if hi < lo:
        lo, hi = hi, lo
    softness = max(softness, 1e-6)
    score = np.ones_like(values)
    below = values < lo
    above = values > hi
    score[below] = np.clip(1.0 - (lo - values[below]) / softness, 0.0, 1.0)
    score[above] = np.clip(1.0 - (values[above] - hi) / softness, 0.0, 1.0)
    return score


def _point_score(values: np.ndarray, target: float, softness: float) -> np.ndarray:
    softness = max(softness, 1e-6)
    return np.clip(1.0 - np.abs(values - target) / softness, 0.0, 1.0)


def _position_score(
    position: str,
    length_norm: np.ndarray,
    width_norm: np.ndarray,
    height_norm: np.ndarray,
    softness: float
) -> Optional[np.ndarray]:
    if position in ("any", "all_around"):
        return np.ones_like(length_norm)

    if position == "center":
        return _point_score(length_norm, 0.5, softness) * _point_score(width_norm, 0.5, softness)
    if position == "front":
        return np.maximum(
            _point_score(length_norm, 0.15, softness),
            _point_score(length_norm, 0.85, softness)
        )
    if position == "back":
        return np.maximum(
            _point_score(length_norm, 0.15, softness),
            _point_score(length_norm, 0.85, softness)
        )
    if position == "left":
        return np.maximum(
            _point_score(width_norm, 0.15, softness),
            _point_score(width_norm, 0.85, softness)
        )
    if position == "right":
        return np.maximum(
            _point_score(width_norm, 0.15, softness),
            _point_score(width_norm, 0.85, softness)
        )
    if position == "top":
        return _point_score(height_norm, 0.85, softness)
    if position == "bottom":
        return _point_score(height_norm, 0.15, softness)

    return None


def _normal_score(normal_direction: str, features: Dict[str, Any]) -> Optional[np.ndarray]:
    normals = features["normals"]
    height_axis = features["height_axis"]
    length_axis = features["length_axis"]
    width_axis = features["width_axis"]
    radial_dirs = features["radial_dirs"]

    if normal_direction == "up":
        return np.clip(normals[:, height_axis], 0.0, 1.0)
    if normal_direction == "down":
        return np.clip(-normals[:, height_axis], 0.0, 1.0)
    if normal_direction == "forward":
        return np.clip(np.abs(normals[:, length_axis]), 0.0, 1.0)
    if normal_direction == "backward":
        return np.clip(np.abs(normals[:, length_axis]), 0.0, 1.0)
    if normal_direction == "right":
        return np.clip(np.abs(normals[:, width_axis]), 0.0, 1.0)
    if normal_direction == "left":
        return np.clip(np.abs(normals[:, width_axis]), 0.0, 1.0)
    if normal_direction == "outward":
        return np.clip(np.sum(normals * radial_dirs, axis=1), 0.0, 1.0)
    if normal_direction == "inward":
        return np.clip(-np.sum(normals * radial_dirs, axis=1), 0.0, 1.0)

    return None


def _face_id_mesh(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    faces = mesh.faces
    vertices = mesh.vertices
    face_vertices = vertices[faces].reshape(-1, 3)
    face_indices = np.arange(face_vertices.shape[0]).reshape(-1, 3)

    colors = np.zeros((len(faces), 3), dtype=np.uint8)
    for i in range(len(faces)):
        colors[i] = _encode_face_id(i)

    vertex_colors = np.repeat(colors, 3, axis=0)

    id_mesh = trimesh.Trimesh(
        vertices=face_vertices,
        faces=face_indices,
        process=False
    )
    id_mesh.visual.vertex_colors = vertex_colors
    return id_mesh


def _encode_face_id(face_id: int) -> np.ndarray:
    r = (face_id & 0xFF)
    g = (face_id >> 8) & 0xFF
    b = (face_id >> 16) & 0xFF
    return np.array([r, g, b], dtype=np.uint8)


def _decode_face_ids(face_id_rgb: np.ndarray) -> np.ndarray:
    r = face_id_rgb[:, :, 0].astype(np.int64)
    g = face_id_rgb[:, :, 1].astype(np.int64)
    b = face_id_rgb[:, :, 2].astype(np.int64)
    face_ids = r + (g << 8) + (b << 16)

    max_id = face_ids.max(initial=0)
    if max_id > 10_000_000:
        face_ids = np.where(face_ids > 10_000_000, -1, face_ids)

    return face_ids


def _visible_mask(depth: np.ndarray) -> np.ndarray:
    if depth is None:
        return np.zeros((1, 1), dtype=bool)
    return np.isfinite(depth) & (depth > 0)


def _look_at_pose(center: np.ndarray, radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    x = center[0] + radius * np.cos(el) * np.cos(az)
    y = center[1] + radius * np.cos(el) * np.sin(az)
    z = center[2] + radius * np.sin(el)

    camera_position = np.array([x, y, z])
    forward = center - camera_position
    forward /= np.linalg.norm(forward)

    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose
