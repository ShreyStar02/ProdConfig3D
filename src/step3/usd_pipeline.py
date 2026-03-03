"""
USD Pipeline Module - Handles USD/Omniverse conversions and operations.
Creates multi-mesh USD from segmented meshes.
"""
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
import json
import tempfile
import shutil
import re
import os
import threading
from contextlib import contextmanager
from urllib.parse import unquote

# Try importing USD libraries
try:
    from pxr import Usd, UsdGeom, Gf, Vt, UsdShade, Sdf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

from ..step1.mesh_processor import MeshData
from ..step1.segmentation import MeshSegment
from ..config import OmniverseConfig

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_known_usd_stderr_warnings():
    """Filter specific noisy USD metadata warnings while preserving all other stderr output."""
    patterns = ("hide_in_stage_window", "no_delete")

    if not USD_AVAILABLE:
        yield
        return

    read_fd = None
    write_fd = None
    original_stderr_fd = None
    reader_thread = None

    try:
        read_fd, write_fd = os.pipe()
        original_stderr_fd = os.dup(2)

        def _forward_filtered_stderr() -> None:
            with os.fdopen(read_fd, "r", encoding="utf-8", errors="ignore") as stream:
                for line in stream:
                    lowered = line.lower()
                    if any(pattern in lowered for pattern in patterns):
                        continue
                    try:
                        os.write(original_stderr_fd, line.encode("utf-8", errors="ignore"))
                    except OSError:
                        return

        reader_thread = threading.Thread(target=_forward_filtered_stderr, daemon=True)
        reader_thread.start()

        os.dup2(write_fd, 2)
        os.close(write_fd)
        write_fd = None
    except Exception:
        # Never let warning suppression block the actual pipeline execution.
        yield
        return

    try:
        yield
    finally:
        if original_stderr_fd is not None:
            try:
                os.dup2(original_stderr_fd, 2)
            except OSError:
                pass
            try:
                os.close(original_stderr_fd)
            except OSError:
                pass

        if write_fd is not None:
            try:
                os.close(write_fd)
            except OSError:
                pass

        if reader_thread is not None:
            reader_thread.join(timeout=0.5)


class USDExporter:
    """
    Exports segmented meshes to USD format with proper hierarchy.
    Creates multi-mesh USD with named components.
    """

    def __init__(self, config: Optional[OmniverseConfig] = None):
        self.config = config or OmniverseConfig()
        self._check_dependencies()

    def _check_dependencies(self):
        if not USD_AVAILABLE:
            raise RuntimeError(
                "ERROR: This is not supported/compatible - USD (pxr) is required for Step3 operations. "
                "Install dependencies using requirements.txt via setup.sh."
            )

    def export_multi_mesh(self, segments: List[MeshSegment],
                         output_path: Path,
                         root_name: str = "Model",
                         up_axis: str = "Y") -> Path:
        """
        Export segmented meshes as multi-mesh USD.

        Args:
            segments: List of mesh segments with labels
            output_path: Output USD file path
            root_name: Name for root transform
            up_axis: Up axis ("Y" or "Z")

        Returns:
            Path to created USD file
        """
        if not USD_AVAILABLE:
            raise RuntimeError(
                "ERROR: This is not supported/compatible - USD (pxr) is required for multi-mesh export."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {len(segments)} segments to USD: {output_path}")

        # Create new USD stage
        with _suppress_known_usd_stderr_warnings():
            stage = Usd.Stage.CreateNew(str(output_path))

        # Set stage metadata
        stage.SetMetadata('upAxis', UsdGeom.Tokens.y if up_axis == "Y" else UsdGeom.Tokens.z)
        stage.SetMetadata('metersPerUnit', 0.01)  # cm

        # Sanitize root name for USD (replace invalid chars with underscores)
        safe_root_name = self._sanitize_identifier(root_name, fallback="Model")

        # Create root transform
        root_path = f"/{safe_root_name}"
        root_xform = UsdGeom.Xform.Define(stage, root_path)
        stage.SetDefaultPrim(root_xform.GetPrim())

        # Create a scope for all meshes
        meshes_scope = UsdGeom.Scope.Define(stage, f"{root_path}/Meshes")

        # Export each segment as a mesh
        for segment in segments:
            # Sanitize segment label for USD path
            safe_label = self._sanitize_identifier(segment.label, fallback="segment")

            mesh_path = f"{root_path}/Meshes/{safe_label}"
            self._create_usd_mesh(stage, mesh_path, segment)

        # Save stage
        with _suppress_known_usd_stderr_warnings():
            stage.Save()
        logger.info(f"USD file saved: {output_path}")

        # Also export metadata
        self._export_metadata(segments, output_path.with_suffix('.json'))

        return output_path

    def apply_curated_materials(self,
                                usd_path: Path,
                                curated: Dict[str, Any],
                                root_name: str = "Model") -> None:
        """Apply curated MDL materials to mesh prims using first candidate source_path."""
        if not USD_AVAILABLE:
            logger.warning("USD (pxr) not available; skipping material application")
            return

        usd_path = Path(usd_path)
        if not usd_path.exists():
            logger.warning("USD file not found for material apply: %s", usd_path)
            return

        with _suppress_known_usd_stderr_warnings():
            stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            logger.warning("Failed to open USD stage for material apply: %s", usd_path)
            return

        mesh_lookup = self._build_mesh_lookup(stage)
        if not mesh_lookup:
            logger.warning("No USD mesh prims found in stage: %s", usd_path)
            return

        segments = curated.get("segments", []) if isinstance(curated, dict) else []
        if not segments:
            logger.info("No curated segments found to apply")
            return

        applied = 0
        skipped = 0
        missing_mdl = 0
        materials_root = f"/{self._sanitize_identifier(root_name, fallback='Model')}/Materials"

        for idx, segment in enumerate(segments):
            if not isinstance(segment, dict):
                skipped += 1
                continue

            candidate = self._first_candidate_with_source(segment)
            if not candidate:
                logger.warning("Skipping segment '%s': no candidate with source_path", segment.get("label", idx))
                skipped += 1
                continue

            source_path = self._resolve_existing_mdl_path(candidate)
            if not source_path:
                logger.warning(
                    "Skipping segment '%s': MDL source file not found (source_path=%s rel_path=%s source_root=%s)",
                    segment.get("label", idx),
                    candidate.get("source_path"),
                    candidate.get("rel_path"),
                    candidate.get("source_root"),
                )
                skipped += 1
                missing_mdl += 1
                continue

            mesh_prim = self._resolve_mesh_prim(segment, mesh_lookup)
            if mesh_prim is None:
                logger.warning(
                    "Skipping segment '%s': no matching mesh prim (mesh_path=%s)",
                    segment.get("label", idx),
                    segment.get("mesh_path")
                )
                skipped += 1
                continue

            preferred_name = str(candidate.get("name") or segment.get("inferred_group") or segment.get("label") or "material")
            material_name = self._sanitize_identifier(f"{preferred_name}_{idx}", fallback=f"material_{idx}")
            material_path = f"{materials_root}/{material_name}"
            shader_path = f"{material_path}/Shader"

            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("mdlMaterial")
            shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)

            asset_path = self._to_asset_path(str(source_path))
            shader.SetSourceAsset(Sdf.AssetPath(asset_path), "mdl")

            exports = self._discover_mdl_exports(str(source_path))
            sub_identifier = self._select_mdl_sub_identifier(exports, preferred_name)
            if sub_identifier:
                shader.SetSourceAssetSubIdentifier(sub_identifier, "mdl")

            # Ensure a named output exists for material surface connection.
            shader.CreateOutput("out", Sdf.ValueTypeNames.Token)

            material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
            applied += 1

        with _suppress_known_usd_stderr_warnings():
            stage.Save()
        logger.info(
            "Applied curated materials to USD: applied=%d skipped=%d missing_mdl=%d file=%s",
            applied,
            skipped,
            missing_mdl,
            usd_path,
        )

    def _build_mesh_lookup(self, stage: "Usd.Stage") -> Dict[str, "Usd.Prim"]:
        lookup: Dict[str, "Usd.Prim"] = {}
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue

            prim_name = prim.GetName()
            keys = {
                prim_name,
                prim_name.lower(),
                self._sanitize_identifier(prim_name, fallback=prim_name),
                self._sanitize_identifier(prim_name, fallback=prim_name).lower(),
            }
            for key in keys:
                lookup.setdefault(key, prim)
        return lookup

    def _resolve_mesh_prim(self, segment: Dict[str, Any], mesh_lookup: Dict[str, "Usd.Prim"]) -> Optional["Usd.Prim"]:
        keys: List[str] = []

        mesh_path = str(segment.get("mesh_path") or "").strip()
        if mesh_path:
            mesh_name = self._extract_leaf_name_from_mesh_path(mesh_path)
            keys.extend([mesh_name, mesh_name.lower(), self._sanitize_identifier(mesh_name, fallback=mesh_name), self._sanitize_identifier(mesh_name, fallback=mesh_name).lower()])

        label = str(segment.get("label") or "").strip()
        if label:
            keys.extend([label, label.lower(), self._sanitize_identifier(label, fallback=label), self._sanitize_identifier(label, fallback=label).lower()])

        for key in keys:
            if key in mesh_lookup:
                return mesh_lookup[key]

        return None

    def _extract_leaf_name_from_mesh_path(self, mesh_path: str) -> str:
        tokens = [tok.strip() for tok in mesh_path.split("->") if tok.strip()]
        leaf = tokens[-1] if tokens else mesh_path
        match = re.match(r"^(.*?)\s*\((Mesh|Xform)\)\s*$", leaf)
        if match:
            return (match.group(1) or leaf).strip()
        return leaf.strip()

    def _first_candidate_with_source(self, segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = segment.get("candidates", [])
        if not isinstance(candidates, list):
            return None
        for candidate in candidates:
            if isinstance(candidate, dict) and str(candidate.get("source_path") or "").strip():
                return candidate
        return None

    def _discover_mdl_exports(self, mdl_path: str) -> List[str]:
        try:
            content = Path(mdl_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        exported = re.findall(r"export\s+material\s+([A-Za-z0-9_]+)", content)
        return list(dict.fromkeys(exported))

    def _create_usd_mesh(self, stage: "Usd.Stage", mesh_path: str, segment: MeshSegment):
        """Create a UsdGeom.Mesh from segment data"""
        mesh_data = segment.mesh_data

        # Define mesh
        usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        # Set vertices (points)
        points = [Gf.Vec3f(*v) for v in mesh_data.vertices]
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

        # Set face vertex counts (all triangles = 3)
        face_counts = [3] * len(mesh_data.faces)
        usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))

        # Set face vertex indices
        indices = mesh_data.faces.flatten().tolist()
        usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))

        # Set normals if available
        if mesh_data.normals is not None:
            normals = [Gf.Vec3f(*n) for n in mesh_data.normals]
            usd_mesh.GetNormalsAttr().Set(Vt.Vec3fArray(normals))
            usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set UV coordinates if available

    def _select_mdl_sub_identifier(self, exports: List[str], preferred_name: Optional[str]) -> Optional[str]:
        if not exports:
            # Do not guess an unverified subIdentifier.
            return None

        if preferred_name and preferred_name in exports:
            return preferred_name

        return exports[0]

    def _to_asset_path(self, path: str) -> str:
        try:
            path_obj = Path(path)
            if path_obj.is_absolute():
                # Omniverse MDL resolution is more reliable with filesystem-style paths
                # than URL-encoded file:// URIs on Windows.
                return path_obj.resolve().as_posix()
            return path_obj.as_posix()
        except Exception:
            return str(path)

    def _resolve_existing_mdl_path(self, candidate: Dict[str, Any]) -> Optional[Path]:
        source_raw = str(candidate.get("source_path") or "").strip()
        rel_raw = str(candidate.get("rel_path") or "").strip()
        root_raw = str(candidate.get("source_root") or "").strip()

        candidates: List[Path] = []

        normalized_source = self._normalize_source_path(source_raw)
        if normalized_source:
            candidates.append(normalized_source)

        if rel_raw and root_raw:
            candidates.append(Path(root_raw) / rel_raw)

        for path in candidates:
            try:
                if path.exists() and path.is_file():
                    return path.resolve()
            except Exception:
                continue

        return None

    def _normalize_source_path(self, raw_path: str) -> Optional[Path]:
        if not raw_path:
            return None

        cleaned = raw_path.strip().strip("@")
        if cleaned.lower().startswith("file://"):
            cleaned = cleaned[7:]
            if cleaned.startswith("/") and len(cleaned) > 2 and cleaned[2] == ":":
                # file:///C:/... -> C:/...
                cleaned = cleaned[1:]

        cleaned = unquote(cleaned)
        return Path(cleaned)

    def _collect_mdl_search_paths(self, mdl_path: str) -> List[str]:
        path_obj = Path(mdl_path)
        parents = list(path_obj.parents)
        search_paths = [str(path_obj.parent)]

        if len(parents) > 1:
            search_paths.append(str(parents[0]))
        if len(parents) > 2:
            search_paths.append(str(parents[1]))

        for parent in parents:
            if (parent / "nvidia").exists():
                search_paths.append(str(parent))
                break

        # Allow platform-specific MDL search locations through environment configuration.
        mdl_roots_raw = os.getenv("PRODCONFIG_MDL_ROOTS", "")
        for token in re.split(r"[;,]", mdl_roots_raw):
            token = token.strip()
            if not token:
                continue
            root = Path(token)
            if root.exists():
                search_paths.append(str(root))

        return search_paths

    def _sanitize_identifier(self, value: str, fallback: str) -> str:
        import re

        safe_value = re.sub(r"[^a-zA-Z0-9_]", "_", value or "")
        if not safe_value:
            safe_value = fallback
        if safe_value[0].isdigit():
            safe_value = "_" + safe_value
        return safe_value

    def _export_metadata(self, segments: List[MeshSegment], output_path: Path):
        """Export segment metadata as JSON"""
        metadata = {
            "version": "1.0",
            "segments": []
        }

        for segment in segments:
            seg_data = {
                "id": segment.segment_id,
                "label": segment.label,
                "confidence": segment.confidence,
                "face_count": segment.properties.get('face_count', 0),
                "area": segment.properties.get('area', 0),
                "center": segment.properties.get('center', [0, 0, 0]),
                "bounds": segment.properties.get('bounds', [[0, 0, 0], [0, 0, 0]])
            }
            metadata["segments"].append(seg_data)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved: {output_path}")

    def _export_fallback(self, segments: List[MeshSegment], output_path: Path, root_name: str) -> Path:
        """
        Fallback export when USD is not available.
        Creates a directory with individual mesh files.
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("Neither USD nor trimesh available for export")

        output_dir = output_path.parent / f"{output_path.stem}_meshes"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"USD not available, exporting to directory: {output_dir}")

        for segment in segments:
            mesh = segment.mesh_data.to_trimesh()
            mesh_file = output_dir / f"{segment.label}.glb"
            mesh.export(str(mesh_file))
            logger.debug(f"  Exported: {mesh_file}")

        # Export metadata
        self._export_metadata(segments, output_dir / "metadata.json")

        # Create scene file referencing all meshes
        scene_data = {
            "root_name": root_name,
            "meshes": [f"{seg.label}.glb" for seg in segments]
        }
        with open(output_dir / "scene.json", 'w') as f:
            json.dump(scene_data, f, indent=2)

        return output_dir

    def export_multi_mesh_glb(self, segments: List[MeshSegment],
                              output_path: Path,
                              root_name: str = "Model") -> Path:
        """
        Export segmented meshes as a multi-mesh GLB file.
        Each segment becomes a named node in the GLB scene.

        Args:
            segments: List of mesh segments with labels
            output_path: Output GLB file path
            root_name: Name for root transform

        Returns:
            Path to created GLB file
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for GLB export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {len(segments)} segments to GLB: {output_path}")

        # Create a trimesh Scene to hold all meshes
        scene = trimesh.Scene()

        for segment in segments:
            # Convert segment mesh data to trimesh (preserves vertex colors and UVs)
            mesh = segment.mesh_data.to_trimesh()

            # Ensure vertex colors are preserved if available
            if segment.mesh_data.vertex_colors is not None:
                try:
                    mesh.visual.vertex_colors = segment.mesh_data.vertex_colors
                    logger.debug(f"  Set vertex colors for {segment.label}: {len(segment.mesh_data.vertex_colors)}")
                except Exception as e:
                    logger.debug(f"  Could not set vertex colors for {segment.label}: {e}")

            # Add metadata to the mesh
            mesh.metadata['segment_id'] = segment.segment_id
            mesh.metadata['label'] = segment.label
            mesh.metadata['confidence'] = segment.confidence
            if segment.properties:
                for key, value in segment.properties.items():
                    if isinstance(value, (int, float, str, bool)):
                        mesh.metadata[f'props_{key}'] = value

            # Add mesh to scene with its label as the node name
            scene.add_geometry(mesh, node_name=segment.label, geom_name=segment.label)
            logger.debug(f"  Added mesh to GLB: {segment.label}")

        # Export scene to GLB
        scene.export(str(output_path), file_type='glb')
        logger.info(f"GLB file saved: {output_path}")

        return output_path


class USDImporter:
    """
    Imports USD files and converts to internal mesh format.
    """

    def __init__(self, config: Optional[OmniverseConfig] = None):
        self.config = config or OmniverseConfig()

    def import_usd(self, usd_path: Path) -> Tuple[List[MeshData], Dict[str, Any]]:
        """
        Import USD file and extract meshes.

        Returns:
            Tuple of (list of MeshData, metadata dict)
        """
        if not USD_AVAILABLE:
            raise ImportError(
                "ERROR: This is not supported/compatible - USD (pxr) is required for import. "
                "Install dependencies using requirements.txt via setup.sh."
            )

        usd_path = Path(usd_path)
        if not usd_path.exists():
            raise FileNotFoundError(f"USD input not found: {usd_path}")

        resolved_path = usd_path.resolve()
        logger.info(f"Importing USD: {usd_path}")

        with _suppress_known_usd_stderr_warnings():
            stage = Usd.Stage.Open(resolved_path.as_posix())
        if not stage:
            raise ValueError(
                "Failed to open USD file: "
                f"{resolved_path}. Ensure the file is a valid .usd/.usda/.usdc/.usdz layer."
            )

        meshes = []
        metadata = {
            "file_path": str(resolved_path),
            "up_axis": str(stage.GetMetadata('upAxis')),
            "meters_per_unit": stage.GetMetadata('metersPerUnit') or 1.0
        }

        # Traverse stage and find meshes
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_data = self._extract_mesh(prim)
                if mesh_data:
                    meshes.append(mesh_data)
                    logger.debug(f"  Extracted mesh: {prim.GetPath()}")

        logger.info(f"Imported {len(meshes)} meshes from USD")
        return meshes, metadata

    def _extract_mesh(self, prim: "Usd.Prim") -> Optional[MeshData]:
        """Extract MeshData from USD mesh prim"""
        usd_mesh = UsdGeom.Mesh(prim)

        # Get points
        points_attr = usd_mesh.GetPointsAttr()
        if not points_attr:
            return None

        points = np.array(points_attr.Get())
        if len(points) == 0:
            return None

        # Get face indices
        face_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
        face_indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())

        # Convert to triangles if needed
        faces = self._convert_to_triangles(face_counts, face_indices)

        # Get normals if available
        normals = None
        normals_attr = usd_mesh.GetNormalsAttr()
        if normals_attr and normals_attr.Get():
            normals = np.array(normals_attr.Get())

        # Get UVs if available
        uvs = None
        primvar_api = UsdGeom.PrimvarsAPI(usd_mesh)
        st_primvar = primvar_api.GetPrimvar("st")
        if st_primvar and st_primvar.Get():
            uvs = np.array(st_primvar.Get())

        return MeshData(
            vertices=points,
            faces=faces,
            normals=normals,
            uv_coords=uvs,
            name=prim.GetName()
        )

    def _convert_to_triangles(self, face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
        """Convert polygon faces to triangles using fan triangulation"""
        triangles = []
        idx = 0

        for count in face_counts:
            if count == 3:
                # Already a triangle
                triangles.append(face_indices[idx:idx+3])
            elif count > 3:
                # Fan triangulation
                for i in range(1, count - 1):
                    triangles.append([
                        face_indices[idx],
                        face_indices[idx + i],
                        face_indices[idx + i + 1]
                    ])
            idx += count

        return np.array(triangles)


class ModelToUSDConverter:
    """
    Converts common 3D formats to USD and supports USD passthrough.
    Prefers Omniverse Kit conversion when available/configured.
    """

    def __init__(self, config: Optional[OmniverseConfig] = None):
        self.config = config or OmniverseConfig()

    def convert(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Convert input model to USD.

        Args:
            input_path: Input model file (USD/GLB/GLTF/FBX/etc.)
            output_path: Output USD path (optional, derived from input if not specified)

        Returns:
            Path to converted USD file
        """
        input_path = Path(input_path)
        input_suffix = input_path.suffix.lower()

        if input_suffix in {".usd", ".usda", ".usdc", ".usdz"}:
            if output_path is None or Path(output_path) == input_path:
                logger.info("Input is already USD-compatible: %s", input_path)
                return input_path

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # For USD family files, avoid byte-copying across potentially different
            # container/encoding types (e.g., .usdz -> .usd). Export through a stage.
            if USD_AVAILABLE:
                with _suppress_known_usd_stderr_warnings():
                    stage = Usd.Stage.Open(str(input_path))
                if not stage:
                    raise ValueError(f"Failed to open USD-compatible input: {input_path}")
                with _suppress_known_usd_stderr_warnings():
                    stage.Export(str(output_path))
                logger.info("Exported USD-compatible input to: %s", output_path)
                return output_path

            # Fallback when pxr is unavailable: keep original file extension to avoid
            # writing misleading content under a different suffix.
            safe_output = output_path.with_suffix(input_path.suffix)
            shutil.copy2(input_path, safe_output)
            logger.info("Copied USD-compatible input to: %s", safe_output)
            return safe_output

            return output_path

        if output_path is None:
            output_path = input_path.with_suffix('.usd')
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting {input_path} to USD")

        if self.config.use_kit_app:
            try:
                return self._convert_via_omniverse_kit(input_path, output_path)
            except Exception as exc:
                logger.warning("Omniverse Kit conversion failed, falling back to local conversion: %s", exc)

        if USD_AVAILABLE and TRIMESH_AVAILABLE:
            return self._convert_via_trimesh_usd(input_path, output_path)
        elif TRIMESH_AVAILABLE:
            # Export as intermediate format that can be used
            return self._convert_via_trimesh(input_path, output_path)
        else:
            raise ImportError("Neither USD nor trimesh available for conversion")

    def _convert_via_omniverse_kit(self, input_path: Path, output_path: Path) -> Path:
        """Convert via Omniverse Kit asset converter when available."""
        integration = OmniverseKitIntegration(self.config)
        if not integration.is_kit_available:
            raise RuntimeError("Omniverse Kit runtime is not available in current environment")

        # convert_with_kit is async; run from sync context.
        return asyncio.run(integration.convert_with_kit(input_path, output_path))

    def _convert_via_trimesh_usd(self, input_path: Path, output_path: Path) -> Path:
        """Convert using trimesh to load and USD to export"""
        # Load with trimesh
        scene = trimesh.load(str(input_path))

        # Create USD stage
        with _suppress_known_usd_stderr_warnings():
            stage = Usd.Stage.CreateNew(str(output_path))
        stage.SetMetadata('upAxis', UsdGeom.Tokens.y)

        root_name = input_path.stem
        root_xform = UsdGeom.Xform.Define(stage, f"/{root_name}")
        stage.SetDefaultPrim(root_xform.GetPrim())

        if isinstance(scene, trimesh.Scene):
            for name, geometry in scene.geometry.items():
                if isinstance(geometry, trimesh.Trimesh):
                    mesh_path = f"/{root_name}/{name}"
                    self._add_trimesh_to_stage(stage, mesh_path, geometry)
        elif isinstance(scene, trimesh.Trimesh):
            mesh_path = f"/{root_name}/mesh"
            self._add_trimesh_to_stage(stage, mesh_path, scene)

        with _suppress_known_usd_stderr_warnings():
            stage.Save()
        logger.info(f"Converted to USD: {output_path}")
        return output_path

    def _add_trimesh_to_stage(self, stage: "Usd.Stage", mesh_path: str, mesh: "trimesh.Trimesh"):
        """Add a trimesh mesh to USD stage"""
        usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        # Points
        points = [Gf.Vec3f(*v) for v in mesh.vertices]
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

        # Faces
        face_counts = [3] * len(mesh.faces)
        usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))

        indices = mesh.faces.flatten().tolist()
        usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))

        # Normals
        if mesh.vertex_normals is not None:
            normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals]
            usd_mesh.GetNormalsAttr().Set(Vt.Vec3fArray(normals))

        usd_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    def _convert_via_trimesh(self, input_path: Path, output_path: Path) -> Path:
        """Convert using trimesh only (exports to GLB with metadata)"""
        scene = trimesh.load(str(input_path))

        # Re-export with processing
        if isinstance(scene, trimesh.Scene):
            scene.export(str(output_path.with_suffix('.glb')))
        else:
            scene.export(str(output_path.with_suffix('.glb')))

        return output_path.with_suffix('.glb')

    def export_usd_to_glb(self, usd_path: Path, output_path: Path) -> Path:
        """Export a USD stage to a multi-mesh GLB scene."""
        if not USD_AVAILABLE or not TRIMESH_AVAILABLE:
            raise ImportError("Both usd-core (pxr) and trimesh are required for USD->GLB export")

        usd_path = Path(usd_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with _suppress_known_usd_stderr_warnings():
            stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            raise ValueError(f"Failed to open USD file for GLB export: {usd_path}")

        scene = trimesh.Scene()

        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue

            usd_mesh = UsdGeom.Mesh(prim)
            points_attr = usd_mesh.GetPointsAttr()
            face_counts_attr = usd_mesh.GetFaceVertexCountsAttr()
            face_indices_attr = usd_mesh.GetFaceVertexIndicesAttr()

            if not points_attr or not face_counts_attr or not face_indices_attr:
                continue

            points = points_attr.Get() or []
            face_counts = face_counts_attr.Get() or []
            face_indices = face_indices_attr.Get() or []
            if not points or not face_counts or not face_indices:
                continue

            vertices = np.array(points, dtype=np.float64)
            faces = self._triangulate_faces(face_counts, face_indices)
            if len(vertices) == 0 or len(faces) == 0:
                continue

            # Preserve authored transforms by baking local-to-world into vertices.
            xformable = UsdGeom.Xformable(prim)
            world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transformed = []
            for vert in vertices:
                v = world_xf.TransformAffine(Gf.Vec3d(float(vert[0]), float(vert[1]), float(vert[2])))
                transformed.append([float(v[0]), float(v[1]), float(v[2])])

            mesh = trimesh.Trimesh(vertices=np.array(transformed), faces=np.array(faces), process=False)
            scene.add_geometry(mesh, node_name=prim.GetName(), geom_name=prim.GetName())

        scene.export(str(output_path), file_type='glb')
        logger.info("Exported GLB from USD: %s", output_path)
        return output_path

    def _triangulate_faces(self, face_counts: List[int], face_indices: List[int]) -> List[List[int]]:
        triangles: List[List[int]] = []
        idx = 0
        for count in face_counts:
            if count == 3:
                triangles.append([
                    int(face_indices[idx]),
                    int(face_indices[idx + 1]),
                    int(face_indices[idx + 2]),
                ])
            elif count > 3:
                for i in range(1, int(count) - 1):
                    triangles.append([
                        int(face_indices[idx]),
                        int(face_indices[idx + i]),
                        int(face_indices[idx + i + 1]),
                    ])
            idx += int(count)
        return triangles


# Backward-compatible alias for existing imports.
GLBToUSDConverter = ModelToUSDConverter


class OmniverseKitIntegration:
    """
    Integration with Omniverse Kit for advanced operations.
    Requires running inside Kit environment or connecting to Kit server.
    """

    def __init__(self, config: Optional[OmniverseConfig] = None):
        self.config = config or OmniverseConfig()
        self._kit_available = False
        self._check_kit()

    def _check_kit(self):
        """Check if running inside Omniverse Kit"""
        try:
            import omni.usd
            import omni.kit.app
            self._kit_available = True
            logger.info("Running inside Omniverse Kit environment")
        except ImportError:
            self._kit_available = False
            logger.info("Not running inside Omniverse Kit - using standalone USD")

    @property
    def is_kit_available(self) -> bool:
        return self._kit_available

    async def convert_with_kit(self, input_path: Path, output_path: Path) -> Path:
        """
        Convert using Omniverse Kit Asset Converter.
        Only available when running inside Kit.
        """
        if not self._kit_available:
            raise RuntimeError("Omniverse Kit not available")

        import omni.kit.asset_converter

        task_manager = omni.kit.asset_converter.get_instance()

        # Configure conversion
        context = omni.kit.asset_converter.AssetConverterContext()
        context.ignore_materials = False
        context.ignore_animations = True
        context.single_mesh = False
        context.merge_all_meshes = False
        context.smooth_normals = True
        context.embed_textures = True

        # Create and run task
        task = task_manager.create_converter_task(
            str(input_path),
            str(output_path),
            progress_callback=lambda curr, total: logger.debug(f"Conversion progress: {curr}/{total}"),
            asset_converter_context=context
        )

        success = await task.wait_until_finished()

        if not success:
            raise RuntimeError(f"Kit asset conversion failed: {input_path}")

        logger.info(f"Kit conversion complete: {output_path}")
        return output_path

    def get_stage(self) -> Optional["Usd.Stage"]:
        """Get current Kit stage if available"""
        if not self._kit_available:
            return None

        import omni.usd
        context = omni.usd.get_context()
        return context.get_stage()
