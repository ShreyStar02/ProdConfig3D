"""
Mesh Processing Module - Handles loading, repairing, and manipulating 3D meshes
Supports AI-generated models with potential imperfections.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from ..config import MeshRepairConfig

logger = logging.getLogger(__name__)


@dataclass
class MeshData:
    """Container for mesh data with metadata"""
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    vertex_colors: Optional[np.ndarray] = None
    uv_coords: Optional[np.ndarray] = None
    face_normals: Optional[np.ndarray] = None
    name: str = "mesh"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def to_trimesh(self) -> "trimesh.Trimesh":
        """Convert to trimesh object, preserving all visual properties"""
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for this operation")

        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            process=False  # Critical: don't alter the mesh
        )

        # Preserve vertex colors if available
        if self.vertex_colors is not None:
            mesh.visual.vertex_colors = self.vertex_colors

        # Store UV coords in metadata for preservation (trimesh doesn't have native UV support)
        if self.uv_coords is not None:
            mesh.metadata['uv_coords'] = self.uv_coords

        return mesh

    @classmethod
    def from_trimesh(cls, mesh: "trimesh.Trimesh", name: str = "mesh",
                     original_data: "MeshData" = None,
                     vertex_map: np.ndarray = None) -> "MeshData":
        """
        Create from trimesh object, preserving visual properties.

        Args:
            mesh: Source trimesh
            name: Mesh name
            original_data: Original MeshData to pull UV/color data from
            vertex_map: Mapping from new vertices to original vertices (for submesh extraction)
        """
        # Extract vertex colors from trimesh visual
        vertex_colors = None
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            try:
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) == len(mesh.vertices):
                    vertex_colors = np.array(vc)
            except Exception:
                pass

        # Check metadata for UV coords
        uv_coords = mesh.metadata.get('uv_coords') if mesh.metadata else None

        # If we have original data and vertex map, try to reconstruct attributes
        if original_data is not None and vertex_map is not None:
            if original_data.vertex_colors is not None and vertex_colors is None:
                try:
                    vertex_colors = original_data.vertex_colors[vertex_map]
                except (IndexError, ValueError):
                    pass
            if original_data.uv_coords is not None and uv_coords is None:
                try:
                    uv_coords = original_data.uv_coords[vertex_map]
                except (IndexError, ValueError):
                    pass

        return cls(
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
            normals=np.array(mesh.vertex_normals) if mesh.vertex_normals is not None else None,
            face_normals=np.array(mesh.face_normals) if mesh.face_normals is not None else None,
            vertex_colors=vertex_colors,
            uv_coords=uv_coords,
            name=name
        )


class MeshLoader:
    """Handles loading of various 3D mesh formats"""

    SUPPORTED_FORMATS = {'.glb', '.gltf', '.obj', '.fbx', '.stl', '.ply', '.off', '.usd', '.usda', '.usdc'}

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        if not TRIMESH_AVAILABLE:
            logger.warning("trimesh not available - some features may be limited")

    def load(self, file_path: Path) -> Tuple[List[MeshData], Dict[str, Any]]:
        """
        Load a 3D model file and return mesh data.

        Args:
            file_path: Path to the 3D model file

        Returns:
            Tuple of (list of MeshData objects, scene metadata dict)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}. Supported: {self.SUPPORTED_FORMATS}")

        logger.info(f"Loading mesh from: {file_path}")

        if TRIMESH_AVAILABLE:
            return self._load_with_trimesh(file_path)
        else:
            raise ImportError("No mesh loading library available. Install trimesh.")

    def _load_with_trimesh(self, file_path: Path) -> Tuple[List[MeshData], Dict[str, Any]]:
        """Load using trimesh library, preserving all visual data"""
        scene = trimesh.load(str(file_path), force='scene', process=False)
        meshes = []
        metadata = {
            'file_path': str(file_path),
            'format': file_path.suffix,
            'units': getattr(scene, 'units', None),
        }

        if isinstance(scene, trimesh.Scene):
            metadata['is_scene'] = True
            metadata['geometry_names'] = list(scene.geometry.keys())
            hierarchy = self._extract_scene_hierarchy(scene)
            metadata['mesh_hierarchy'] = hierarchy

            by_geometry: Dict[str, List[Dict[str, Any]]] = {}
            for entry in hierarchy:
                by_geometry.setdefault(entry.get('geometry_name', ''), []).append(entry)

            for name, geometry in scene.geometry.items():
                if isinstance(geometry, trimesh.Trimesh):
                    hierarchy_entry = self._select_hierarchy_entry(by_geometry.get(name, []))
                    mesh_name = hierarchy_entry.get('node_name') if hierarchy_entry else name
                    mesh_data = self._extract_mesh_data(geometry, mesh_name)
                    if hierarchy_entry:
                        mesh_data.metadata.update({
                            'scene_node': hierarchy_entry.get('node_name'),
                            'scene_path': hierarchy_entry.get('path'),
                            'mesh_ancestor': hierarchy_entry.get('first_mesh_ancestor') or hierarchy_entry.get('node_name'),
                            'is_leaf_mesh': hierarchy_entry.get('is_leaf_mesh', True),
                        })
                    meshes.append(mesh_data)
                    visual_info = ""
                    if mesh_data.vertex_colors is not None:
                        visual_info += f", {len(mesh_data.vertex_colors)} colors"
                    if mesh_data.uv_coords is not None:
                        visual_info += f", has UVs"
                    logger.info(f"  Loaded mesh '{name}': {mesh_data.num_vertices} vertices, "
                               f"{mesh_data.num_faces} faces{visual_info}")
        elif isinstance(scene, trimesh.Trimesh):
            metadata['is_scene'] = False
            mesh_data = self._extract_mesh_data(scene, file_path.stem)
            meshes.append(mesh_data)
            logger.info(f"  Loaded single mesh: {mesh_data.num_vertices} vertices, {mesh_data.num_faces} faces")

        return meshes, metadata

    def _select_hierarchy_entry(self, entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not entries:
            return None
        for entry in entries:
            if entry.get('is_leaf_mesh'):
                return entry
        return entries[0]

    def _extract_scene_hierarchy(self, scene: "trimesh.Scene") -> List[Dict[str, Any]]:
        graph = scene.graph
        edges = graph.to_edgelist()

        children: Dict[str, List[str]] = {}
        parents: Dict[str, str] = {}
        for edge in edges:
            parent = str(edge[0])
            child = str(edge[1])
            children.setdefault(parent, []).append(child)
            parents.setdefault(child, parent)

        mesh_nodes = set(getattr(graph, 'nodes_geometry', []) or [])

        def _is_leaf_mesh(node: str) -> bool:
            stack = list(children.get(node, []))
            while stack:
                current = stack.pop()
                if current in mesh_nodes:
                    return False
                stack.extend(children.get(current, []))
            return True

        def _node_chain(node: str) -> List[str]:
            chain = [node]
            current = node
            while current in parents:
                current = parents[current]
                chain.append(current)
            chain.reverse()
            return chain

        entries: List[Dict[str, Any]] = []
        for node in mesh_nodes:
            _, geometry_name = graph.get(node)
            chain = _node_chain(str(node))
            path_tokens = []
            first_mesh_ancestor = None
            for chain_node in chain:
                node_type = "Mesh" if chain_node in mesh_nodes else "Xform"
                path_tokens.append(f"{chain_node}({node_type})")
                if node_type == "Mesh" and first_mesh_ancestor is None:
                    first_mesh_ancestor = chain_node

            entries.append({
                'node_name': str(node),
                'geometry_name': str(geometry_name) if geometry_name is not None else str(node),
                'path': " -> ".join(path_tokens),
                'is_leaf_mesh': _is_leaf_mesh(str(node)),
                'first_mesh_ancestor': first_mesh_ancestor,
            })

        return entries

    def _extract_mesh_data(self, mesh: "trimesh.Trimesh", name: str) -> MeshData:
        """Extract all mesh data including visual properties from trimesh"""
        vertex_colors = None
        uv_coords = None

        # Extract vertex colors
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            visual = mesh.visual

            # Handle different visual types
            if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None:
                try:
                    vc = np.array(visual.vertex_colors)
                    if len(vc) == len(mesh.vertices):
                        vertex_colors = vc
                        logger.debug(f"  Extracted {len(vc)} vertex colors")
                except Exception as e:
                    logger.debug(f"  Could not extract vertex colors: {e}")

            # Try to extract UV coordinates
            if hasattr(visual, 'uv') and visual.uv is not None:
                try:
                    uv_coords = np.array(visual.uv)
                    logger.debug(f"  Extracted UV coords: {uv_coords.shape}")
                except Exception as e:
                    logger.debug(f"  Could not extract UVs: {e}")

            # For TextureVisuals, try to get UVs differently
            if uv_coords is None and hasattr(visual, 'to_texture'):
                try:
                    tex_visual = visual.to_texture()
                    if hasattr(tex_visual, 'uv') and tex_visual.uv is not None:
                        uv_coords = np.array(tex_visual.uv)
                        logger.debug(f"  Extracted texture UVs: {uv_coords.shape}")
                except Exception as e:
                    logger.debug(f"  Could not extract texture UVs: {e}")

        return MeshData(
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
            normals=np.array(mesh.vertex_normals) if mesh.vertex_normals is not None else None,
            face_normals=np.array(mesh.face_normals) if mesh.face_normals is not None else None,
            vertex_colors=vertex_colors,
            uv_coords=uv_coords,
            name=name
        )


class MeshRepairer:
    """
    Handles mesh repair operations for AI-generated models.
    Fixes common issues like non-manifold edges, holes, degenerate faces, etc.
    """

    def __init__(self, config: Optional[MeshRepairConfig] = None):
        self.config = config or MeshRepairConfig()
        self._check_dependencies()

    def _check_dependencies(self):
        if not PYMESHLAB_AVAILABLE and not TRIMESH_AVAILABLE:
            logger.warning("Neither pymeshlab nor trimesh available - repair capabilities limited")

    def repair(self, mesh_data: MeshData) -> MeshData:
        """
        Apply mesh repair operations with detail preservation.

        Args:
            mesh_data: Input mesh data

        Returns:
            Repaired mesh data (or original if no repair needed)
        """
        logger.info(f"Starting mesh repair for '{mesh_data.name}' (mode: {self.config.mode.value})")
        logger.info(f"  Initial: {mesh_data.num_vertices} vertices, {mesh_data.num_faces} faces")

        # Check if mesh appears clean and skip_if_clean is enabled
        if self.config.skip_if_clean:
            if self._is_mesh_clean(mesh_data):
                logger.info("  Mesh appears clean, skipping repair to preserve details")
                return mesh_data

        # Apply mode defaults if needed
        self.config.apply_mode_defaults()

        # For MINIMAL mode, just return original mesh with minor fixes
        if self.config.mode.value == "minimal":
            logger.info("  MINIMAL mode: Preserving original mesh with minimal fixes")
            if TRIMESH_AVAILABLE:
                result = self._minimal_repair(mesh_data)
            else:
                return mesh_data
        # Try PyMeshLab first (more comprehensive), fallback to trimesh
        elif PYMESHLAB_AVAILABLE:
            result = self._repair_with_pymeshlab(mesh_data)
        elif TRIMESH_AVAILABLE:
            result = self._repair_with_trimesh(mesh_data)
        else:
            logger.warning("No repair library available, returning original mesh")
            return mesh_data

        logger.info(f"  After repair: {result.num_vertices} vertices, {result.num_faces} faces")
        return result

    def _is_mesh_clean(self, mesh_data: MeshData) -> bool:
        """Check if mesh appears clean and doesn't need repair"""
        if not TRIMESH_AVAILABLE:
            return False

        mesh = mesh_data.to_trimesh()

        # For meshes with reasonable complexity, assume they're production-ready
        # GLB files from modeling software are typically already clean
        if mesh_data.num_faces > 500 and mesh_data.num_vertices > 250:
            logger.info(f"  Mesh has {mesh_data.num_faces} faces - assuming production-ready, skipping repair")
            return True

        # Check for obvious issues
        has_degenerate = len(mesh.faces) != len(np.unique(mesh.faces, axis=0))
        is_watertight = mesh.is_watertight

        # If watertight and no degenerate faces, probably clean
        if is_watertight and not has_degenerate:
            return True

        return False

    def _minimal_repair(self, mesh_data: MeshData) -> MeshData:
        """Minimal repair - only fix critical issues, preserve everything else"""
        mesh = mesh_data.to_trimesh()

        # Only remove truly degenerate faces (zero area)
        if self.config.remove_degenerate_faces:
            mesh.remove_degenerate_faces()

        # Fix normals if needed (non-destructive)
        if self.config.fix_normals:
            mesh.fix_normals()

        result = MeshData.from_trimesh(mesh, name=mesh_data.name)

        # Preserve original UV coords and colors
        if self.config.preserve_uv_coords and mesh_data.uv_coords is not None:
            result.uv_coords = mesh_data.uv_coords
        if self.config.preserve_vertex_colors and mesh_data.vertex_colors is not None:
            result.vertex_colors = mesh_data.vertex_colors

        result.metadata = {**mesh_data.metadata, 'repaired': True, 'repair_method': 'minimal'}
        return result

    def _repair_with_pymeshlab(self, mesh_data: MeshData) -> MeshData:
        """Repair using PyMeshLab with detail preservation"""
        ms = pymeshlab.MeshSet()

        # Create mesh from numpy arrays
        m = pymeshlab.Mesh(
            vertex_matrix=mesh_data.vertices,
            face_matrix=mesh_data.faces
        )
        ms.add_mesh(m)

        # Only merge vertices if enabled and threshold is very small
        if self.config.merge_close_vertices:
            logger.debug("  Merging close vertices (conservative)...")
            ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(self.config.merge_threshold * 100))

        # Remove duplicate faces
        if self.config.remove_duplicate_faces:
            logger.debug("  Removing duplicate faces...")
            ms.meshing_remove_duplicate_faces()

        # Remove degenerate faces (zero area)
        if self.config.remove_degenerate_faces:
            logger.debug("  Removing degenerate faces...")
            ms.meshing_remove_null_faces()

        # Fix non-manifold edges (only if enabled)
        if self.config.fix_non_manifold:
            logger.debug("  Fixing non-manifold edges...")
            try:
                ms.meshing_repair_non_manifold_edges()
                ms.meshing_repair_non_manifold_vertices()
            except Exception as e:
                logger.warning(f"  Non-manifold repair warning: {e}")

        # Fill holes ONLY if explicitly enabled and only small holes
        if self.config.fill_holes:
            logger.debug(f"  Filling holes (max size: {self.config.max_hole_size})...")
            try:
                ms.meshing_close_holes(maxholesize=self.config.max_hole_size)
            except Exception as e:
                logger.warning(f"  Hole filling warning: {e}")

        # Fix normals (non-destructive operation)
        if self.config.fix_normals:
            logger.debug("  Fixing normals...")
            ms.meshing_re_orient_faces_coherently()
            ms.compute_normal_per_vertex()
            ms.compute_normal_per_face()

        # Smooth surface ONLY if explicitly enabled (DESTRUCTIVE - avoid by default)
        if self.config.smooth_surface:
            logger.warning("  WARNING: Applying smoothing - this may destroy fine details!")
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=self.config.smooth_iterations)

        # Get repaired mesh
        repaired_mesh = ms.current_mesh()

        # Try to get normals if available
        try:
            normals = repaired_mesh.vertex_normal_matrix()
        except Exception:
            normals = None

        result = MeshData(
            vertices=repaired_mesh.vertex_matrix(),
            faces=repaired_mesh.face_matrix(),
            normals=normals,
            name=mesh_data.name,
            metadata={**mesh_data.metadata, 'repaired': True, 'repair_method': 'pymeshlab'}
        )

        # Try to preserve UV coords if unchanged vertex count
        if self.config.preserve_uv_coords and mesh_data.uv_coords is not None:
            if result.num_vertices == mesh_data.num_vertices:
                result.uv_coords = mesh_data.uv_coords
                logger.debug("  Preserved UV coordinates")
            else:
                logger.warning(f"  UV coords lost (vertex count changed: {mesh_data.num_vertices} -> {result.num_vertices})")

        # Try to preserve vertex colors
        if self.config.preserve_vertex_colors and mesh_data.vertex_colors is not None:
            if result.num_vertices == mesh_data.num_vertices:
                result.vertex_colors = mesh_data.vertex_colors
                logger.debug("  Preserved vertex colors")

        return result

    def _repair_with_trimesh(self, mesh_data: MeshData) -> MeshData:
        """Basic repair using trimesh"""
        mesh = mesh_data.to_trimesh()

        # Merge duplicate vertices
        if self.config.merge_close_vertices:
            mesh.merge_vertices()

        # Remove duplicate faces
        if self.config.remove_duplicate_faces:
            mesh.remove_duplicate_faces()

        # Remove degenerate faces
        if self.config.remove_degenerate_faces:
            mesh.remove_degenerate_faces()

        # Fill holes (trimesh has limited hole filling)
        if self.config.fill_holes:
            try:
                trimesh.repair.fill_holes(mesh)
            except Exception as e:
                logger.warning(f"  Hole filling warning: {e}")

        # Fix normals
        if self.config.fix_normals:
            mesh.fix_normals()

        return MeshData.from_trimesh(mesh, name=mesh_data.name)


class MeshAnalyzer:
    """Analyzes mesh properties for segmentation preparation"""

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for mesh analysis")

    def analyze(self, mesh_data: MeshData) -> Dict[str, Any]:
        """
        Analyze mesh properties.

        Returns dict with:
            - bounds: (min, max) bounding box
            - center: center of mass
            - volume: mesh volume (if watertight)
            - surface_area: total surface area
            - is_watertight: whether mesh is closed
            - euler_number: topological euler number
            - connected_components: number of separate parts
            - curvature_stats: curvature statistics
        """
        mesh = mesh_data.to_trimesh()
        analysis = {}

        # Basic properties
        analysis['bounds'] = mesh.bounds.tolist()
        analysis['center'] = mesh.centroid.tolist()
        analysis['surface_area'] = float(mesh.area)
        analysis['is_watertight'] = bool(mesh.is_watertight)

        if mesh.is_watertight:
            analysis['volume'] = float(mesh.volume)
        else:
            analysis['volume'] = None

        # Topology
        analysis['euler_number'] = int(mesh.euler_number)

        # Connected components
        try:
            components = mesh.split(only_watertight=False)
            analysis['connected_components'] = len(components)
        except Exception:
            analysis['connected_components'] = 1

        # Curvature analysis (useful for segmentation)
        try:
            # Compute discrete gaussian curvature
            curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
                mesh, mesh.vertices, radius=mesh.scale / 100
            )
            analysis['curvature_stats'] = {
                'mean': float(np.mean(curvature)),
                'std': float(np.std(curvature)),
                'min': float(np.min(curvature)),
                'max': float(np.max(curvature))
            }
        except Exception as e:
            logger.warning(f"Curvature analysis failed: {e}")
            analysis['curvature_stats'] = None

        # Edge analysis
        edges = mesh.edges_unique
        edge_lengths = mesh.edges_unique_length
        analysis['edge_stats'] = {
            'count': len(edges),
            'mean_length': float(np.mean(edge_lengths)),
            'std_length': float(np.std(edge_lengths)),
            'min_length': float(np.min(edge_lengths)),
            'max_length': float(np.max(edge_lengths))
        }

        return analysis

    def compute_face_features(self, mesh_data: MeshData) -> np.ndarray:
        """
        Compute per-face features for segmentation.

        Returns array of shape (num_faces, num_features) with:
            - face normal (3)
            - face center (3)
            - face area (1)
            - mean curvature at face (1)
        """
        mesh = mesh_data.to_trimesh()

        # Face normals
        face_normals = mesh.face_normals  # (N, 3)

        # Face centers
        face_centers = mesh.triangles_center  # (N, 3)

        # Face areas
        face_areas = mesh.area_faces.reshape(-1, 1)  # (N, 1)

        # Compute vertex curvature and average to faces
        try:
            vertex_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
                mesh, mesh.vertices, radius=mesh.scale / 50
            )
            # Average curvature of face vertices
            face_curvature = np.mean(vertex_curvature[mesh.faces], axis=1, keepdims=True)
        except Exception:
            face_curvature = np.zeros((len(mesh.faces), 1))

        # Combine features
        features = np.hstack([
            face_normals,      # 3
            face_centers,      # 3
            face_areas,        # 1
            face_curvature     # 1
        ])

        return features


def merge_meshes(meshes: List[MeshData]) -> MeshData:
    """Merge multiple meshes into a single mesh"""
    if not meshes:
        raise ValueError("No meshes to merge")

    if len(meshes) == 1:
        return meshes[0]

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for mesh in meshes:
        all_vertices.append(mesh.vertices)
        all_faces.append(mesh.faces + vertex_offset)
        vertex_offset += mesh.num_vertices

    return MeshData(
        vertices=np.vstack(all_vertices),
        faces=np.vstack(all_faces),
        name="merged_mesh"
    )
