"""
Universal 3D Product Configurator - Configuration Settings
"""
import os
import re
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, List, Any
from pathlib import Path
from enum import Enum

# Load .env file at module import time
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load from .env in current directory or parent directories
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars


class SegmentationMethod(str, Enum):
    """Mesh segmentation methods"""
    GEOMETRIC = "geometric"  # Based on geometry analysis
    AI_ASSISTED = "ai_assisted"  # Using AI/NIM models
    HYBRID = "hybrid"  # Combination of both
    SAM3_ZEROSHOT = "sam3_zeroshot"  # SAM3 with text prompts (BEST - default)


class MeshRepairMode(str, Enum):
    """Mesh repair aggressiveness modes"""
    MINIMAL = "minimal"        # Only fix critical issues, preserve all details
    CONSERVATIVE = "conservative"  # Light repair, preserve most details (default)
    AGGRESSIVE = "aggressive"  # Full repair for heavily damaged meshes


class MeshRepairConfig(BaseModel):
    """Configuration for mesh repair operations"""
    # Repair mode - controls overall aggressiveness
    mode: MeshRepairMode = MeshRepairMode.CONSERVATIVE

    # Individual controls (defaults based on CONSERVATIVE mode)
    fix_non_manifold: bool = True
    fill_holes: bool = False  # DISABLED by default - preserves lacing holes, vents, etc.
    max_hole_size: int = 10   # Reduced from 100 - only fill tiny holes
    remove_duplicate_faces: bool = True
    remove_degenerate_faces: bool = True
    smooth_surface: bool = False  # DISABLED by default - preserves fine details
    smooth_iterations: int = 1    # Reduced from 2
    fix_normals: bool = True
    merge_close_vertices: bool = True
    merge_threshold: float = 0.00001  # Reduced from 0.0001 - more conservative

    # New options for detail preservation
    preserve_uv_coords: bool = True   # Keep UV coordinates intact
    preserve_vertex_colors: bool = True  # Keep vertex colors
    skip_if_clean: bool = True  # Skip repair if mesh appears clean

    def apply_mode_defaults(self):
        """Apply defaults based on selected mode"""
        if self.mode == MeshRepairMode.MINIMAL:
            self.fill_holes = False
            self.smooth_surface = False
            self.fix_non_manifold = False  # Even skip this in minimal
            self.merge_close_vertices = False
        elif self.mode == MeshRepairMode.CONSERVATIVE:
            self.fill_holes = False
            self.smooth_surface = False
            self.max_hole_size = 10
        elif self.mode == MeshRepairMode.AGGRESSIVE:
            self.fill_holes = True
            self.smooth_surface = True
            self.max_hole_size = 100
            self.smooth_iterations = 2


class SegmentationConfig(BaseModel):
    """Configuration for mesh segmentation"""
    method: SegmentationMethod = SegmentationMethod.SAM3_ZEROSHOT  # SAM3 is best default
    min_segment_faces: int = 50  # Minimum faces per segment
    curvature_threshold: float = 0.3  # For geometric segmentation
    angle_threshold: float = 45.0  # Degrees for edge detection
    smoothing_iterations: int = 3
    use_connected_components: bool = True
    merge_small_segments: bool = True
    target_num_segments: Optional[int] = None  # If specified, targets this count

    # Enhanced segmentation options (v2.0)
    use_soft_scoring: bool = True  # Probabilistic face assignment
    enable_boundary_refinement: bool = True  # Graph-cut boundary smoothing
    enable_overlap_resolution: bool = True  # Multi-pass ambiguity resolution
    enable_thin_element_detection: bool = True  # Topology-based thin parts
    enable_curvature_boundaries: bool = True  # Curvature-based edge detection

    # Soft scoring parameters
    height_sigma_factor: float = 1.5  # Gaussian falloff width for height
    normal_weight: float = 1.0  # Weight for normal direction scoring
    position_weight: float = 0.8  # Weight for position scoring
    size_weight: float = 0.5  # Weight for size constraints

    # Boundary refinement parameters
    boundary_smoothing_iterations: int = 3
    edge_weight_decay: float = 0.5  # Dihedral angle sensitivity
    low_confidence_threshold: float = 0.3  # Below this = ambiguous

    # Thin element detection parameters
    thin_aspect_ratio_threshold: float = 5.0
    thin_min_faces: int = 20

    # Advanced options
    use_uv_seam_hints: bool = True  # Use UV seams as boundary hints
    use_convex_decomposition: bool = False  # PhysX-based decomposition


class MaterialRAGConfig(BaseModel):
    """Configuration for material retrieval and curation."""
    material_roots: List[Path] = Field(default_factory=list)
    index_dir: Path = Field(default=Path("material_rag"))
    top_k: Optional[int] = None
    candidate_pool_size: Optional[int] = None
    similarity_threshold: Optional[float] = None
    roughness_tolerance: Optional[float] = None
    metallic_tolerance: Optional[float] = None
    opacity_tolerance: Optional[float] = None
    allowlist_strict: Optional[bool] = None
    allowlist_policy: Optional[str] = None  # off | soft | strict
    use_product_name_in_query: Optional[bool] = None
    nim_rerank_enabled: Optional[bool] = None
    nim_rerank_temperature: Optional[float] = None
    nim_rerank_max_tokens: Optional[int] = None

    @field_validator("material_roots", mode="before")
    @classmethod
    def _parse_material_roots(cls, value: Any) -> List[Path]:
        if value is None:
            return []
        if isinstance(value, list):
            return [Path(item) for item in value if str(item).strip()]
        if isinstance(value, str):
            parts = []
            for token in re.split(r"[;,]", value):
                token = token.strip()
                if token:
                    parts.append(Path(token))
            return parts
        return [Path(value)]


class NIMConfig(BaseModel):
    """Configuration for NVIDIA NIM services"""
    profile: str = os.getenv("PRODCONFIG_NIM__PROFILE", "local")  # cloud | local | custom
    auth_mode: str = os.getenv("PRODCONFIG_NIM__AUTH_MODE", "none")  # auto | required | none
    api_key: Optional[str] = None
    base_url: str = os.getenv(
        "PRODCONFIG_NIM__BASE_URL",
        f"http://localhost:{os.getenv('PRODCONFIG_NIM__PORT', '19002')}/v1",
    )
    model: str = os.getenv("PRODCONFIG_NIM__MODEL", "meta/llama-3.1-70b-instruct")
    usd_code_endpoint: str = "/usd-code"
    mesh_segmentation_endpoint: str = "/mesh-segmentation"
    timeout: int = 120
    max_concurrency: int = 3
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    @field_validator("profile", mode="before")
    @classmethod
    def _normalize_profile(cls, value: Any) -> str:
        if value is None:
            return "cloud"
        profile = str(value).strip().lower()
        if profile in {"cloud", "local", "custom"}:
            return profile
        return "cloud"

    @field_validator("auth_mode", mode="before")
    @classmethod
    def _normalize_auth_mode(cls, value: Any) -> str:
        if value is None:
            return "auto"
        mode = str(value).strip().lower()
        if mode in {"auto", "required", "none"}:
            return mode
        return "auto"

    @field_validator("max_concurrency", mode="before")
    @classmethod
    def _normalize_max_concurrency(cls, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 3
        return max(1, parsed)

    @field_validator("max_retries", mode="before")
    @classmethod
    def _normalize_max_retries(cls, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 3
        return max(0, parsed)

    @field_validator("retry_backoff_seconds", mode="before")
    @classmethod
    def _normalize_retry_backoff(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, parsed)

    def __init__(self, **data):
        super().__init__(**data)
        self.base_url = self.resolve_base_url(self.profile, self.base_url)

        # Canonical key source for all NIM requests.
        if self.api_key is None:
            self.api_key = os.getenv("PRODCONFIG_NIM__API_KEY")

    @staticmethod
    def resolve_base_url(profile: str, base_url: str) -> str:
        if str(profile).strip().lower() == "local" and base_url == "https://integrate.api.nvidia.com/v1":
            return f"http://localhost:{os.getenv('PRODCONFIG_NIM__PORT', '19002')}/v1"
        return base_url

    def should_send_auth_header(self) -> bool:
        mode = str(self.auth_mode).strip().lower()
        if mode == "none":
            return False
        if mode == "required":
            return True
        return bool(self.api_key)

    def is_configured_for_inference(self) -> bool:
        mode = str(self.auth_mode).strip().lower()
        if mode == "required":
            return bool(self.api_key)
        return True

    def apply_overrides(
        self,
        base_url: Optional[str] = None,
        profile: Optional[str] = None,
        auth_mode: Optional[str] = None,
        max_concurrency: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff_seconds: Optional[float] = None,
    ) -> None:
        if profile is not None:
            self.profile = self._normalize_profile(profile)
        if auth_mode is not None:
            self.auth_mode = self._normalize_auth_mode(auth_mode)
        if max_concurrency is not None:
            self.max_concurrency = self._normalize_max_concurrency(max_concurrency)
        if max_retries is not None:
            self.max_retries = self._normalize_max_retries(max_retries)
        if retry_backoff_seconds is not None:
            self.retry_backoff_seconds = self._normalize_retry_backoff(retry_backoff_seconds)

        if base_url is not None:
            self.base_url = str(base_url).strip()
        else:
            self.base_url = self.resolve_base_url(self.profile, self.base_url)


class OmniverseConfig(BaseModel):
    """Configuration for Omniverse integration"""
    nucleus_server: Optional[str] = None
    local_cache_path: str = "./cache/omniverse"
    use_kit_app: bool = False  # If True, uses Kit app; else standalone USD
    kit_app_path: Optional[str] = None


class AppConfig(BaseSettings):
    """Main application configuration"""
    # Paths
    input_path: Path = Field(default=Path("./input"))
    output_path: Path = Field(default=Path("./output"))
    temp_path: Path = Field(default=Path("./temp"))

    # Product configuration
    custom_mesh_parts: Optional[List[str]] = None

    # Processing configs
    mesh_repair: MeshRepairConfig = Field(default_factory=MeshRepairConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    rag: MaterialRAGConfig = Field(default_factory=MaterialRAGConfig)
    mdl_roots: Optional[str] = None

    # Integration configs
    nim: NIMConfig = Field(default_factory=NIMConfig)
    omniverse: OmniverseConfig = Field(default_factory=OmniverseConfig)

    # Processing options
    parallel_processing: bool = True
    max_workers: int = 4
    verbose: bool = True

    model_config = SettingsConfigDict(
        env_prefix="PRODCONFIG_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore"
    )

    @field_validator("mdl_roots", mode="before")
    @classmethod
    def _normalize_mdl_roots(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            cleaned = [str(item).strip().strip('"').strip("'") for item in value if str(item).strip()]
            return ";".join(cleaned) if cleaned else None
        normalized = str(value).strip()
        return normalized or None

    def ensure_directories(self):
        """Create required directories if they don't exist"""
        for path in [self.input_path, self.output_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = AppConfig()
