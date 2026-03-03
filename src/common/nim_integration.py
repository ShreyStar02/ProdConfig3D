"""
NVIDIA NIM Integration - Connects to NVIDIA Inference Microservices for AI-assisted operations.
Supports mesh segmentation, intelligent part labeling, and material recommendations.
"""
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import base64
import numpy as np

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ..config import NIMConfig

logger = logging.getLogger(__name__)

CLOUD_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass
class NIMResponse:
    """Response from NIM service"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


@dataclass
class SegmentAnalysis:
    """Analysis result for a mesh segment"""
    segment_id: int
    original_label: str
    ai_label: str
    part_type: str
    confidence: float
    material_suggestions: List[str]
    properties: Dict[str, Any]


class NIMClient:
    """
    Client for NVIDIA Inference Microservices (NIMs).
    Provides access to AI models for 3D processing.
    """

    # Available NIM endpoints
    ENDPOINTS = {
        "usd_code": "/nvidia/usd-code",
        "usd_search": "/nvidia/usd-search",
        "chat": "/chat/completions",
        "embeddings": "/embeddings",
    }

    def __init__(self, config: Optional[NIMConfig] = None):
        self.config = config or NIMConfig()
        self._check_dependencies()
        self._client = None
        self._client_lock = asyncio.Lock()

    def _check_dependencies(self):
        if not HTTPX_AVAILABLE and not AIOHTTP_AVAILABLE:
            raise RuntimeError("ERROR: This is not supported/compatible - neither httpx nor aiohttp is installed.")

    @property
    def is_configured(self) -> bool:
        """Check if NIM client can execute requests with current auth mode."""
        if not HTTPX_AVAILABLE:
            return False
        return self.config.is_configured_for_inference()

    def _should_send_auth_header(self) -> bool:
        """Resolve auth behavior for cloud/local endpoints."""
        return self.config.should_send_auth_header()

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self._should_send_auth_header() and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    @staticmethod
    def _looks_like_local_nim(url: str) -> bool:
        normalized = str(url).strip().lower()
        return normalized.startswith("http://localhost:") or normalized.startswith("http://127.0.0.1:")

    def _endpoint_mode(self) -> str:
        """Human-readable endpoint mode for diagnostics."""
        base = str(self.config.base_url).strip().lower()
        if self._looks_like_local_nim(base):
            return "local"
        if base.startswith(CLOUD_NIM_BASE_URL.lower()):
            return "cloud"
        profile = str(self.config.profile).strip().lower()
        return profile or "custom"

    async def _activate_cloud_fallback(self, reason: str) -> bool:
        """Switch client base URL from local NIM to cloud endpoint once."""
        async with self._client_lock:
            current_base = str(self.config.base_url).strip()
            if not self._looks_like_local_nim(current_base):
                return False

            if not self.config.api_key:
                logger.warning(
                    "Local NIM unavailable (%s), but cloud fallback is disabled because no API key is configured.",
                    reason,
                )
                return False

            prior_auth_mode = str(self.config.auth_mode).strip().lower()
            self.config.base_url = CLOUD_NIM_BASE_URL
            self.config.profile = "cloud"
            if prior_auth_mode == "none":
                # Cloud NIM requires auth; auto mode preserves required behavior without forcing strict mode.
                self.config.auth_mode = "auto"

            if self._client:
                await self._client.aclose()
                self._client = None

        logger.warning(
            "Local NIM unavailable (%s). Falling back to cloud NIM endpoint: %s",
            reason,
            self.config.base_url,
        )
        return True

    async def _get_client(self):
        """Get or create HTTP client"""
        async with self._client_lock:
            if self._client is None and HTTPX_AVAILABLE:
                self._client = httpx.AsyncClient(
                    base_url=self.config.base_url,
                    headers=self._build_headers(),
                    timeout=self.config.timeout
                )
            return self._client

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> httpx.Response:
        """POST helper with retry/backoff for transient HTTP failures."""
        attempts = max(1, int(self.config.max_retries) + 1)
        backoff_base = max(0.0, float(self.config.retry_backoff_seconds))
        fallback_used = False

        for attempt in range(attempts):
            client = await self._get_client()
            request_mode = self._endpoint_mode()
            try:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                retryable = status == 429 or (status is not None and status >= 500)
                if not retryable or attempt >= attempts - 1:
                    raise
                delay = backoff_base * (2 ** attempt)
                logger.warning(
                    "NIM %s request failed (%s) at %s attempt %s/%s; retrying in %.2fs",
                    request_mode,
                    status,
                    path,
                    attempt + 1,
                    attempts,
                    delay,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                if "client has been closed" in str(exc).lower():
                    async with self._client_lock:
                        self._client = None

                switched = False
                if request_mode == "local" and not fallback_used:
                    switched = await self._activate_cloud_fallback(exc.__class__.__name__)
                    if switched:
                        fallback_used = True
                        logger.warning(
                            "NIM local request failed at %s attempt %s/%s (%s); retrying on cloud endpoint.",
                            path,
                            attempt + 1,
                            attempts,
                            exc.__class__.__name__,
                        )
                        continue

                current_mode = self._endpoint_mode()
                if request_mode == "local" and current_mode == "cloud":
                    # Another concurrent request already switched endpoint mode.
                    fallback_used = True
                    if attempt >= attempts - 1:
                        raise
                    delay = backoff_base * (2 ** attempt)
                    logger.warning(
                        "NIM local request error at %s attempt %s/%s (%s) while cloud fallback activated; retrying in %.2fs",
                        path,
                        attempt + 1,
                        attempts,
                        exc.__class__.__name__,
                        delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue

                if attempt >= attempts - 1:
                    raise
                delay = backoff_base * (2 ** attempt)
                logger.warning(
                    "NIM %s request error at %s attempt %s/%s (%s); retrying in %.2fs",
                    request_mode,
                    path,
                    attempt + 1,
                    attempts,
                    exc.__class__.__name__,
                    delay,
                )
                if delay > 0:
                    await asyncio.sleep(delay)

        raise RuntimeError(f"NIM {self._endpoint_mode()} request failed after {attempts} attempts: {path}")

    async def close(self):
        """Close HTTP client"""
        async with self._client_lock:
            if self._client:
                await self._client.aclose()
                self._client = None

    async def chat_completion(self, prompt: str,
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.3,
                              max_tokens: int = 2048) -> NIMResponse:
        """
        Generic chat completion request to the NIM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            temperature: Response temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            NIMResponse with the AI's response content
        """
        if not self.is_configured:
            return NIMResponse(success=False, error="NIM API key not configured")

        try:
            client = await self._get_client()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = await self._post_json("/chat/completions", payload)

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            return NIMResponse(
                success=True,
                data=content,
                usage=data.get("usage")
            )

        except httpx.HTTPStatusError as e:
            mode = self._endpoint_mode()
            logger.error("Chat completion NIM HTTP error (%s): %s", mode, e)
            return NIMResponse(success=False, error=f"{mode} NIM HTTP error: {e}")
        except Exception as e:
            mode = self._endpoint_mode()
            logger.error("Chat completion NIM error (%s): %s", mode, e)
            return NIMResponse(success=False, error=f"{mode} NIM error: {e}")

    async def generate_usd_code(self, prompt: str,
                                context: Optional[str] = None) -> NIMResponse:
        """
        Use USD Code NIM to generate USD Python code from natural language.

        Args:
            prompt: Natural language description of desired USD operation
            context: Optional existing USD stage content for context

        Returns:
            NIMResponse with generated code
        """
        if not self.is_configured:
            return NIMResponse(success=False, error="NIM API key not configured")

        try:
            client = await self._get_client()

            payload = {
                "model": "nvidia/usd-code",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a USD (Universal Scene Description) expert. Generate Python code using the pxr library."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 2048
            }

            if context:
                payload["messages"].insert(1, {
                    "role": "user",
                    "content": f"Current USD stage context:\n{context}"
                })

            response = await self._post_json("/chat/completions", payload)

            data = response.json()
            generated_code = data["choices"][0]["message"]["content"]

            return NIMResponse(
                success=True,
                data={"code": generated_code},
                usage=data.get("usage")
            )

        except Exception as e:
            mode = self._endpoint_mode()
            logger.error("USD Code NIM error (%s): %s", mode, e)
            return NIMResponse(success=False, error=f"{mode} NIM error: {e}")

    async def analyze_mesh_for_segmentation(self, mesh_description: str,
                                           product_type: str) -> NIMResponse:
        """
        Use AI to suggest mesh segmentation based on a product name hint.

        Args:
            mesh_description: Description of the mesh (vertices, faces, bounds)
            product_type: Product name hint derived from the filename

        Returns:
            NIMResponse with segmentation suggestions
        """
        if not self.is_configured:
            return NIMResponse(success=False, error="NIM API key not configured")

        try:
            client = await self._get_client()

            prompt = f"""Analyze this 3D mesh and suggest how to segment it into meaningful parts.

Product Name: {product_type}
Mesh Details: {mesh_description}

Provide a JSON response with:
1. suggested_parts: List of part names appropriate for this product
2. segmentation_hints: Geometric hints for identifying each part (e.g., "bottom faces", "top region")
3. expected_part_count: Estimated number of distinct parts

Format as valid JSON only."""

            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a 3D modeling expert specializing in product design and mesh analysis. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }

            response = await self._post_json("/chat/completions", payload)

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            try:
                # Find JSON in response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    suggestions = json.loads(content[json_start:json_end])
                else:
                    suggestions = {"raw_response": content}
            except json.JSONDecodeError:
                suggestions = {"raw_response": content}

            return NIMResponse(
                success=True,
                data=suggestions,
                usage=data.get("usage")
            )

        except Exception as e:
            mode = self._endpoint_mode()
            logger.error("Mesh analysis NIM error (%s): %s", mode, e)
            return NIMResponse(success=False, error=f"{mode} NIM error: {e}")

    async def suggest_materials(self, mesh_parts: List[str],
                               product_type: str) -> NIMResponse:
        """
        Get AI suggestions for appropriate materials for each mesh part.

        Args:
            mesh_parts: List of mesh part names
            product_type: Product name hint

        Returns:
            NIMResponse with material suggestions per part
        """
        if not self.is_configured:
            return NIMResponse(success=False, error="NIM API key not configured")

        try:
            client = await self._get_client()

            prompt = f"""Suggest appropriate materials for each part of this product ({product_type}).

Parts: {', '.join(mesh_parts)}

For each part, suggest:
1. Primary material type (leather, fabric, rubber, plastic, metal, glass)
2. Specific material variants (e.g., for leather: smooth, aged, patent, suede)
3. Typical PBR properties (roughness range, metallic value)

Format as JSON with structure:
{{
    "part_name": {{
        "primary_material": "type",
        "variants": ["var1", "var2"],
        "roughness_range": [min, max],
        "metallic": value
    }}
}}"""

            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a material science expert for product design. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2048
            }

            response = await self._post_json("/chat/completions", payload)

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                suggestions = json.loads(content[json_start:json_end])
            except json.JSONDecodeError:
                suggestions = {"raw_response": content}

            return NIMResponse(
                success=True,
                data=suggestions,
                usage=data.get("usage")
            )

        except Exception as e:
            mode = self._endpoint_mode()
            logger.error("Material suggestion NIM error (%s): %s", mode, e)
            return NIMResponse(success=False, error=f"{mode} NIM error: {e}")

    async def detect_product_type(self, mesh_info: Dict[str, Any]) -> NIMResponse:
        """
        Use AI to automatically detect what kind of product or object the mesh represents.

        Args:
            mesh_info: Dictionary with mesh properties (bounds, aspect ratio, etc.)

        Returns:
            NIMResponse with detected product name and confidence
        """
        if not self.is_configured:
            return NIMResponse(success=False, error="NIM API key not configured")

        try:
            client = await self._get_client()

            # Calculate mesh characteristics
            bounds = mesh_info.get('bounds', [[0, 0, 0], [1, 1, 1]])
            size = [bounds[1][i] - bounds[0][i] for i in range(3)]
            aspect_ratios = {
                'width_height': size[0] / max(size[1], 0.001),
                'width_depth': size[0] / max(size[2], 0.001),
                'height_depth': size[1] / max(size[2], 0.001)
            }

            prompt = f"""Analyze this 3D mesh and identify what type of product or object it represents.

Mesh Properties:
- Dimensions (W x H x D): {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}
- Aspect ratios: width/height={aspect_ratios['width_height']:.2f}, width/depth={aspect_ratios['width_depth']:.2f}
- Vertex count: {mesh_info.get('num_vertices', 'unknown')}
- Face count: {mesh_info.get('num_faces', 'unknown')}
- Is watertight: {mesh_info.get('is_watertight', 'unknown')}
- Connected components: {mesh_info.get('connected_components', 'unknown')}

Based on the shape and proportions, identify the most likely product name or object category.

Respond with ONLY valid JSON in this exact format:
{{
    "product_type": "detected product name or object category",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation",
    "expected_parts": ["list", "of", "expected", "component", "names"]
}}"""

            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a 3D product analysis expert. Identify products from their geometric properties. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 512
            }

            response = await self._post_json("/chat/completions", payload)

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(content[json_start:json_end])
                else:
                    result = {"product_type": "generic", "confidence": 0.5, "expected_parts": []}
            except json.JSONDecodeError:
                result = {"product_type": "generic", "confidence": 0.5, "expected_parts": []}

            return NIMResponse(success=True, data=result, usage=data.get("usage"))

        except Exception as e:
            mode = self._endpoint_mode()
            logger.error("Product detection NIM error (%s): %s", mode, e)
            return NIMResponse(success=False, error=f"{mode} NIM error: {e}")

    async def identify_segment_parts(self, segments_info: List[Dict[str, Any]],
                                     product_type: str,
                                     mesh_bounds: List[List[float]]) -> NIMResponse:
        """
        Use AI to identify what each mesh segment represents based on its geometric properties.

        Args:
            segments_info: List of segment properties (center, normal, area, bounds, etc.)
            product_type: The detected or specified product name
            mesh_bounds: Overall mesh bounding box

        Returns:
            NIMResponse with identified part names for each segment
        """
        if not self.is_configured:
            return NIMResponse(success=False, error="NIM API key not configured")

        try:
            client = await self._get_client()

            # Calculate relative positions
            mesh_min = np.array(mesh_bounds[0])
            mesh_max = np.array(mesh_bounds[1])
            mesh_size = mesh_max - mesh_min
            mesh_center = (mesh_min + mesh_max) / 2

            segments_desc = []
            for i, seg in enumerate(segments_info):
                center = np.array(seg.get('center', [0, 0, 0]))
                rel_pos = (center - mesh_min) / (mesh_size + 1e-8)
                normal = seg.get('mean_normal', [0, 1, 0])

                # Determine position descriptors
                pos_desc = []
                if rel_pos[1] > 0.7:
                    pos_desc.append("top")
                elif rel_pos[1] < 0.3:
                    pos_desc.append("bottom")
                else:
                    pos_desc.append("middle")

                if rel_pos[0] > 0.7:
                    pos_desc.append("front")
                elif rel_pos[0] < 0.3:
                    pos_desc.append("back")

                if rel_pos[2] > 0.6:
                    pos_desc.append("right")
                elif rel_pos[2] < 0.4:
                    pos_desc.append("left")
                else:
                    pos_desc.append("center")

                # Dominant normal direction
                abs_normal = np.abs(normal)
                dom_axis = np.argmax(abs_normal)
                normal_dirs = {0: ("facing front/back", "X"), 1: ("facing up/down", "Y"), 2: ("facing left/right", "Z")}
                normal_sign = "positive" if normal[dom_axis] > 0 else "negative"

                segments_desc.append({
                    "id": i,
                    "face_count": seg.get('face_count', 0),
                    "area_percent": seg.get('area', 0) * 100,
                    "position": " ".join(pos_desc),
                    "relative_height": f"{rel_pos[1]:.2f}",
                    "dominant_facing": f"{normal_dirs[dom_axis][0]} ({normal_sign} {normal_dirs[dom_axis][1]})"
                })

            prompt = f"""Identify what part of a {product_type} each mesh segment represents.

Product Name: {product_type}

Segments to identify:
{json.dumps(segments_desc, indent=2)}

For each segment, determine the most appropriate part name based on:
1. Its position in the mesh (top/bottom/front/back/left/right/center)
2. Its relative size (larger segments are typically main body parts)
3. Its facing direction (normals indicate surface orientation)
4. What parts a typical object like this would have

Respond with ONLY valid JSON in this exact format:
{{
    "segments": [
        {{
            "id": 0,
            "part_name": "descriptive_part_name",
            "part_type": "category (e.g., structural, functional, decorative, connector)",
            "confidence": 0.0 to 1.0,
            "material_suggestions": ["material1", "material2", "material3"]
        }}
    ],
    "product_analysis": "brief description of what the product appears to be"
}}

Use meaningful, specific part names that match visible geometry (e.g., base, body, cap, handle, hinge, panel, strap, latch)."""

            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a 3D product design expert who can identify parts of products from their geometric properties. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2048
            }

            response = await self._post_json("/chat/completions", payload)

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(content[json_start:json_end])
                else:
                    result = {"segments": [], "product_analysis": "unknown"}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse segment identification response")
                result = {"segments": [], "product_analysis": "unknown"}

            return NIMResponse(success=True, data=result, usage=data.get("usage"))

        except Exception as e:
            mode = self._endpoint_mode()
            logger.error("Segment identification NIM error (%s): %s", mode, e)
            return NIMResponse(success=False, error=f"{mode} NIM error: {e}")

    async def get_curated_materials(self, part_name: str, part_type: str,
                                    product_type: str) -> NIMResponse:
        """
        Get AI-curated material recommendations for a specific part.

        Args:
            part_name: Name of the part (e.g., "outsole", "upper")
            part_type: Category of part (structural, functional, decorative)
            product_type: Overall product name

        Returns:
            NIMResponse with detailed material specifications
        """
        if not self.is_configured:
            return NIMResponse(success=False, error="NIM API key not configured")

        try:
            prompt = f"""Recommend materials for a {part_name} ({part_type} part) of this product ({product_type}).

Provide detailed material suggestions with PBR properties for realistic rendering.

Respond with ONLY valid JSON:
{{
    "primary_material": {{
        "name": "material_name",
        "category": "leather/fabric/rubber/plastic/metal/glass/wood/ceramic",
        "pbr_properties": {{
            "base_color": [R, G, B] (0-1 range),
            "roughness": 0.0-1.0,
            "metallic": 0.0-1.0,
            "normal_strength": 0.0-2.0,
            "ao_strength": 0.0-1.0
        }}
    }},
    "alternative_materials": [
        {{
            "name": "alt_material_name",
            "category": "category",
            "pbr_properties": {{ ... }}
        }}
    ],
    "texture_recommendations": ["texture_type1", "texture_type2"]
}}"""

            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a materials scientist specializing in product design and PBR rendering. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }

            response = await self._post_json("/chat/completions", payload)

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                result = json.loads(content[json_start:json_end])
            except json.JSONDecodeError:
                result = {}

            return NIMResponse(success=True, data=result, usage=data.get("usage"))

        except Exception as e:
            mode = self._endpoint_mode()
            logger.error("Material curation NIM error (%s): %s", mode, e)
            return NIMResponse(success=False, error=f"{mode} NIM error: {e}")


class TripoNIMClient:
    """
    Client for Tripo AI NIM - 3D model generation and processing.
    Can be used for mesh repair and enhancement.
    """

    def __init__(self, config: Optional[NIMConfig] = None):
        self.config = config or NIMConfig()
        self.base_url = "https://integrate.api.nvidia.com/v1"

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    async def generate_from_image(self, image_path: Path,
                                 output_format: str = "glb") -> NIMResponse:
        """
        Generate 3D model from image using Tripo NIM.

        Args:
            image_path: Path to input image
            output_format: Output format (glb, gltf, usdz, fbx, obj)

        Returns:
            NIMResponse with generated model data or URL
        """
        if not self.is_configured:
            return NIMResponse(success=False, error="NIM API key not configured")

        if not HTTPX_AVAILABLE:
            return NIMResponse(success=False, error="httpx required for Tripo NIM")

        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/ai/tripo/generate",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "image": image_data,
                        "output_format": output_format,
                        "generate_textures": True
                    }
                )
                response.raise_for_status()
                data = response.json()

                return NIMResponse(
                    success=True,
                    data=data
                )

        except Exception as e:
            logger.error(f"Tripo NIM error: {e}")
            return NIMResponse(success=False, error=str(e))


class NIMPipeline:
    """
    Orchestrates NIM services for AI-powered mesh analysis and labeling.
    Provides automatic product detection, segment identification, and material curation.
    """

    def __init__(self, config: Optional[NIMConfig] = None):
        self.config = config or NIMConfig()
        self.nim_client = NIMClient(config)
        self.tripo_client = TripoNIMClient(config)
        self._detected_product_type = None
        self._segment_analysis = None

    @property
    def is_configured(self) -> bool:
        return self.nim_client.is_configured

    def detect_product_type_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract a product name from the filename and pass it through unchanged.

        Args:
            filename: Name of the input file (e.g., 'model.glb')

        Returns:
            Dictionary with filename-derived product name for downstream NIM prompts
        """
        product_name_raw = Path(filename).stem.strip() or "generic"
        product_context = self._normalize_product_context(product_name_raw)
        product_category = product_context.get("product_category") or "generic"

        # Keep detected product type generic and reusable for prompts.
        self._detected_product_type = product_category

        logger.info(
            "  Product context from filename '%s': raw='%s' category='%s'",
            filename,
            product_name_raw,
            product_category,
        )

        return {
            "product_name": product_name_raw,
            "product_type": product_category,
            "product_category": product_category,
            "product_tokens": product_context.get("product_tokens", []),
            "confidence": product_context.get("category_confidence", 0.5),
            "expected_parts": [],
            "detection_method": "filename",
            "product_name_raw": product_name_raw,
        }

    def _normalize_product_context(self, product_name_raw: str) -> Dict[str, Any]:
        lowered = product_name_raw.lower()
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", lowered) if tok]

        keyword_map = {
            "shoe": {"shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "loafer", "moccasin", "sandal"},
            "bottle": {"bottle", "flask", "jar"},
            "chair": {"chair", "stool", "seat"},
        }

        for category, keywords in keyword_map.items():
            if any(token in keywords for token in tokens):
                return {
                    "product_category": category,
                    "product_tokens": tokens,
                    "category_confidence": 0.95,
                }

        return {
            "product_category": "generic",
            "product_tokens": tokens,
            "category_confidence": 0.5,
        }

    async def analyze_mesh_and_detect_product(self, mesh_info: Dict[str, Any],
                                               source_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Detect product name from filename.

        Args:
            mesh_info: Dictionary with mesh analysis data (not used, kept for compatibility)
            source_file: Path to original file - product name extracted from filename

        Returns:
            Dictionary with filename-derived product name
        """
        if source_file is not None:
            return self.detect_product_type_from_filename(str(source_file))

        # Fallback to generic if no filename provided
        logger.info("  No filename provided - using generic product name")
        return {"product_name": "generic", "product_type": "generic", "confidence": 0.0, "expected_parts": []}

    async def identify_segment_parts(self, segments: List[Any],
                                     mesh_bounds: List[List[float]],
                                     product_type: Optional[str] = None) -> List[SegmentAnalysis]:
        """
        Use AI to identify what each mesh segment represents.

        Args:
            segments: List of MeshSegment objects
            mesh_bounds: Overall mesh bounding box [[min], [max]]
            product_type: Product name hint (uses filename-derived name if not specified)

        Returns:
            List of SegmentAnalysis with AI-identified part names
        """
        product_type = product_type or self._detected_product_type or "generic"

        if not self.nim_client.is_configured:
            logger.info("NIM not configured - using geometric labels")
            return self._fallback_labeling(segments)

        logger.info(f"  Using AI to identify {len(segments)} segment parts for {product_type}...")

        # Prepare segment info for AI
        segments_info = []
        for seg in segments:
            seg_info = {
                'center': seg.properties.get('center', [0, 0, 0]),
                'mean_normal': seg.properties.get('mean_normal', [0, 1, 0]),
                'area': seg.properties.get('area', 0),
                'face_count': seg.properties.get('face_count', 0),
                'bounds': seg.properties.get('bounds', [[0, 0, 0], [1, 1, 1]])
            }
            segments_info.append(seg_info)

        response = await self.nim_client.identify_segment_parts(
            segments_info, product_type, mesh_bounds
        )

        if not response.success:
            logger.warning(f"Segment identification failed: {response.error}")
            return self._fallback_labeling(segments)

        result = response.data
        ai_segments = result.get('segments', [])

        # Match AI results to segments
        analyses = []
        for i, seg in enumerate(segments):
            if i < len(ai_segments):
                ai_seg = ai_segments[i]
                analysis = SegmentAnalysis(
                    segment_id=seg.segment_id,
                    original_label=seg.label,
                    ai_label=ai_seg.get('part_name', seg.label),
                    part_type=ai_seg.get('part_type', 'unknown'),
                    confidence=ai_seg.get('confidence', 0.5),
                    material_suggestions=ai_seg.get('material_suggestions', []),
                    properties=seg.properties
                )
            else:
                analysis = SegmentAnalysis(
                    segment_id=seg.segment_id,
                    original_label=seg.label,
                    ai_label=seg.label,
                    part_type='unknown',
                    confidence=0.0,
                    material_suggestions=[],
                    properties=seg.properties
                )
            analyses.append(analysis)
            logger.info(f"    Segment {i}: {analysis.ai_label} ({analysis.part_type}, "
                       f"confidence: {analysis.confidence:.2f})")

        self._segment_analysis = analyses
        return analyses

    def _fallback_labeling(self, segments: List[Any]) -> List[SegmentAnalysis]:
        """Fallback labeling when AI is not available"""
        analyses = []
        for seg in segments:
            analysis = SegmentAnalysis(
                segment_id=seg.segment_id,
                original_label=seg.label,
                ai_label=seg.label,
                part_type='unknown',
                confidence=0.0,
                material_suggestions=[],
                properties=seg.properties
            )
            analyses.append(analysis)
        return analyses

    async def get_part_segmentation_criteria(self, product_name: str,
                                              mesh_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask AI to describe the geometric characteristics of each part for segmentation.

        This is the KEY method for intelligent segmentation - the AI tells us:
        - What parts to expect for this product
        - Where each part is located (height range, position)
        - What direction the faces point (normal direction)
        - Special characteristics (small, large, connected, etc.)

        Args:
            product_name: Product name derived from the filename
            mesh_info: Basic mesh properties (bounds, vertex count, etc.)

        Returns:
            Dictionary with part names and their segmentation criteria
        """
        if not self.nim_client.is_configured:
            logger.info("NIM not configured - using default part criteria")
            return self._get_default_segmentation_criteria(product_name)

        logger.info(f"  Asking AI for segmentation criteria for '{product_name}'...")

        examples = self._criteria_prompt_examples()

        prompt = f"""You are a 3D product expert. I need to segment a 3D mesh of a {product_name} into its component parts using SAM3 vision AI with text prompts.

    For this {product_name}, describe each expected part and its geometric characteristics that can be used to identify it in a 3D mesh.

The mesh has:
- Bounding box: {mesh_info.get('bounds', 'unknown')}
- Vertex count: {mesh_info.get('num_vertices', 'unknown')}
- Face count: {mesh_info.get('num_faces', 'unknown')}

For EACH part, provide:
1. part_name: A clear name for the part (use underscores, e.g., "rubber_sole")
2. sam_prompt: A short noun phrase (1-3 words) for SAM3 vision detection (e.g., "flat base", "top cap", "outer shell")
3. height_range: [min, max] as normalized 0-1 values (0=bottom, 1=top)
4. normal_direction: Which way the faces point ("down", "up", "outward", "inward", "forward", "backward", "any")
5. position: Where it's located ("center", "front", "back", "left", "right", "all_around", "any")
6. relative_size: Expected size ("tiny", "small", "medium", "large", "dominant")
7. special_features: Any special characteristics like "has_holes", "thin_elements", "curved", "flat", "disconnected_parts"

IMPORTANT: The sam_prompt should be a simple noun phrase that a vision AI can recognize visually (e.g., "base", "top cap", "metal handle").

Here are brief examples for formatting only (do NOT copy these parts verbatim):
{examples}

Respond with ONLY valid JSON in this exact format:
{{
    "product_type": "{product_name}",
    "parts": [
        {{
            "part_name": "base",
            "sam_prompt": "flat base",
            "height_range": [0.0, 0.15],
            "normal_direction": "down",
            "position": "center",
            "relative_size": "medium",
            "special_features": ["flat"],
            "description": "The bottom base surface"
        }}
    ]
}}

Be specific and accurate for a {product_name}. Include 8-12 parts if the product is complex, and at least 6 parts for simple products. Use visually distinctive sam_prompts."""

        response = await self.nim_client.chat_completion(prompt)

        if not response.success:
            logger.warning(f"Failed to get segmentation criteria: {response.error}")
            return self._get_default_segmentation_criteria(product_name)

        try:
            content = response.data
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                criteria = json.loads(content[json_start:json_end])
                logger.info(f"  AI provided criteria for {len(criteria.get('parts', []))} parts")
                for part in criteria.get('parts', []):
                    logger.info(f"    - {part.get('part_name')}: height {part.get('height_range')}, "
                               f"normal {part.get('normal_direction')}, size {part.get('relative_size')}")
                return criteria
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse AI response: {e}")

        return self._get_default_segmentation_criteria(product_name)

    def _get_default_segmentation_criteria(self, product_name: str) -> Dict[str, Any]:
        """Default segmentation criteria when AI is not available"""
        return {
            'product_type': product_name,
            'parts': [
                {'part_name': 'base', 'height_range': [0.0, 0.20], 'normal_direction': 'down',
                 'position': 'center', 'relative_size': 'small', 'special_features': ['flat']},
                {'part_name': 'main_body', 'height_range': [0.10, 0.90], 'normal_direction': 'outward',
                 'position': 'all_around', 'relative_size': 'dominant', 'special_features': []},
                {'part_name': 'top', 'height_range': [0.80, 1.0], 'normal_direction': 'up',
                 'position': 'center', 'relative_size': 'small', 'special_features': []},
            ]
        }

    def _criteria_prompt_examples(self) -> str:
        """Short example snippets to guide the AI on output shape and quality."""
        example = {
            "examples": [
                {
                    "product_type": "sample_object",
                    "parts": [
                        {"part_name": "base", "sam_prompt": "flat base", "height_range": [0.0, 0.15],
                         "normal_direction": "down", "position": "any", "relative_size": "medium",
                         "special_features": ["flat"]},
                        {"part_name": "main_body", "sam_prompt": "main body", "height_range": [0.15, 0.85],
                         "normal_direction": "outward", "position": "all_around", "relative_size": "dominant",
                         "special_features": []},
                        {"part_name": "top_cap", "sam_prompt": "top cap", "height_range": [0.85, 1.0],
                         "normal_direction": "up", "position": "center", "relative_size": "small",
                         "special_features": []}
                    ]
                },
                {
                    "product_type": "sample_container",
                    "parts": [
                        {"part_name": "body", "sam_prompt": "container body", "height_range": [0.05, 0.70],
                         "normal_direction": "outward", "position": "all_around", "relative_size": "dominant",
                         "special_features": ["curved"]},
                        {"part_name": "neck", "sam_prompt": "narrow neck", "height_range": [0.75, 0.90],
                         "normal_direction": "outward", "position": "center", "relative_size": "small",
                         "special_features": ["narrow"]},
                        {"part_name": "cap", "sam_prompt": "top cap", "height_range": [0.85, 1.0],
                         "normal_direction": "up", "position": "center", "relative_size": "small",
                         "special_features": []}
                    ]
                }
            ]
        }
        return json.dumps(example, indent=2)

    async def get_curated_materials_for_segments(self, analyses: List[SegmentAnalysis],
                                                  product_type: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get AI-curated material recommendations for all identified segments.

        Args:
            analyses: List of SegmentAnalysis from identify_segment_parts
            product_type: Product name hint

        Returns:
            Dictionary mapping part names to material specifications
        """
        product_type = product_type or self._detected_product_type or "generic"

        if not self.nim_client.is_configured:
            logger.info("NIM not configured - using default materials")
            return {}

        logger.info(f"  Getting AI-curated materials for {len(analyses)} parts...")

        materials: Dict[str, Dict] = {}
        semaphore = asyncio.Semaphore(max(1, int(self.config.max_concurrency)))

        async def _fetch(idx: int, analysis: SegmentAnalysis) -> tuple[int, Optional[Dict[str, Dict]]]:
            # Preserve deterministic output order by carrying index through gather.
            if analysis.material_suggestions:
                return idx, {
                    analysis.ai_label: {
                        "suggestions": analysis.material_suggestions,
                        "part_type": analysis.part_type,
                    }
                }

            async with semaphore:
                response = await self.nim_client.get_curated_materials(
                    analysis.ai_label,
                    analysis.part_type,
                    product_type,
                )

            if not response.success:
                return idx, None

            data = dict(response.data or {})
            data["part_type"] = analysis.part_type
            return idx, {analysis.ai_label: data}

        tasks = [_fetch(idx, analysis) for idx, analysis in enumerate(analyses)]
        for _, item in sorted(await asyncio.gather(*tasks), key=lambda pair: pair[0]):
            if item:
                materials.update(item)

        return materials

    async def enhance_segmentation(self, mesh_data_dict: Dict[str, Any],
                                  product_type: str,
                                  base_segments: List[Dict]) -> List[Dict]:
        """
        Use AI to enhance mesh segmentation with better labels and suggestions.
        Legacy method for backwards compatibility.
        """
        if not self.nim_client.is_configured:
            logger.info("NIM not configured - using base segmentation")
            return base_segments

        # Get AI analysis
        mesh_desc = f"Vertices: {mesh_data_dict.get('num_vertices', 'unknown')}, " \
                   f"Faces: {mesh_data_dict.get('num_faces', 'unknown')}, " \
                   f"Bounds: {mesh_data_dict.get('bounds', 'unknown')}"

        response = await self.nim_client.analyze_mesh_for_segmentation(
            mesh_desc, product_type
        )

        if not response.success:
            logger.warning(f"NIM analysis failed: {response.error}")
            return base_segments

        suggestions = response.data

        # Enhance segments with AI suggestions
        enhanced = []
        suggested_parts = suggestions.get('suggested_parts', [])

        for i, segment in enumerate(base_segments):
            enhanced_segment = segment.copy()

            # Try to match with suggested parts
            if i < len(suggested_parts):
                enhanced_segment['ai_suggested_label'] = suggested_parts[i]

            hints = suggestions.get('segmentation_hints', {})
            if enhanced_segment.get('label') in hints:
                enhanced_segment['ai_hint'] = hints[enhanced_segment['label']]

            enhanced.append(enhanced_segment)

        return enhanced

    async def get_material_recommendations(self, segments: List[Dict],
                                          product_type: str) -> Dict[str, Any]:
        """
        Get AI-powered material recommendations for segments.
        Legacy method for backwards compatibility.
        """
        if not self.nim_client.is_configured:
            return {}

        part_names = [seg.get('label', f'part_{i}') for i, seg in enumerate(segments)]

        response = await self.nim_client.suggest_materials(part_names, product_type)

        if response.success:
            return response.data
        else:
            logger.warning(f"Material suggestions failed: {response.error}")
            return {}

    async def close(self):
        """Clean up resources"""
        await self.nim_client.close()

