"""Step 2 runner: multi-mesh model to curated MDL JSON."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..common.naming import derive_product_name
from ..common.nim_cli import apply_nim_cli_overrides
from ..config import AppConfig
from .material_rag import MaterialRAGCurator
from ..common.nim_integration import NIMPipeline, SegmentAnalysis
from ..step3.usd_pipeline import ModelToUSDConverter, USDImporter

logger = logging.getLogger(__name__)


def _parse_root_list(raw_value: Optional[str]) -> List[Path]:
    if not raw_value:
        return []

    normalized = str(raw_value).strip()
    if not normalized:
        return []

    # Accept JSON-list syntax (e.g. ["C:/path"]) and plain delimited strings.
    try:
        decoded = json.loads(normalized)
        if isinstance(decoded, list):
            return [Path(str(item).strip()) for item in decoded if str(item).strip()]
    except Exception:
        pass

    roots: List[Path] = []
    for token in re.split(r"[;,]", normalized):
        value = token.strip().strip('"').strip("'")
        if value:
            roots.append(Path(value))
    return roots


def _read_dotenv_map(dotenv_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not dotenv_path.exists():
        return values

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_material_rag_paths(config: AppConfig) -> Tuple[List[Path], Path]:
    roots = [Path(root) for root in config.rag.material_roots if str(root).strip()]
    existing_roots = [root for root in roots if root.exists()]

    if not existing_roots:
        fallback_roots = [root for root in _parse_root_list(config.mdl_roots) if root.exists()]
        if fallback_roots:
            existing_roots = fallback_roots

    dotenv_roots: List[Path] = []
    if not existing_roots:
        dotenv_values = _read_dotenv_map(Path(".env"))
        dotenv_rag = [
            root
            for root in _parse_root_list(dotenv_values.get("PRODCONFIG_RAG__MATERIAL_ROOTS"))
            if root.exists()
        ]
        dotenv_mdl = [
            root
            for root in _parse_root_list(dotenv_values.get("PRODCONFIG_MDL_ROOTS"))
            if root.exists()
        ]
        dotenv_roots = dotenv_rag or dotenv_mdl
        if dotenv_roots:
            existing_roots = dotenv_roots

            process_rag = os.getenv("PRODCONFIG_RAG__MATERIAL_ROOTS")
            process_mdl = os.getenv("PRODCONFIG_MDL_ROOTS")
            if process_rag or process_mdl:
                logger.warning(
                    "Detected stale process material-root env override. "
                    "Using .env roots instead. process(PRODCONFIG_RAG__MATERIAL_ROOTS=%r, PRODCONFIG_MDL_ROOTS=%r)",
                    process_rag,
                    process_mdl,
                )

    if not existing_roots:
        rag_roots_debug = [str(root) for root in roots]
        mdl_roots_debug = [str(root) for root in _parse_root_list(config.mdl_roots)]
        dotenv_values = _read_dotenv_map(Path(".env"))
        dotenv_rag_debug = [str(root) for root in _parse_root_list(dotenv_values.get("PRODCONFIG_RAG__MATERIAL_ROOTS"))]
        dotenv_mdl_debug = [str(root) for root in _parse_root_list(dotenv_values.get("PRODCONFIG_MDL_ROOTS"))]
        raise RuntimeError(
            "ERROR: This is not supported/compatible - no material roots configured. "
            "Set PRODCONFIG_RAG__MATERIAL_ROOTS (or PRODCONFIG_MDL_ROOTS) to a valid existing path, "
            "or provide --root. "
            f"Resolved RAG roots: {rag_roots_debug or '[]'}; "
            f"Resolved MDL roots: {mdl_roots_debug or '[]'}; "
            f".env RAG roots: {dotenv_rag_debug or '[]'}; "
            f".env MDL roots: {dotenv_mdl_debug or '[]'}."
        )

    index_dir = config.rag.index_dir
    if not index_dir.is_absolute():
        index_dir = config.temp_path / index_dir

    return existing_roots, index_dir


async def run_multimesh_to_mdl_json(
    source: Path,
    dest: Path,
    name: Optional[str],
    api_key: Optional[str],
    top_k: Optional[int],
    nim_rerank: Optional[bool],
    nim_base_url: Optional[str],
    nim_profile: Optional[str],
    nim_auth_mode: Optional[str],
    nim_max_concurrency: Optional[int],
    nim_max_retries: Optional[int],
    nim_retry_backoff: Optional[float],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    if progress_cb:
        progress_cb("[Step 2/3] Validating source input...")

    source = Path(source)
    if not source.exists():
        parent = source.parent if source.parent != Path("") else Path(".")
        stem = source.stem
        candidates = sorted(parent.glob(f"{stem}.*")) if parent.exists() else []
        hint = ""
        if candidates:
            hint = " Available files: " + ", ".join(str(path) for path in candidates[:5])
        raise FileNotFoundError(f"Input source not found: {source}.{hint}")

    output_path = Path(dest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb("[Step 2/3] Preparing runtime configuration...")

    config = AppConfig(output_path=output_path.parent)
    if api_key:
        config.nim.api_key = api_key

    apply_nim_cli_overrides(
        config,
        nim_base_url=nim_base_url,
        nim_profile=nim_profile,
        nim_auth_mode=nim_auth_mode,
        nim_max_concurrency=nim_max_concurrency,
        nim_max_retries=nim_max_retries,
        nim_retry_backoff=nim_retry_backoff,
    )

    product_name = name or derive_product_name(source)

    converter = ModelToUSDConverter(config.omniverse)
    if progress_cb:
        progress_cb("[Step 2/3] Normalizing source to USD if needed...")
    runtime_usd = source if source.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"} else converter.convert(
        source,
        output_path.parent / f"{source.stem}_input.usd",
    )

    importer = USDImporter(config.omniverse)
    if progress_cb:
        progress_cb("[Step 2/3] Importing mesh parts from USD...")
    meshes, _ = importer.import_usd(runtime_usd)
    if not meshes:
        raise ValueError(f"No meshes found in source file: {source}")

    segments: List[Dict[str, Any]] = []
    analyses: List[SegmentAnalysis] = []
    for idx, mesh in enumerate(meshes):
        label = mesh.name or f"mesh_{idx}"
        segments.append({"id": idx, "label": label, "part_type": "unknown"})
        analyses.append(
            SegmentAnalysis(
                segment_id=idx,
                original_label=label,
                ai_label=label,
                part_type="unknown",
                confidence=1.0,
                material_suggestions=[],
                properties={},
            )
        )

    nim_pipeline = NIMPipeline(config.nim) if config.nim.is_configured_for_inference() else None
    ai_materials: Dict[str, Any] = {}
    product_context = {
        "product_name_raw": product_name,
        "product_category": "generic",
        "product_tokens": [tok for tok in re.split(r"[^a-zA-Z0-9]+", product_name.lower()) if tok],
    }

    if nim_pipeline and nim_pipeline.is_configured:
        if progress_cb:
            progress_cb("[Step 2/3] Fetching AI material recommendations (NIM)...")
        detected = nim_pipeline.detect_product_type_from_filename(str(source))
        product_context = {
            "product_name_raw": detected.get("product_name_raw") or product_name,
            "product_category": detected.get("product_category") or detected.get("product_type") or "generic",
            "product_tokens": detected.get("product_tokens") or product_context["product_tokens"],
        }
        ai_materials = await nim_pipeline.get_curated_materials_for_segments(analyses, product_name)

    roots, index_dir = resolve_material_rag_paths(config)
    curator = MaterialRAGCurator(roots, index_dir)
    if progress_cb:
        progress_cb("[Step 2/3] Building/loading material retrieval index...")

    resolved_top_k = top_k or config.rag.top_k
    if resolved_top_k is None:
        raise ValueError("PRODCONFIG_RAG__TOP_K is required or pass --top-k")

    candidate_pool_size = config.rag.candidate_pool_size or resolved_top_k
    constraints: Dict[str, float] = {}
    if config.rag.roughness_tolerance is not None:
        constraints["roughness_tolerance"] = config.rag.roughness_tolerance
    if config.rag.metallic_tolerance is not None:
        constraints["metallic_tolerance"] = config.rag.metallic_tolerance
    if config.rag.opacity_tolerance is not None:
        constraints["opacity_tolerance"] = config.rag.opacity_tolerance

    rerank_enabled = config.rag.nim_rerank_enabled if nim_rerank is None else nim_rerank
    if rerank_enabled is None:
        rerank_enabled = bool(nim_pipeline and nim_pipeline.is_configured)

    curated = await curator.curate(
        segments=segments,
        ai_materials=ai_materials,
        product_name=product_name,
        product_context=product_context,
        top_k=resolved_top_k,
        candidate_pool_size=candidate_pool_size,
        constraints=constraints,
        similarity_threshold=config.rag.similarity_threshold,
        allowlist_strict=config.rag.allowlist_strict,
        allowlist_policy=config.rag.allowlist_policy,
        use_product_name_in_query=config.rag.use_product_name_in_query,
        nim_client=nim_pipeline.nim_client if nim_pipeline else None,
        nim_rerank_enabled=rerank_enabled,
        nim_rerank_temperature=config.rag.nim_rerank_temperature,
        nim_rerank_max_tokens=config.rag.nim_rerank_max_tokens,
    )

    if progress_cb:
        progress_cb("[Step 2/3] Writing curated materials JSON...")
    with open(output_path, "w") as f:
        json.dump(curated, f, indent=2)

    if nim_pipeline:
        await nim_pipeline.close()

    return output_path
