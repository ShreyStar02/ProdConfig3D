"""Step 1 runner: single mesh to multi-mesh USD/GLB."""

import asyncio
from pathlib import Path
from typing import Optional

from ..common.nim_cli import apply_nim_cli_overrides
from ..config import AppConfig, SegmentationMethod
from .pipeline import ProductConfiguratorPipeline, PipelineResult


async def run_mesh_to_multimesh_async(
    source: Path,
    dest: Path,
    name: Optional[str],
    segmentation_method: SegmentationMethod,
    api_key: Optional[str],
    nim_base_url: Optional[str],
    nim_profile: Optional[str],
    nim_auth_mode: Optional[str],
    nim_max_concurrency: Optional[int],
    nim_max_retries: Optional[int],
    nim_retry_backoff: Optional[float],
    verbose: bool,
) -> PipelineResult:
    config = AppConfig(output_path=dest, verbose=verbose)
    config.segmentation.method = segmentation_method
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

    pipeline = ProductConfiguratorPipeline(config)
    return await pipeline.process(
        source,
        output_dir=dest,
        model_name=name or source.stem,
        run_material_curation=False,
        apply_curated_materials=False,
    )


def run_mesh_to_multimesh(
    source: Path,
    dest: Path,
    name: Optional[str],
    segmentation_method: SegmentationMethod,
    api_key: Optional[str],
    nim_base_url: Optional[str],
    nim_profile: Optional[str],
    nim_auth_mode: Optional[str],
    nim_max_concurrency: Optional[int],
    nim_max_retries: Optional[int],
    nim_retry_backoff: Optional[float],
    verbose: bool,
) -> PipelineResult:
    return asyncio.run(
        run_mesh_to_multimesh_async(
            source=source,
            dest=dest,
            name=name,
            segmentation_method=segmentation_method,
            api_key=api_key,
            nim_base_url=nim_base_url,
            nim_profile=nim_profile,
            nim_auth_mode=nim_auth_mode,
            nim_max_concurrency=nim_max_concurrency,
            nim_max_retries=nim_max_retries,
            nim_retry_backoff=nim_retry_backoff,
            verbose=verbose,
        )
    )
