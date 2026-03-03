"""Combined step workflow orchestration."""

from pathlib import Path
from typing import Callable, Optional, Tuple

from ..config import SegmentationMethod
from ..step1.runner import run_mesh_to_multimesh_async
from ..step2.runner import run_multimesh_to_mdl_json
from ..step3.runner import run_curate_multimesh
from .naming import derive_product_name


async def run_step12(
    source: Path,
    dest: Path,
    name: Optional[str],
    segmentation_method: SegmentationMethod,
    api_key: Optional[str],
    top_k: Optional[int],
    nim_rerank: Optional[bool],
    nim_base_url: Optional[str],
    nim_profile: Optional[str],
    nim_auth_mode: Optional[str],
    nim_max_concurrency: Optional[int],
    nim_max_retries: Optional[int],
    nim_retry_backoff: Optional[float],
    verbose: bool,
) -> Path:
    output_json = Path(dest)
    output_dir = output_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = name or derive_product_name(Path(source))
    step1_result = await run_mesh_to_multimesh_async(
        source=Path(source),
        dest=output_dir,
        name=model_name,
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
    if not step1_result.success:
        errors = "; ".join(step1_result.errors or ["step1 failed"])
        raise RuntimeError(f"step1 failed: {errors}")

    step1_usd = output_dir / f"{model_name}_multi_mesh.usd"
    return await run_multimesh_to_mdl_json(
        source=step1_usd,
        dest=output_json,
        name=model_name,
        api_key=api_key,
        top_k=top_k,
        nim_rerank=nim_rerank,
        nim_base_url=nim_base_url,
        nim_profile=nim_profile,
        nim_auth_mode=nim_auth_mode,
        nim_max_concurrency=nim_max_concurrency,
        nim_max_retries=nim_max_retries,
        nim_retry_backoff=nim_retry_backoff,
    )


async def run_step23(
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
) -> Tuple[Path, Path, Path]:
    output_dir = Path(dest)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = name or derive_product_name(Path(source))
    materials_json = output_dir / f"{model_name}_materials.json"

    if progress_cb:
        progress_cb("[Step 2/3] Building curated materials JSON...")

    curated_path = await run_multimesh_to_mdl_json(
        source=Path(source),
        dest=materials_json,
        name=model_name,
        api_key=api_key,
        top_k=top_k,
        nim_rerank=nim_rerank,
        nim_base_url=nim_base_url,
        nim_profile=nim_profile,
        nim_auth_mode=nim_auth_mode,
        nim_max_concurrency=nim_max_concurrency,
        nim_max_retries=nim_max_retries,
        nim_retry_backoff=nim_retry_backoff,
        progress_cb=progress_cb,
    )

    if progress_cb:
        progress_cb("[Step 3/3] Applying curated materials and exporting bound outputs...")
    usd_out, glb_out = run_curate_multimesh(
        Path(source),
        curated_path,
        output_dir,
        model_name,
        progress_cb=progress_cb,
    )
    return curated_path, usd_out, glb_out


async def run_step123(
    source: Path,
    dest: Path,
    name: Optional[str],
    segmentation_method: SegmentationMethod,
    api_key: Optional[str],
    top_k: Optional[int],
    nim_rerank: Optional[bool],
    nim_base_url: Optional[str],
    nim_profile: Optional[str],
    nim_auth_mode: Optional[str],
    nim_max_concurrency: Optional[int],
    nim_max_retries: Optional[int],
    nim_retry_backoff: Optional[float],
    verbose: bool,
) -> Tuple[Path, Path, Path]:
    output_dir = Path(dest)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = name or derive_product_name(Path(source))

    step1_result = await run_mesh_to_multimesh_async(
        source=Path(source),
        dest=output_dir,
        name=model_name,
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
    if not step1_result.success:
        errors = "; ".join(step1_result.errors or ["step1 failed"])
        raise RuntimeError(f"step1 failed: {errors}")

    step1_usd = output_dir / f"{model_name}_multi_mesh.usd"
    materials_json = output_dir / f"{model_name}_materials.json"
    curated_path = await run_multimesh_to_mdl_json(
        source=step1_usd,
        dest=materials_json,
        name=model_name,
        api_key=api_key,
        top_k=top_k,
        nim_rerank=nim_rerank,
        nim_base_url=nim_base_url,
        nim_profile=nim_profile,
        nim_auth_mode=nim_auth_mode,
        nim_max_concurrency=nim_max_concurrency,
        nim_max_retries=nim_max_retries,
        nim_retry_backoff=nim_retry_backoff,
    )
    usd_out, glb_out = run_curate_multimesh(step1_usd, curated_path, output_dir, model_name)
    return curated_path, usd_out, glb_out
