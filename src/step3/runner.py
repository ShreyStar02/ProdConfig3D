"""Step 3 runner: curated JSON to bound multi-mesh outputs."""

import json
from pathlib import Path
from typing import Callable, Optional, Tuple

from ..common.naming import derive_product_name
from ..config import AppConfig
from .usd_pipeline import ModelToUSDConverter, USDExporter


def run_curate_multimesh(
    source: Path,
    materials_json: Path,
    dest: Path,
    name: Optional[str],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, Path]:
    if progress_cb:
        progress_cb("[Step 3/3] Validating input files...")

    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    if not materials_json.exists():
        raise FileNotFoundError(f"Materials JSON not found: {materials_json}")

    output_dir = Path(dest)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = name or derive_product_name(source)

    with open(materials_json, "r") as f:
        curated = json.load(f)
    if not isinstance(curated, dict) or not isinstance(curated.get("segments"), list):
        raise ValueError("Invalid materials JSON: expected top-level object with 'segments' list")

    config = AppConfig(output_path=output_dir)
    converter = ModelToUSDConverter(config.omniverse)
    exporter = USDExporter(config.omniverse)

    usd_out = output_dir / f"{model_name}_multi_mesh.usd"
    glb_out = output_dir / f"{model_name}_multi_mesh.glb"
    if progress_cb:
        progress_cb("[Step 3/3] Preparing bound USD stage...")
    runtime_usd = converter.convert(source, usd_out)
    if runtime_usd.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise RuntimeError(f"Expected USD-compatible file after conversion, got: {runtime_usd}")

    if progress_cb:
        progress_cb("[Step 3/3] Applying curated materials to USD...")
    exporter.apply_curated_materials(runtime_usd, curated, root_name=model_name)
    if progress_cb:
        progress_cb("[Step 3/3] Exporting bound GLB...")
    converter.export_usd_to_glb(runtime_usd, glb_out)

    if progress_cb:
        progress_cb("[Step 3/3] Completed binding outputs.")

    return runtime_usd, glb_out
