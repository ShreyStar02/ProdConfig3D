#!/usr/bin/env python3
"""
Universal 3D Product Configurator
Main entry point for the application.

This is a GENERIC system that works on ANY product type.
AI automatically detects product types and identifies meaningful parts.

Usage:
    # CLI
    python main.py process model.glb --output ./output

    # Python API
    from main import process_model
    result = process_model("model.glb")
"""
import sys
import asyncio
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import AppConfig
from src.step1.pipeline import ProductConfiguratorPipeline, quick_process, PipelineResult


def process_model(
    input_path: str,
    output_dir: str = "./output",
    nim_api_key: Optional[str] = None,
    target_segments: Optional[int] = None
) -> PipelineResult:
    """
    Process a single 3D model through the complete pipeline.

    This is a GENERIC function that works on ANY product type.
    When using AI mode (with NIM API key), the system automatically
    detects the product type and identifies meaningful parts.

    Args:
        input_path: Path to input 3D model (GLB, GLTF, OBJ, FBX, STL, etc.)
        output_dir: Directory for output files
        nim_api_key: Optional NIM API key for AI-enhanced features
        target_segments: Optional target number of mesh segments

    Returns:
        PipelineResult with paths to generated files

    Example:
        >>> result = process_model("product.glb", output_dir="./output")
        >>> print(f"Multi-mesh USD: {result.multi_mesh_usd_path}")
        >>> print(f"Curated materials: {result.metadata.get('curated_materials_path')}")
    """
    # Configure
    config = AppConfig(
        output_path=Path(output_dir),
        verbose=True
    )

    if nim_api_key:
        config.nim.api_key = nim_api_key

    if target_segments:
        config.segmentation.target_num_segments = target_segments

    # Process
    pipeline = ProductConfiguratorPipeline(config)
    result = pipeline.process_sync(Path(input_path))

    return result


async def process_model_async(
    input_path: str,
    output_dir: str = "./output",
    nim_api_key: Optional[str] = None
) -> PipelineResult:
    """Async version of process_model"""
    config = AppConfig(
        output_path=Path(output_dir),
        verbose=True
    )

    if nim_api_key:
        config.nim.api_key = nim_api_key

    pipeline = ProductConfiguratorPipeline(config)
    return await pipeline.process(Path(input_path))


def main():
    """Main entry point - runs CLI or shows usage"""
    # Check if CLI args provided
    if len(sys.argv) > 1:
        # Run CLI
        from src.cli import main as cli_main
        cli_main()
    else:
        # Show usage
        print("=" * 60)
        print("Universal 3D Product Configurator")
        print("=" * 60)
        print("\nThis is a GENERIC system that works on ANY product type.")
        print("AI automatically detects product types and identifies parts.")
        print("\nUsage:")
        print("  python main.py process <input.glb> --output ./output")
        print("  python main.py batch ./models --pattern '*.glb'")
        print("  python main.py info")
        print("\nExamples:")
        print("  python main.py process input/product.glb")
        print("  python examples/demo_zeroshot_segmentation.py --mesh input/product.glb")
        print("\nPython API:")
        print("  from main import process_model")
        print("  result = process_model('product.glb')")
        print("\nRun 'python main.py --help' for more options.")


if __name__ == "__main__":
    main()
