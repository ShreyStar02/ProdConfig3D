"""
Zero-shot mesh segmentation demo using SAM3 text prompts.

This script renders multiple views of a mesh, runs SAM3 Promptable Concept
Segmentation (PCS) on each view, projects masks back to mesh faces via a
face-id render pass, and returns MeshSegments.

Usage:
    python examples/demo_zeroshot_segmentation.py --mesh input/shoe.glb
    python examples/demo_zeroshot_segmentation.py --mesh input/shoe.glb --parts sole upper
"""

import sys
import logging
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from src.step1.mesh_processor import MeshLoader
from src.step1.zeroshot_segmentation import (
    segment_any_product,
    ZeroShotMaxAccuracySegmenter,
    ZeroShotConfig,
    SAM3_AVAILABLE,
    TORCH_AVAILABLE
)
from src.common.nim_integration import NIMPipeline


async def get_parts_from_nim(mesh_path: str) -> tuple:
    """
    Use NIM to auto-detect product type and part names.

    Args:
        mesh_path: Path to mesh file

    Returns:
        tuple: (product_type, part_names_list)
    """
    nim = NIMPipeline()
    filename = Path(mesh_path).stem

    mesh_info = {
        "filename": filename,
        "file_extension": Path(mesh_path).suffix
    }
    product_info = nim.detect_product_type_from_filename(filename)
    product_type = product_info.get("product_type", "generic")

    print(f"  [NIM] Detected product type: {product_type}")
    print(f"  [NIM] Confidence: {product_info.get('confidence', 'N/A')}")

    try:
        criteria = await nim.get_part_segmentation_criteria(product_type, mesh_info)
        parts = criteria.get("parts", [])
        part_names = []
        for part in parts:
            if isinstance(part, dict):
                sam_prompt = part.get("sam_prompt", part.get("part_name", part.get("name", "unknown")))
                part_names.append(sam_prompt)
            elif isinstance(part, str):
                part_names.append(part)

        if part_names:
            print(f"  [NIM] Auto-generated parts: {', '.join(part_names)}")
            return product_type, part_names

    except Exception as exc:
        print(f"  [NIM] Warning: Could not get parts from NIM: {exc}")

    expected_parts = product_info.get("expected_parts", [])
    if expected_parts:
        print(f"  [NIM] Using expected parts: {', '.join(expected_parts)}")
        return product_type, expected_parts

    print("  [NIM] Using generic part names")
    return product_type, ["main_body", "top", "bottom", "attachment"]


def build_ai_criteria(part_names: List[str], geometric_hints: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    ai_criteria = {"parts": []}
    for part_name in part_names:
        part_spec = {"part_name": part_name}
        if geometric_hints and part_name in geometric_hints:
            part_spec.update(geometric_hints[part_name])
        ai_criteria["parts"].append(part_spec)
    return ai_criteria


def segment_generic_product(
    mesh_path: str,
    part_names: Optional[List[str]] = None,
    geometric_hints: Optional[Dict[str, Dict[str, Any]]] = None,
    image_size: int = 640,
    conf: float = 0.25,
    half: bool = True,
    model_path: str = "./models/sam3.pt",
    min_faces: int = 50
) -> None:
    """
    Segment ANY product mesh into parts using SAM3 PCS.
    """
    print("\n" + "=" * 70)
    print("ZERO-SHOT GENERIC MESH SEGMENTATION (SAM3 PCS)")
    print("=" * 70)

    mesh_file = Path(mesh_path)
    if not mesh_file.exists():
        print(f"  [ERROR] Mesh not found: {mesh_path}")
        return

    if part_names is None or len(part_names) == 0:
        print("\n  [AUTO-DETECT] Using NIM to detect product type and parts...")
        _, part_names = asyncio.run(get_parts_from_nim(mesh_path))
    else:
        print(f"\n  [MANUAL] Using provided parts: {part_names}")

    loader = MeshLoader()
    meshes, _ = loader.load(mesh_file)
    mesh_data = meshes[0]
    print(f"\n  Loaded: {mesh_file.name}")
    print(f"  Vertices: {mesh_data.num_vertices}, Faces: {mesh_data.num_faces}")
    print(f"  Parts to detect: {part_names}")

    ai_criteria = build_ai_criteria(part_names, geometric_hints)

    config = ZeroShotConfig(
        image_size=image_size,
        conf=conf,
        half=half,
        model_path=model_path,
        min_segment_faces=min_faces
    )
    segmenter = ZeroShotMaxAccuracySegmenter(config)
    segments = segmenter.segment(mesh_data, ai_criteria)

    print("\n" + "-" * 50)
    print("SEGMENTATION RESULTS:")
    print("-" * 50)
    for seg in segments:
        print(
            f"  {seg.label}: {seg.properties.get('face_count', 0)} faces, "
            f"confidence={seg.confidence:.2f}"
        )

    return segments


def check_dependencies():
    """Check which components are available."""
    print("\n" + "=" * 70)
    print("DEPENDENCY CHECK")
    print("=" * 70)

    try:
        import trimesh  # noqa: F401
        trimesh_ok = True
    except Exception:
        trimesh_ok = False

    try:
        import pyrender  # noqa: F401
        pyrender_ok = True
    except Exception:
        pyrender_ok = False

    print(f"  PyTorch: {'INSTALLED' if TORCH_AVAILABLE else 'NOT FOUND'}")
    print(f"  SAM 3 (Ultralytics): {'INSTALLED' if SAM3_AVAILABLE else 'NOT FOUND'}")
    print(f"  trimesh: {'INSTALLED' if trimesh_ok else 'NOT FOUND'}")
    print(f"  pyrender: {'INSTALLED' if pyrender_ok else 'NOT FOUND'}")

    if TORCH_AVAILABLE:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  GPU: Not available (CPU mode)")

    print("\n  Installation command:")
    if not SAM3_AVAILABLE or not trimesh_ok or not pyrender_ok:
        print("    bash setup.sh")


def show_usage_examples():
    """Show usage examples for the segmentation system."""
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)

    print("""
  # FULLY AUTOMATIC: NIM detects product type and parts
  python examples/demo_zeroshot_segmentation.py --mesh input/shoe.glb

  # MANUAL OVERRIDE: explicit part names for SAM3 PCS
  python examples/demo_zeroshot_segmentation.py --mesh input/shoe.glb --parts sole upper laces

  # Change image size and confidence threshold
  python examples/demo_zeroshot_segmentation.py --mesh input/shoe.glb --image-size 768 --conf 0.3
""")


def parse_hint(hint_str: str) -> tuple:
    """Parse a hint string like 'part_name:property=value'"""
    try:
        part_and_prop = hint_str.split(":")
        part_name = part_and_prop[0]
        prop_value = part_and_prop[1].split("=")
        prop_name = prop_value[0]
        value = prop_value[1]

        if "-" in value and not value.startswith("-"):
            parts = value.split("-")
            parsed_value = [float(parts[0]), float(parts[1])]
        elif value in ["up", "down", "outward", "inward", "forward", "backward"]:
            parsed_value = value
        else:
            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value

        return part_name, prop_name, parsed_value
    except Exception:
        return None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-Shot Generic Mesh Segmentation (SAM3 PCS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mesh product.glb              # Auto-detect parts via NIM
  %(prog)s --mesh product.glb --parts a b  # Manual override with specific parts
  %(prog)s --check                         # Check dependencies
  %(prog)s --examples                      # Show usage examples
        """
    )

    parser.add_argument("--mesh", type=str, help="Path to mesh file (.glb, .obj, .stl)")
    parser.add_argument("--parts", nargs="+", help="Optional: Part names to detect")
    parser.add_argument("--hint", action="append", help="Geometric hint: part:property=value")
    parser.add_argument("--check", action="store_true", help="Check dependencies")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    parser.add_argument("--image-size", type=int, default=640, help="Render size for SAM3 (square)")
    parser.add_argument("--conf", type=float, default=0.25, help="SAM3 confidence threshold")
    parser.add_argument("--half", action="store_true", help="Use FP16 if supported")
    parser.add_argument("--model", type=str, default="./models/sam3.pt", help="Path to sam3.pt")
    parser.add_argument("--min-faces", type=int, default=50, help="Minimum faces per segment")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           ZERO-SHOT GENERIC MESH SEGMENTATION                            ║
║                                                                          ║
║  SAM 3 PCS (text prompts) over multi-view renders                         ║
║  Masks are projected back to faces using face-id render pass              ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    if args.check:
        check_dependencies()
    elif args.examples:
        show_usage_examples()
    elif args.mesh:
        geometric_hints = {}
        if args.hint:
            for hint in args.hint:
                part_name, prop_name, value = parse_hint(hint)
                if part_name:
                    if part_name not in geometric_hints:
                        geometric_hints[part_name] = {}
                    geometric_hints[part_name][prop_name] = value

        segment_generic_product(
            mesh_path=args.mesh,
            part_names=args.parts,
            geometric_hints=geometric_hints if geometric_hints else None,
            image_size=args.image_size,
            conf=args.conf,
            half=args.half,
            model_path=args.model,
            min_faces=args.min_faces
        )
    else:
        check_dependencies()
        show_usage_examples()
        print("\nRun with --help for command line options.")
