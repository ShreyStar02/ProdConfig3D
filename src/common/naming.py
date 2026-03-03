"""Naming helpers shared across step workflows."""

from pathlib import Path


def derive_product_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_multi_mesh"):
        stem = stem[: -len("_multi_mesh")].rstrip("_")
    return stem or "model"
