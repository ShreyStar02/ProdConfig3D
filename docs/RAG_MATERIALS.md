# Material RAG Pipeline

## Overview

The material RAG pipeline curates MDL materials for each mesh segment using:

1. NIM material JSON (primary/alternative materials and PBR hints).
2. A local MDL scan of configured material roots.
3. Text retrieval plus PBR constraint filtering.

The output is written to `{product}_materials.json` alongside the normal pipeline outputs.

## Configuration

Material curation is configured via `PRODCONFIG_RAG__*` settings:

```
# One or more material roots (comma/semicolon separated)
PRODCONFIG_RAG__MATERIAL_ROOTS=./materials

# Optional overrides
PRODCONFIG_RAG__TOP_K=3
PRODCONFIG_RAG__CANDIDATE_POOL_SIZE=20
PRODCONFIG_RAG__SIMILARITY_THRESHOLD=0.0
PRODCONFIG_RAG__ROUGHNESS_TOLERANCE=0.2
PRODCONFIG_RAG__METALLIC_TOLERANCE=0.2
PRODCONFIG_RAG__OPACITY_TOLERANCE=0.2
PRODCONFIG_RAG__ALLOWLIST_STRICT=true
PRODCONFIG_RAG__NIM_RERANK_ENABLED=true
```

The index cache is stored under `temp/material_rag` by default.

## CLI Usage

```
python main.py materials-rag output/shoe_result.json --top-k 5
python main.py materials-rag output/shoe_result.json --root "./materials"
python main.py materials-rag output/shoe_result.json --nim-rerank --nim-profile cloud
```

## Output Schema

`{product}_materials.json`

```
{
  "version": "1.0",
  "product_name": "shoe",
  "index_info": {
    "backend": "clip",
    "index_hash": "...",
    "built_at": "2026-02-12T10:22:15Z",
    "material_roots": ["./materials"],
    "total_materials": 1234
  },
  "segments": [
    {
      "label": "upper_material",
      "segment_id": 1,
      "part_type": "unknown",
      "query": {
        "text": "shoe upper_material fabric leather",
        "textures": ["leather_grain"],
        "desired_pbr": {
          "roughness": 0.6,
          "metallic": 0.0
        }
      },
      "candidates": [
        {
          "doc_id": "...",
          "name": "Leather_Brown_Smooth",
          "category": "Leather",
          "source_root": "./materials",
          "source_path": "./materials/nvidia/vMaterials_2/Leather/Leather_Brown_Smooth.mdl",
          "rel_path": "nvidia/vMaterials_2/Leather/Leather_Brown_Smooth.mdl",
          "tags": ["nvidia", "vmaterials_2", "leather"],
          "score": 0.8231,
          "similarity": 0.7810,
          "pbr_values": {
            "roughness": 0.55,
            "metallic": 0.0,
            "opacity": 1.0
          },
          "recommended_texture_path": "./materials/nvidia/vMaterials_2/Leather/textures/leather_brown_smooth_norm.jpg",
          "recommended_texture_paths": [
            "./materials/nvidia/vMaterials_2/Leather/textures/leather_brown_smooth_norm.jpg",
            "./materials/nvidia/vMaterials_2/Leather/textures/leather_brown_smooth_color.jpg",
            "./materials/nvidia/vMaterials_2/Leather/textures/leather_brown_smooth_roughness.jpg"
          ],
          "texture_match_reason": "mdl_name_match"
        }
      ]
    }
  ],
  "warnings": []
}
```

## Notes

- PBR constraints are applied only when a material has matching numeric values in its MDL file.
- If no material roots are configured, the pipeline fails with `ERROR: This is not supported/compatible - no material roots configured`.
- Texture recommendations are additive metadata and do not affect Step3 MDL binding. The current texture extension set is `.jpg`, `.jpeg`, and `.png`.

