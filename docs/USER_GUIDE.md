# User Guide

## Overview

The Universal 3D Product Configurator segments 3D meshes into meaningful parts automatically. The recommended production input mode is USD family (`.usd`, `.usda`, `.usdc`, `.usdz`).

## Basic Usage

### Command Line

```bash
# Step 1: single mesh -> multi-mesh outputs
python main.py step1 --source input/product.usd --dest output/product

# Step 2: multi-mesh -> curated materials JSON
python main.py step2 --source output/product/product_multi_mesh.usd --dest output/product/product_materials.json --top-k 3

# Step 3: apply curated JSON -> bound USD/GLB
python main.py step3 --source output/product/product_multi_mesh.usd --materials-json output/product/product_materials.json --dest output/product

# Combined options
python main.py step12 --source input/product.usd --dest output/product/product_materials.json
python main.py step23 --source output/product/product_multi_mesh.usd --dest output/product
python main.py step123 --source input/product.usd --dest output/product
```

### CLI Reference

#### Process a Single Mesh

```bash
python main.py process <input_path> [OPTIONS]
```

Flags:
- `--output`, `-o` Output directory (default: `./output`)
- `--name`, `-n` Model name for output files
- `--api-key`, `-k` NIM API key (or set `PRODCONFIG_NIM__API_KEY`)
- `--segments`, `-s` Target number of segments
- `--verbose`, `-v` Verbose output

#### Batch Processing

```bash
python main.py batch <input_dir> [OPTIONS]
```

Flags:
- `--output`, `-o` Output directory (default: `./output`)
- `--pattern`, `-p` File glob pattern (default: `*.glb`; recommended in USD-first workflows: `*.usd`)
- `--api-key`, `-k` NIM API key (or set `PRODCONFIG_NIM__API_KEY`)

#### Material RAG (Standalone)

```bash
python main.py materials-rag <input_json> [OPTIONS]
```

Flags:
- `--output`, `-o` Output curated JSON path
- `--root`, `-r` Material root folder(s) to scan (repeatable)
- `--top-k`, `-k` Top materials per segment
- `--rebuild` Force rebuild of the material index
- `--nim-rerank` / `--no-nim-rerank` Enable or disable NIM rerank

Notes:
- `<input_json>` can be a pipeline result JSON (`*_result.json`) or a NIM material JSON payload.
- Output defaults to `<input_stem>_materials.json` if `--output` is not provided.

### Zero-Shot Demo (Maximum Accuracy)

```bash
# Auto-detect product type and parts via NIM
python examples/demo_zeroshot_segmentation.py --mesh input/shoe.usd

# Manual override with specific parts
python examples/demo_zeroshot_segmentation.py --mesh input/product.usd --parts handle body base

# Check available components
python examples/demo_zeroshot_segmentation.py --check
```

### Material RAG Curation

```bash
# From a pipeline result JSON
python main.py materials-rag output/product_result.json --top-k 5

# From a NIM material JSON payload
python main.py materials-rag input/nim_materials.json --top-k 5

# Specify material roots and output file
python main.py materials-rag output/product_result.json \
	--root "./materials" \
	--output output/product_materials.json
```

### Example Wrappers (CLI Delegation)

```bash
# Step 1 wrapper
python examples/step1_mesh_to_multimesh.py \
	--source input/product.usd \
	--dest output/product

# Step 2 wrapper
python examples/step2_multimesh_to_mdl_json.py \
	--source output/product/product_multi_mesh.usd \
	--dest output/product/product_materials.json

# Step 3 wrapper
python examples/step3_mdl_json_to_bound_multimesh.py \
	--source output/product/product_multi_mesh.usd \
	--materials-json output/product/product_materials.json \
	--dest output/product

# End-to-end wrapper (step1 -> step2 -> step3)
python examples/step123_entire_flow.py --source input/product.usd --dest output/product

# Step12 wrapper
python examples/step12_mesh_to_mdl_json.py --source input/product.usd --dest output/product/product_materials.json

# Step23 wrapper
python examples/step23_multimesh_to_bound_multimesh.py --source output/product/product_multi_mesh.usd --dest output/product
```

## Input Formats

Recommended input formats (USD-first):
- `.usd`
- `.usda`
- `.usdc`
- `.usdz`

Common interchange formats:
- `.glb`
- `.gltf`

Best-effort formats (accepted by loader, quality varies by source mesh):
- `.obj`
- `.fbx`
- `.stl`
- `.ply`
- `.off`

## Output Files

After processing, the system generates:

```
output/
├── {product}_multi_mesh.usd    # Segmented USD for Omniverse
├── {product}_multi_mesh.glb    # Segmented GLB for viewers
├── {product}_result.json       # Metadata and segment info
└── {product}_materials.json     # Curated MDL candidates (RAG output)
```

For details on the RAG material pipeline and schema, see [docs/RAG_MATERIALS.md](docs/RAG_MATERIALS.md).

## Configuration

### Environment Variables (.env)

```bash
# NIM API key for NIM LLM features
PRODCONFIG_NIM__API_KEY=nvapi-xxxxx

# Default product type (generic = auto-detect)
PRODCONFIG_PRODUCT_TYPE=generic

# Output directory
PRODCONFIG_OUTPUT_PATH=./output

# Enable verbose logging
PRODCONFIG_VERBOSE=true

# Material RAG settings
PRODCONFIG_RAG__MATERIAL_ROOTS=./materials
PRODCONFIG_RAG__TOP_K=3
PRODCONFIG_RAG__CANDIDATE_POOL_SIZE=20
PRODCONFIG_RAG__ALLOWLIST_STRICT=true
PRODCONFIG_RAG__NIM_RERANK_ENABLED=true
```

### Hardware Compatibility

Required features fail fast when unsupported:

- **NIM local runtime** requires Docker + NVIDIA runtime access (`nvidia-smi`).
- **USD/Omniverse export path** requires `usd-core` (`pxr`).

If required compatibility is missing, commands return:

```text
ERROR: This is not supported/compatible - <reason>
```

## Segmentation Modes

### 1. Basic Pipeline (Fast)
Uses AI-guided geometric analysis + UV boundaries.
```bash
python main.py process input/product.usd
```

### 2. Zero-Shot Maximum Accuracy (Best Quality)
Combines SAM vision + xatlas + geometric + MRF fusion.
```bash
python examples/demo_zeroshot_segmentation.py --mesh input/product.usd
```

## Naming Convention

The system uses the filename to detect product type:

| Filename | Detected Type | Generated Parts |
|----------|--------------|-----------------|
| `shoe.usd` | shoe | sole, midsole, upper, tongue, lacing... |
| `bottle.usd` | bottle | cap, neck, body, base |
| `chair.usd` | furniture | seat, backrest, legs, armrests |
| `custom.usd` | generic | main_body, top, bottom, details |

## Troubleshooting

### "NIM not configured"
Set your NIM API key in `.env`:
```
PRODCONFIG_NIM__API_KEY=your_key_here
```

### "SAM checkpoint not found"
Download SAM model to `models/` folder:
```bash
# Download sam2.1_b.pt from:
# https://github.com/ultralytics/assets/releases
```

### "No module named 'clip'"
Install dependencies through the standard setup path:
```bash
bash setup.sh
```

### "No MDL materials found"
Ensure a valid material root is configured:
```
PRODCONFIG_RAG__MATERIAL_ROOTS=./materials
```

### GPU not detected
GPU features require a compatible NVIDIA runtime. Re-run standard setup after confirming the runtime:
```bash
nvidia-smi
bash setup.sh
```


