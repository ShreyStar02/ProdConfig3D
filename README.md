# Universal 3D Product Configurator

A **zero-shot mesh segmentation system** that works on **ANY product type** without training data.

## Key Features

- **Zero-Shot**: No training required - works on any product instantly
- **AI-Powered**: Uses NVIDIA NIM (LLaMA 3.1 70B) for automatic part detection
- **Multi-Signal Fusion**: Combines SAM vision + UV boundaries + geometric analysis
- **Omniverse Ready**: Outputs USD files and curated MDL material references

## Quick Start

```bash
# 1. Setup environment (Windows/Linux)
bash setup.sh

# 2. Configure API key (optional but recommended)
# Edit .env and set
PRODCONFIG_NIM__API_KEY=your_nim_api_key
PRODCONFIG_NIM__PORT=19002
PRODCONFIG_NIM__MODEL=meta/llama-3.1-70b-instruct

# 3. Run using USD-first input
python main.py process input/your_product.usd
```

## Input Mode (USD-First)

Recommended input mode is USD family for stable production behavior:
- `.usd`
- `.usda`
- `.usdc`
- `.usdz`

Common interchange formats are also supported:
- `.glb`
- `.gltf`

Additional formats (`.obj`, `.fbx`, `.stl`, `.ply`, `.off`) are accepted by the loader but are best-effort and may require normalization/cleanup before segmentation.

## CLI Reference

### 3-Step Workflow

```bash
# Step 1: single mesh -> multi-mesh USD/GLB
python main.py mesh-to-multimesh --source input/your_product.usd --dest output/your_product

# Step 2: multi-mesh -> curated MDL JSON
python main.py multimesh-to-mdl-json --source output/your_product/your_product_multi_mesh.usd --dest output/your_product/your_product_materials.json

# Step 3: apply curated JSON -> bound USD/GLB
python main.py curate-multimesh --source output/your_product/your_product_multi_mesh.usd --materials-json output/your_product/your_product_materials.json --dest output/your_product

# Optional combined runs
python main.py step12 --source input/your_product.usd --dest output/your_product/your_product_materials.json
python main.py step23 --source output/your_product/your_product_multi_mesh.usd --dest output/your_product
python main.py step123 --source input/your_product.usd --dest output/your_product
```

Aliases are also available: `step1`, `step2`, `step3`.
Combined aliases are available: `step12`, `step23`, `step123`.

Combined descriptive commands are also available:
- `mesh-to-mdl-json` (same as `step12`)
- `mdl-json-to-bound-multimesh` (same as `step23`)
- `run-all` (same as `step123`)

Descriptive example wrappers are available in `examples/`:
- `step1_mesh_to_multimesh.py`
- `step2_multimesh_to_mdl_json.py`
- `step3_mdl_json_to_bound_multimesh.py`
- `step12_mesh_to_mdl_json.py`
- `step23_multimesh_to_bound_multimesh.py`
- `step123_entire_flow.py`

NIM endpoint/runtime flags are available on step and process commands:
- `--nim-profile cloud|local|custom`
- `--nim-base-url <url>`
- `--nim-auth-mode auto|required|none`
- `--nim-max-concurrency <int>`
- `--nim-max-retries <int>`
- `--nim-retry-backoff <seconds>`

### Process a Single Mesh

```bash
python main.py process <input_path> [OPTIONS]
```

Flags:
- `--output`, `-o` Output directory (default: `./output`)
- `--name`, `-n` Model name for output files
- `--api-key`, `-k` NIM API key (or set `PRODCONFIG_NIM__API_KEY`)
- `--segments`, `-s` Target number of segments
- `--verbose`, `-v` Verbose output

### Batch Processing

```bash
python main.py batch <input_dir> [OPTIONS]
```

Flags:
- `--output`, `-o` Output directory (default: `./output`)
- `--pattern`, `-p` File glob pattern (default: `*.glb`; recommended: `*.usd`)
- `--api-key`, `-k` NIM API key (or set `PRODCONFIG_NIM__API_KEY`)

### Material RAG (Standalone)

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

## How It Works

1. **Input**: USD-first mesh input (`.usd`, `.usda`, `.usdc`, `.usdz`; `.glb`/`.gltf` also supported)
2. **AI Detection**: NIM analyzes filename and detects product type
3. **Part Generation**: LLaMA 3.1 70B generates part names + geometric hints
4. **Segmentation**: Multi-signal fusion (SAM + xatlas + geometric)
5. **Output**: Segmented USD/GLB + curated MDL material references

## Project Structure

```
ProdConfig/
├── main.py              # Main entry point
├── src/                 # Core source code
├── examples/            # Demo scripts
├── models/              # SAM checkpoints
├── docs/                # Documentation
├── requirements.txt     # Dependency file
├── input/               # Input mesh files
└── output/              # Generated outputs
```

## Documentation

- [User Guide](docs/USER_GUIDE.md) - How to use the system
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Modules](docs/MODULES.md) - Module reference
- [Setup](docs/SETUP.md) - Installation and configuration

## Requirements

- Python 3.10+
- NVIDIA GPU (optional, for SAM acceleration)
- NIM API Key (optional, for LLM features)

## License

Proprietary - EY Internal Use


