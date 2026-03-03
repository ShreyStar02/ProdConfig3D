# System Architecture

## Overview

The Universal 3D Product Configurator is a multi-stage pipeline that transforms any 3D mesh into semantically segmented parts with material recommendations.

Primary execution surface is a 3-step CLI workflow:

1. `step1` / `mesh-to-multimesh`: input mesh -> multi-mesh USD/GLB.
2. `step2` / `multimesh-to-mdl-json`: multi-mesh -> curated MDL JSON.
3. `step3` / `curate-multimesh`: curated JSON + multi-mesh -> bound USD/GLB.

Combined workflows are also first-class:

1. `step12` / `mesh-to-mdl-json`: mesh -> curated MDL JSON.
2. `step23` / `mdl-json-to-bound-multimesh`: multi-mesh -> curated JSON -> bound USD/GLB.
3. `step123` / `run-all`: mesh -> bound USD/GLB end-to-end.

Operational readiness is validated with `nim-smoke`, which probes cloud/local NIM endpoint reachability using configured profile, auth mode, and base URL.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: USD-First Mesh                        │
│                    (shoe.usd, bottle.usda, etc.)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Product Detection                       │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐   │
│  │ Filename Parser │───▶│  NIM (LLaMA 3.1 70B)                 │   │
│  │ "shoe.usd"      │     │  - Detect product type               │   │
│  └─────────────────┘     │  - Generate part names               │   │
│                          │  - Provide geometric hints           │   │
│                          └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Multi-Signal Segmentation               │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐    ┌─────────────────────────┐   │
│  │   xatlas    │   │    SAM 3    │    │   AI-Guided Geometric   │   │
│  │ UV Seams    │   │ Text Prompt │    │   Height/Normal/Size    │   │
│  │ (weight:    │   │ (weight:    │    │   (weight: 0.25)        │   │
│  │  0.20)      │   │  0.55)      │    │                         │   │
│  └──────┬──────┘   └──────┬──────┘    └───────────┬─────────────┘   │
│         │                 │                       │                 │
│         └─────────────────┴───────────────────────┘                 │
│                           │                                         │
│                           ▼                                         │
│              ┌────────────────────────┐                             │
│              │     MRF Optimization   │                             │
│              │  (Markov Random Field) │                             │
│              │   Signal Fusion        │                             │
│              └────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: Output Generation                       │
│                                                                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌────────────────────┐ │
│  │  Segmented USD  │   │  Segmented GLB  │   │  Material RAG      │ │
│  │  (Omniverse)    │   │  (Universal)    │   │  (Curated MDL)     │ │
│  └─────────────────┘   └─────────────────┘   └────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. NIM Integration (`src/common/nim_integration.py`)

Handles all AI/LLM communication via NVIDIA NIM:
- **Product Detection**: Analyzes filename to determine product type
- **Part Generation**: Uses LLaMA 3.1 70B to generate contextually appropriate part names
- **Geometric Hints**: Provides height ranges, normal directions, and size expectations

### 2. Zero-Shot Segmentation (`src/step1/zeroshot_segmentation.py`)

The maximum accuracy segmentation pipeline:
- **SAM 3 (Ultralytics)**: Vision-based segmentation using text prompts
- **xatlas**: UV unwrapping for natural boundary detection
- **Geometric Scoring**: Height, normal, curvature analysis
- **MRF Fusion**: Optimally combines all signals

### 3. Mesh Processing (`src/step1/mesh_processor.py`)

Handles mesh I/O and manipulation:
- **MeshLoader**: Loads various 3D formats via trimesh
- **MeshRepairer**: Fixes non-manifold edges, holes
- **MeshAnalyzer**: Extracts geometric features

### 4. Pipeline Orchestration (`src/step1/pipeline.py`)

Coordinates the full processing workflow:
- Loads and validates input
- Runs segmentation
- Generates outputs (USD, GLB, MDL)
- Supports step1 mode by skipping curation/apply when requested by CLI

### 5. CLI Orchestration (`src/cli.py`)

Defines high-level user workflows:
- Full pipeline: `process`
- Step commands: `mesh-to-multimesh`, `multimesh-to-mdl-json`, `curate-multimesh`
- Combined commands: `mesh-to-mdl-json`, `mdl-json-to-bound-multimesh`, `run-all`
- Aliases: `step1`, `step2`, `step3`, `step12`, `step23`, `step123`
- NIM readiness probe: `nim-smoke`

Step orchestration logic is split into focused modules:
- `src/step1/runner.py`
- `src/step2/runner.py`
- `src/step3/runner.py`
- `src/common/step_combos.py`

## Segmentation Weights

The MRF fusion uses these default weights:

| Signal | Weight | Description |
|--------|--------|-------------|
| SAM | 0.55 | Vision-based semantic segmentation |
| Geometric | 0.25 | Height, normal, curvature analysis |
| UV Boundaries | 0.20 | xatlas UV seam detection |

## Data Flow

```
Input Mesh
    │
    ├──▶ Load (trimesh) ──▶ MeshData
    │
    ├──▶ Filename ──▶ NIM ──▶ Product Type + Parts + Hints
    │
    ├──▶ Render Views ──▶ SAM 3 ──▶ Vision Masks
    │
    ├──▶ xatlas Unwrap ──▶ UV Charts ──▶ Boundary Faces
    │
    ├──▶ Geometric Analysis ──▶ Per-Face Scores
    │
    └──▶ MRF Fusion ──▶ Final Labels ──▶ Segments
                                            │
                                            ├──▶ USD
                                            ├──▶ GLB
                                            └──▶ Material RAG
```

## GPU vs CPU Mode

| Feature | GPU (CUDA) | CPU |
|---------|------------|-----|
| SAM Text Prompts | Full support | Limited |
| Processing Speed | Fast | Slower |
| Memory Usage | Higher | Lower |
| Accuracy | Best | Good |

Compatibility checks are strict for required accelerated paths; unsupported GPU/runtime states return explicit errors.

## Extension Points

### Adding New Product Types

Edit `src/config.py` to add predefined product types, or let NIM auto-detect.

### Custom Segmentation Signals

Implement the `SegmentationSignal` interface in `src/step1/zeroshot_segmentation.py`.

### New Output Formats

Extend `src/step3/usd_pipeline.py` for additional export formats.
