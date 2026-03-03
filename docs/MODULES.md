# Module Reference

## Source Modules (`src/`)

### `config.py`
Configuration management using Pydantic.

**Classes:**
- `NIMConfig`: NVIDIA NIM API configuration
- `SegmentationConfig`: Segmentation parameters
- `AppConfig`: Main application settings

**Key Settings:**
```python
NIMConfig(
    api_key="nvapi-xxx",           # NIM API key
    base_url=f"http://localhost:{os.getenv('PRODCONFIG_NIM__PORT', '19002')}/v1",
    model=os.getenv("PRODCONFIG_NIM__MODEL", "meta/llama-3.1-70b-instruct")
)
```

---

### `mesh_processor.py`
3D mesh loading, repair, and analysis.

**Classes:**
- `MeshData`: Container for mesh data (vertices, faces, UVs, normals)
- `MeshLoader`: Loads meshes from various formats
- `MeshRepairer`: Fixes mesh issues (holes, non-manifold edges)
- `MeshAnalyzer`: Extracts geometric features

**Usage:**
```python
from src.step1.mesh_processor import MeshLoader, MeshData

loader = MeshLoader()
meshes, metadata = loader.load("product.usd")
mesh = meshes[0]
print(f"Vertices: {mesh.num_vertices}, Faces: {mesh.num_faces}")
```

---

### `common/nim_integration.py`
NVIDIA NIM API client for LLM features.

**Classes:**
- `NIMClient`: Low-level API client
- `NIMPipeline`: High-level orchestrator for mesh analysis

**Key Methods:**
```python
nim = NIMPipeline()

# Detect product type from filename
info = nim.detect_product_type_from_filename("shoe.usd")
# Returns: {"product_type": "shoe", "confidence": 1.0, "expected_parts": [...]}

# Get segmentation criteria from LLM
criteria = await nim.get_part_segmentation_criteria("shoe", mesh_info)
# Returns: {"parts": [{"part_name": "sole", "height_range": [0, 0.1]}, ...]}
```

---

### `step1/zeroshot_segmentation.py`
Zero-shot mesh segmentation without training data.

**Classes:**
- `ZeroShotConfig`: Segmentation configuration
- `ZeroShotMaxAccuracySegmenter`: Main segmentation engine
- `MeshSegment`: Output segment container

**Key Functions:**
```python
from src.step1.zeroshot_segmentation import segment_any_product

segments = segment_any_product(
    mesh_data=mesh,
    ai_criteria={"parts": [{"part_name": "sole"}, {"part_name": "upper"}]}
)
```

**Segmentation Pipeline:**
1. UV Boundary Detection (xatlas)
2. SAM Vision Segmentation
3. AI-Guided Geometric Scoring
4. MRF Signal Fusion
5. Segment Creation

---

### `step1/segmentation.py`
Basic geometric segmentation methods.

**Key Types / Functions:**
- `MeshSegment`: Segment container
- `MeshSegmentationPipeline.process()`: Base segmentation pass
- `AIGuidedSegmenter.segment_with_criteria()`: Criteria-guided geometric segmentation
- `merge_segments_by_label()`: Post-classification semantic merge

---

### `step1/pipeline.py`
Main processing pipeline orchestration.

**Classes:**
- `ProductConfiguratorPipeline`: Full processing workflow
- `PipelineResult`: Output container

**Usage:**
```python
from src.step1.pipeline import ProductConfiguratorPipeline

pipeline = ProductConfiguratorPipeline()
result = await pipeline.process(
    input_path="product.usd",
    output_dir="./output"
)
```

---

### `step2/material_rag.py`
Material retrieval and curation from local MDL libraries.

**Classes:**
- `MaterialIndexer`: Scans MDL roots and builds a local index
- `MaterialRAGCurator`: Ranks materials using text similarity and PBR constraints

**Usage:**
```python
from src.step2.material_rag import MaterialRAGCurator

curator = MaterialRAGCurator([Path("./materials")], Path("./temp/material_rag"))
curated = curator.curate(segments, ai_materials, product_name="shoe", top_k=3, constraints={
    "roughness_tolerance": 0.2,
    "metallic_tolerance": 0.2,
    "opacity_tolerance": 0.2
})
```

---

### `step3/usd_pipeline.py`
USD/GLB output generation for Omniverse.

**Key Classes / Methods:**
- `USDExporter.export_multi_mesh()`: Generate multi-mesh USD
- `USDExporter.export_multi_mesh_glb()`: Generate multi-mesh GLB
- `USDExporter.apply_curated_materials()`: Bind curated MDL materials
- `USDImporter.import_usd()`: Load meshes from USD
- `ModelToUSDConverter.convert()`: Normalize model input to USD family

---

### `common/` and `step*/` modules
Workflow logic is split into human-friendly modules:

- `src/common/naming.py`: product naming helpers
- `src/common/nim_cli.py`: NIM CLI override wiring
- `src/common/nim_probe.py`: endpoint smoke probing
- `src/common/step_combos.py`: step12/step23/step123 orchestration
- `src/step1/runner.py`: step1 execution (sync/async)
- `src/step2/runner.py`: step2 execution
- `src/step3/runner.py`: step3 material binding

---

### `cli.py`
Command-line interface using Typer.

**Commands:**
```bash
python main.py process input/product.usd
python main.py step1 --source input/product.usd --dest output/product
python main.py step2 --source output/product/product_multi_mesh.usd --dest output/product/product_materials.json
python main.py step3 --source output/product/product_multi_mesh.usd --materials-json output/product/product_materials.json --dest output/product
python main.py step12 --source input/product.usd --dest output/product/product_materials.json
python main.py step23 --source output/product/product_multi_mesh.usd --dest output/product
python main.py step123 --source input/product.usd --dest output/product
python main.py nim-smoke --nim-profile local --nim-base-url http://localhost:$PRODCONFIG_NIM__PORT/v1
```

**Step Aliases:**
- `step1` = `mesh-to-multimesh`
- `step2` = `multimesh-to-mdl-json`
- `step3` = `curate-multimesh`
- `step12` = `mesh-to-mdl-json`
- `step23` = `mdl-json-to-bound-multimesh`
- `step123` = `run-all`

---

## Example Scripts (`examples/`)

### `demo_zeroshot_segmentation.py`
Demonstrates zero-shot segmentation with NIM auto-detection.

```bash
# Auto-detect everything
python examples/demo_zeroshot_segmentation.py --mesh input/shoe.usd

# Manual part specification
python examples/demo_zeroshot_segmentation.py --mesh input/product.usd --parts handle body base
```

### `step1_mesh_to_multimesh.py`
Thin wrapper that delegates to `main.py step1`.

### `step2_multimesh_to_mdl_json.py`
Thin wrapper that delegates to `main.py step2`.

### `step3_mdl_json_to_bound_multimesh.py`
Thin wrapper that delegates to `main.py step3`.

### `step12_mesh_to_mdl_json.py`
Thin wrapper that delegates to `main.py step12`.

### `step23_multimesh_to_bound_multimesh.py`
Thin wrapper that delegates to `main.py step23`.

### `step123_entire_flow.py`
Thin end-to-end wrapper that delegates to `main.py step123`.

## External Dependencies

| Package | Purpose |
|---------|---------|
| `trimesh` | Mesh loading and manipulation |
| `numpy` | Numerical operations |
| `scipy` | Scientific computing (MRF optimization) |
| `xatlas` | UV unwrapping and boundary detection |
| `torch` | Deep learning (SAM) |
| `ultralytics` | SAM 3 implementation |
| `clip` | Text-image encoding for SAM |
| `httpx` | Async HTTP client for NIM |
| `pydantic` | Configuration management |


