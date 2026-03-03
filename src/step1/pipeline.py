"""
Product Configurator Pipeline - Main orchestration module.
Handles the complete workflow from single mesh to multi-mesh with curated materials.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import shutil

from ..config import AppConfig, SegmentationMethod, MeshRepairConfig
from .mesh_processor import MeshLoader, MeshRepairer, MeshAnalyzer, MeshData, merge_meshes
from .segmentation import MeshSegmentationPipeline, MeshSegment, merge_segments_by_label, AIGuidedSegmenter
from ..step3.usd_pipeline import USDExporter, ModelToUSDConverter, USDImporter
from ..common.nim_integration import NIMPipeline
from .segment_classifier import SegmentClassifier, classify_all_segments

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of the complete pipeline execution"""
    success: bool
    input_path: Path
    multi_mesh_usd_path: Optional[Path] = None
    multi_mesh_glb_path: Optional[Path] = None
    segments: Optional[List[MeshSegment]] = None
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    processing_time_seconds: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class ProductConfiguratorPipeline:
    """
    Main pipeline class that orchestrates the complete workflow:
    1. Normalize input to USD and load mesh(es)
    2. Repair mesh (fix AI-generated model issues)
    3. Segment mesh into named parts
    4. Export as multi-mesh USD
    5. Curate MDL material references
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.config.ensure_directories()

        # Initialize components
        self.mesh_loader = MeshLoader()
        self.mesh_repairer = MeshRepairer(self.config.mesh_repair)
        self.mesh_analyzer = MeshAnalyzer()
        self.segmentation_pipeline = MeshSegmentationPipeline(
            self.config.segmentation
        )
        self.usd_exporter = USDExporter(self.config.omniverse)
        self.model_to_usd_converter = ModelToUSDConverter(self.config.omniverse)
        self.usd_importer = USDImporter(self.config.omniverse)
        # Material curation uses local MDL scans and RAG ranking

        # NIM integration (optional)
        self.nim_pipeline = NIMPipeline(self.config.nim) if self.config.nim.api_key else None

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging"""
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def process(self, input_path: Path,
                     output_dir: Optional[Path] = None,
                     model_name: Optional[str] = None,
                     run_material_curation: bool = True,
                     apply_curated_materials: bool = True) -> PipelineResult:
        """
        Execute the complete pipeline.

        Args:
            input_path: Path to input 3D model (GLB, GLTF, OBJ, etc.)
            output_dir: Output directory (uses config default if not specified)
            model_name: Name for the output model (derived from input if not specified)

        Returns:
            PipelineResult with all outputs and metadata
        """
        start_time = datetime.now()
        input_path = Path(input_path)
        output_dir = Path(output_dir) if output_dir else self.config.output_path
        model_name = model_name or input_path.stem

        result = PipelineResult(
            success=False,
            input_path=input_path,
            metadata={
                'model_name': model_name,
                'product_name': model_name,
                'started_at': start_time.isoformat()
            }
        )

        try:
            logger.info(f"=" * 60)
            logger.info(f"Starting Product Configurator Pipeline")
            logger.info(f"Input: {input_path}")
            logger.info(f"Product Name: {model_name}")
            logger.info(f"=" * 60)

            # Step 1: Normalize input to USD and load meshes from USD
            logger.info("\n[Step 1/5] Normalizing input to USD and loading meshes...")
            runtime_usd_path = self._resolve_runtime_usd_input(input_path, output_dir)
            meshes, load_metadata = self.usd_importer.import_usd(runtime_usd_path)

            if not meshes:
                result.errors.append("No meshes found in input file")
                return result

            result.metadata['runtime_input_usd'] = str(runtime_usd_path)
            result.metadata['load_metadata'] = load_metadata
            logger.info(f"  Loaded {len(meshes)} mesh(es)")

            # Merge if multiple meshes
            if len(meshes) > 1:
                logger.info(f"  Merging {len(meshes)} meshes into single mesh...")
                mesh_data = merge_meshes(meshes)
            else:
                mesh_data = meshes[0]

            logger.info(f"  Total: {mesh_data.num_vertices} vertices, {mesh_data.num_faces} faces")

            # Step 2: Repair mesh
            logger.info("\n[Step 2/5] Repairing mesh...")
            segmentation_mesh = self._prepare_segmentation_mesh(mesh_data)
            result.metadata['repair_applied'] = True

            # Step 3: Analyze and segment mesh
            logger.info("\n[Step 3/5] Analyzing and segmenting mesh...")
            analysis = self.mesh_analyzer.analyze(segmentation_mesh)
            result.metadata['mesh_analysis'] = analysis

            detected_product_name = model_name
            product_context: Dict[str, Any] = {
                "product_name_raw": model_name,
                "product_category": "generic",
                "product_tokens": [],
            }
            ai_material_recommendations = {}
            segmentation_criteria = None

            # Step 3a: Get AI criteria from NIM (if available)
            # This provides part names and hints for segmentation
            if self.nim_pipeline and self.nim_pipeline.is_configured:
                logger.info("  [3a] Sending filename-derived product name to NIM...")
                product_detection = await self.nim_pipeline.analyze_mesh_and_detect_product(
                    {},  # mesh_info not used in filename-based detection
                    source_file=input_path
                )
                detected_product_name = product_detection.get('product_name', detected_product_name)
                product_context = {
                    "product_name_raw": product_detection.get("product_name_raw") or detected_product_name,
                    "product_category": product_detection.get("product_category") or product_detection.get("product_type") or "generic",
                    "product_tokens": product_detection.get("product_tokens") or [],
                }
                result.metadata['ai_detected_product'] = product_detection

                logger.info("  [3b] Getting AI segmentation criteria...")
                mesh_info = {
                    'bounds': analysis.get('bounds'),
                    'num_vertices': segmentation_mesh.num_vertices,
                    'num_faces': segmentation_mesh.num_faces,
                    'is_watertight': analysis.get('is_watertight'),
                    'filename': input_path.stem
                }
                segmentation_criteria = await self.nim_pipeline.get_part_segmentation_criteria(
                    detected_product_name, mesh_info
                )
                if segmentation_criteria is not None:
                    segmentation_criteria.setdefault("product_type", detected_product_name)
                result.metadata['segmentation_criteria'] = segmentation_criteria

            # Step 3b: Perform segmentation based on configured method
            segmentation_method = self.config.segmentation.method
            logger.info(f"  [3c] Segmenting with method: {segmentation_method.value}")

            if segmentation_method == SegmentationMethod.SAM3_ZEROSHOT:
                # SAM3 with text prompts - BEST quality
                from .zeroshot_segmentation import ZeroShotMaxAccuracySegmenter, ZeroShotConfig
                target_num_parts = None
                if segmentation_criteria and isinstance(segmentation_criteria.get("parts"), list):
                    base_count = len(segmentation_criteria.get("parts"))
                    target_num_parts = max(base_count + 3, 10)
                min_faces = min(self.config.segmentation.min_segment_faces, 10)
                sam3_segmenter = ZeroShotMaxAccuracySegmenter(
                    ZeroShotConfig(
                        min_segment_faces=min_faces,
                        min_small_segment_faces=12,
                        retry_conf=0.1,
                        retry_enabled=True,
                        target_num_parts=target_num_parts,
                        min_visible_views=3,
                        min_face_confidence=0.08,
                        min_component_faces=12,
                        cluster_dominance=0.6,
                        cluster_min_faces=80
                    )
                )
                segments = sam3_segmenter.segment(segmentation_mesh, segmentation_criteria)

            elif segmentation_method in (SegmentationMethod.HYBRID, SegmentationMethod.AI_ASSISTED):
                # AI-guided geometric segmentation
                if segmentation_criteria:
                    ai_segmenter = AIGuidedSegmenter(self.config.segmentation)
                    segments = ai_segmenter.segment_with_criteria(segmentation_mesh, segmentation_criteria)
                else:
                    # Fallback to geometric if no AI criteria
                    segments = self.segmentation_pipeline.process(
                        segmentation_mesh, self.config.custom_mesh_parts
                    )

            else:
                # Pure geometric segmentation
                segments = self.segmentation_pipeline.process(
                    segmentation_mesh, self.config.custom_mesh_parts
                )

            # Step 3d: Post-segmentation classification
            # Ensures ALL segments get proper names (no segment_X)
            # Uses dynamic vocabulary from AI criteria or NIM
            logger.info("  [3d] Classifying segments with CLIP...")
            segments = classify_all_segments(
                segments,
                product_description=product_context.get("product_category") or detected_product_name,
                ai_criteria=segmentation_criteria,
                confidence_threshold=0.25,
                nim_config=self.config.nim if self.config.nim.api_key else None
            )

            # Collapse micro-segments into semantic part-level meshes.
            before_merge_count = len(segments)
            segments = merge_segments_by_label(segments)
            if len(segments) != before_merge_count:
                logger.info(
                    "  [3d] Merged segments by label: %d -> %d",
                    before_merge_count,
                    len(segments),
                )

            # Step 3e: Get material recommendations (if NIM available)
            if self.nim_pipeline and self.nim_pipeline.is_configured and segments:
                logger.info("  [3e] Getting material recommendations...")
                from ..common.nim_integration import SegmentAnalysis
                segment_analyses = []
                # Deduplicate by semantic label to avoid sending thousands of near-identical
                # requests when geometric segmentation creates many small segment instances.
                seen_labels = set()
                for seg in segments:
                    label_key = str(seg.label or "").strip().lower()
                    if not label_key or label_key in seen_labels:
                        continue
                    seen_labels.add(label_key)
                    segment_analyses.append(SegmentAnalysis(
                        segment_id=seg.segment_id,
                        original_label=seg.label,
                        ai_label=seg.label,
                        part_type=seg.properties.get('part_type', 'structural'),
                        confidence=seg.confidence,
                        material_suggestions=[],
                        properties=seg.properties
                    ))

                logger.info(
                    "  [3e] Material request scope: unique_labels=%d total_segments=%d",
                    len(segment_analyses),
                    len(segments),
                )

                ai_material_recommendations = await self.nim_pipeline.get_curated_materials_for_segments(
                    segment_analyses, detected_product_name
                )
                result.metadata['ai_material_recommendations'] = ai_material_recommendations

                # Apply material suggestions to segments
                for seg, seg_analysis in zip(segments, segment_analyses):
                    if seg.label in ai_material_recommendations:
                        seg.properties['ai_material_suggestions'] = ai_material_recommendations[seg.label].get('materials', [])

            if self.config.mesh_repair.smooth_surface:
                logger.info("  [3f] Smoothing segment meshes for export...")
                segments = self._smooth_segment_meshes(segments)
                result.metadata['export_smoothing'] = True

            result.segments = segments
            result.metadata['detected_product_name'] = detected_product_name
            result.metadata['product_context'] = product_context
            logger.info(f"\n  Created {len(segments)} segments for product: {detected_product_name}")

            curated = None
            if run_material_curation:
                # Step 4: Curate material references from local MDL libraries
                logger.info("\n[Step 4/5] Curating material references...")
                curated_path = output_dir / f"{model_name}_materials.json"
                curated = await self._save_curated_materials(
                    segments,
                    curated_path,
                    detected_product_name,
                    ai_material_recommendations,
                    product_context,
                )
                result.metadata["curated_materials"] = curated
                result.metadata["curated_materials_path"] = str(curated_path)
                logger.info(f"  Curated materials saved: {curated_path}")
            else:
                logger.info("\n[Step 4/5] Skipping material curation (--step1 mode)")

            # Step 5: Export as multi-mesh USD and GLB
            logger.info("\n[Step 5/5] Exporting multi-mesh USD and GLB...")
            output_dir.mkdir(parents=True, exist_ok=True)
            usd_path = output_dir / f"{model_name}_multi_mesh.usd"
            glb_path = output_dir / f"{model_name}_multi_mesh.glb"

            result.multi_mesh_usd_path = self.usd_exporter.export_multi_mesh(
                segments,
                usd_path,
                root_name=model_name
            )
            logger.info(f"  USD exported: {result.multi_mesh_usd_path}")

            if result.multi_mesh_usd_path and run_material_curation and apply_curated_materials and curated:
                self.usd_exporter.apply_curated_materials(
                    result.multi_mesh_usd_path,
                    curated,
                    root_name=model_name
                )

            result.multi_mesh_glb_path = self.usd_exporter.export_multi_mesh_glb(
                segments,
                glb_path,
                root_name=model_name
            )
            logger.info(f"  GLB exported: {result.multi_mesh_glb_path}")

            # Success
            result.success = True
            end_time = datetime.now()
            result.processing_time_seconds = (end_time - start_time).total_seconds()
            result.metadata['completed_at'] = end_time.isoformat()

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Processing time: {result.processing_time_seconds:.2f} seconds")
            logger.info(f"Outputs:")
            logger.info(f"  - Multi-mesh USD: {result.multi_mesh_usd_path}")
            logger.info(f"  - Multi-mesh GLB: {result.multi_mesh_glb_path}")
            if run_material_curation:
                logger.info(f"  - Curated materials: {result.metadata.get('curated_materials_path')}")
            else:
                logger.info("  - Curated materials: skipped (step1 mode)")
            
            logger.info(f"{'=' * 60}")

            # Save result metadata
            self._save_result_metadata(result, output_dir / f"{model_name}_result.json")

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            result.errors.append(str(e))

        finally:
            # Cleanup NIM resources
            if self.nim_pipeline:
                await self.nim_pipeline.close()

        return result

    def _resolve_runtime_usd_input(self, input_path: Path, output_dir: Path) -> Path:
        input_suffix = input_path.suffix.lower()
        if input_suffix in {".usd", ".usda", ".usdc", ".usdz"}:
            logger.info("  Input is already USD-compatible: %s", input_path)
            return input_path

        runtime_usd = output_dir / f"{input_path.stem}_input.usd"
        converted = self.model_to_usd_converter.convert(input_path, runtime_usd)
        if converted.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
            raise RuntimeError(
                f"Input normalization did not produce USD (got: {converted}). "
                "Ensure USD conversion dependencies are available."
            )

        logger.info("  Converted input to USD: %s", converted)
        return converted

    def _prepare_segmentation_mesh(self, mesh_data: MeshData) -> MeshData:
        if not self.config.mesh_repair.smooth_surface:
            return self.mesh_repairer.repair(mesh_data)

        seg_config = self._clone_repair_config()
        seg_config.smooth_surface = False
        seg_config.merge_close_vertices = False
        seg_config.skip_if_clean = True
        seg_repairer = MeshRepairer(seg_config)
        return seg_repairer.repair(mesh_data)

    def _smooth_segment_meshes(self, segments: List[MeshSegment]) -> List[MeshSegment]:
        smooth_config = self._clone_repair_config()
        smooth_config.smooth_surface = True
        smooth_config.skip_if_clean = False
        smooth_config.merge_close_vertices = False
        smooth_config.fill_holes = False
        smooth_repairer = MeshRepairer(smooth_config)

        for seg in segments:
            seg.mesh_data = smooth_repairer.repair(seg.mesh_data)
        return segments

    def _clone_repair_config(self) -> MeshRepairConfig:
        try:
            return self.config.mesh_repair.model_copy(deep=True)
        except Exception:
            return MeshRepairConfig(**self.config.mesh_repair.model_dump())

    async def _segment_with_ai(self, mesh_data: MeshData,
                              analysis: Dict[str, Any]) -> List[MeshSegment]:
        """Use AI-enhanced segmentation"""
        # First do geometric segmentation
        base_segments = self.segmentation_pipeline.process(
            mesh_data,
            self.config.custom_mesh_parts
        )

        # Enhance with AI
        mesh_info = {
            'num_vertices': mesh_data.num_vertices,
            'num_faces': mesh_data.num_faces,
            'bounds': analysis.get('bounds'),
            'is_watertight': analysis.get('is_watertight')
        }

        # Convert segments to dict for NIM
        segment_dicts = [
            {
                'id': seg.segment_id,
                'label': seg.label,
                'face_count': seg.properties.get('face_count', 0),
                'center': seg.properties.get('center', [0, 0, 0])
            }
            for seg in base_segments
        ]

        # Get AI enhancements
        enhanced_dicts = await self.nim_pipeline.enhance_segmentation(
            mesh_info,
            analysis.get('filename') or (mesh_data.name or 'generic'),
            segment_dicts
        )

        # Apply AI labels if available
        for seg, enhanced in zip(base_segments, enhanced_dicts):
            if 'ai_suggested_label' in enhanced:
                seg.label = enhanced['ai_suggested_label']
                seg.mesh_data.name = enhanced['ai_suggested_label']

        return base_segments

    def _save_result_metadata(self, result: PipelineResult, output_path: Path):
        """Save pipeline result metadata to JSON"""
        metadata = {
            'success': result.success,
            'input_path': str(result.input_path),
            'multi_mesh_usd_path': str(result.multi_mesh_usd_path) if result.multi_mesh_usd_path else None,
            'multi_mesh_glb_path': str(result.multi_mesh_glb_path) if result.multi_mesh_glb_path else None,
            'segments': [
                {
                    'id': seg.segment_id,
                    'label': seg.label,
                    'confidence': seg.confidence,
                    'face_count': seg.properties.get('face_count', 0)
                }
                for seg in (result.segments or [])
            ],
            'metadata': result.metadata,
            'errors': result.errors,
            'warnings': result.warnings,
            'processing_time_seconds': result.processing_time_seconds
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Result metadata saved: {output_path}")

    def _save_material_references(self, curated: Dict[str, Any],
                                   output_path: Path,
                                   product_name: str) -> Dict[str, Any]:
        """Save lightweight material references derived from curated results."""
        material_refs = {
            "version": "1.0",
            "product_name": product_name,
            "note": "Derived from local MDL scan; curated list stored separately.",
            "index_info": curated.get("index_info", {}),
            "segments": []
        }

        for segment in curated.get("segments", []):
            seg_data = {
                "label": segment.get("label"),
                "segment_id": segment.get("segment_id"),
                "candidate_materials": [
                    {
                        "name": candidate.get("name"),
                        "category": candidate.get("category"),
                        "source_path": candidate.get("source_path"),
                        "score": candidate.get("score")
                    }
                    for candidate in segment.get("candidates", [])
                ]
            }
            material_refs["segments"].append(seg_data)

        with open(output_path, "w") as f:
            json.dump(material_refs, f, indent=2)

        return material_refs

    async def _save_curated_materials(self, segments: List[MeshSegment],
                                      output_path: Path,
                                      product_name: str,
                                      ai_recommendations: Dict[str, Any],
                                      product_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run material RAG and save curated results."""
        from ..step2.material_rag import MaterialRAGCurator

        material_roots, index_dir = self._resolve_material_rag_paths()
        curator = MaterialRAGCurator(material_roots, index_dir)
        top_k = self.config.rag.top_k
        if top_k is None:
            raise ValueError("Material RAG requires PRODCONFIG_RAG__TOP_K")

        candidate_pool_size = self.config.rag.candidate_pool_size or top_k
        constraints = {}
        if self.config.rag.roughness_tolerance is not None:
            constraints["roughness_tolerance"] = self.config.rag.roughness_tolerance
        if self.config.rag.metallic_tolerance is not None:
            constraints["metallic_tolerance"] = self.config.rag.metallic_tolerance
        if self.config.rag.opacity_tolerance is not None:
            constraints["opacity_tolerance"] = self.config.rag.opacity_tolerance

        nim_client = self.nim_pipeline.nim_client if self.nim_pipeline else None
        rerank_enabled = self.config.rag.nim_rerank_enabled
        if rerank_enabled is None:
            rerank_enabled = nim_client is not None

        curated = await curator.curate(
            segments=segments,
            ai_materials=ai_recommendations or {},
            product_name=product_name,
            product_context=product_context,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            constraints=constraints,
            similarity_threshold=self.config.rag.similarity_threshold,
            allowlist_strict=self.config.rag.allowlist_strict,
            allowlist_policy=self.config.rag.allowlist_policy,
            use_product_name_in_query=self.config.rag.use_product_name_in_query,
            nim_client=nim_client,
            nim_rerank_enabled=rerank_enabled,
            nim_rerank_temperature=self.config.rag.nim_rerank_temperature,
            nim_rerank_max_tokens=self.config.rag.nim_rerank_max_tokens,
        )

        with open(output_path, "w") as f:
            json.dump(curated, f, indent=2)

        return curated

    def _resolve_material_rag_paths(self) -> Tuple[List[Path], Path]:
        roots = [Path(root) for root in self.config.rag.material_roots]
        if not roots:
            raise RuntimeError(
                "ERROR: This is not supported/compatible - no material roots configured. "
                "Set PRODCONFIG_RAG__MATERIAL_ROOTS before running curation."
            )

        index_dir = Path(self.config.rag.index_dir)
        if not index_dir.is_absolute():
            index_dir = self.config.temp_path / index_dir

        return roots, index_dir

    def process_sync(self, input_path: Path,
                    output_dir: Optional[Path] = None,
                    model_name: Optional[str] = None,
                    run_material_curation: bool = True,
                    apply_curated_materials: bool = True) -> PipelineResult:
        """
        Synchronous wrapper for process() method.
        Use this when not in an async context.
        """
        return asyncio.run(
            self.process(
                input_path,
                output_dir,
                model_name,
                run_material_curation=run_material_curation,
                apply_curated_materials=apply_curated_materials,
            )
        )


class BatchProcessor:
    """
    Process multiple models in batch.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.pipeline = ProductConfiguratorPipeline(config)

    async def process_batch(self, input_paths: List[Path],
                           output_dir: Optional[Path] = None) -> List[PipelineResult]:
        """
        Process multiple models.

        Args:
            input_paths: List of input model paths
            output_dir: Base output directory

        Returns:
            List of PipelineResult for each input
        """
        output_dir = Path(output_dir) if output_dir else self.config.output_path
        results = []

        logger.info(f"Starting batch processing of {len(input_paths)} models")

        for i, input_path in enumerate(input_paths):
            logger.info(f"\nProcessing {i+1}/{len(input_paths)}: {input_path.name}")

            # Create subdirectory for each model
            model_output_dir = output_dir / input_path.stem

            result = await self.pipeline.process(
                input_path,
                model_output_dir,
                input_path.stem
            )
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"\nBatch complete: {successful}/{len(results)} successful")

        return results

    def process_batch_sync(self, input_paths: List[Path],
                          output_dir: Optional[Path] = None) -> List[PipelineResult]:
        """Synchronous batch processing"""
        return asyncio.run(self.process_batch(input_paths, output_dir))


def quick_process(input_path: str,
                 output_dir: str = "./output",
                 api_key: Optional[str] = None) -> PipelineResult:
    """
    Quick one-liner function for processing a single model.

    Usage:
        from src.step1.pipeline import quick_process
        result = quick_process("model.glb", "./output")

    Args:
        input_path: Path to input model
        output_dir: Output directory
        api_key: NIM API key for NIM services (optional)

    Returns:
        PipelineResult
    """
    config = AppConfig(
        output_path=Path(output_dir)
    )

    if api_key:
        config.nim.api_key = api_key

    pipeline = ProductConfiguratorPipeline(config)
    return pipeline.process_sync(Path(input_path))
