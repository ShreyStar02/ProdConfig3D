"""
Command Line Interface for Product Configurator Pipeline.
"""
import sys
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    typer = None

from .config import AppConfig, SegmentationMethod
from .step2.material_rag import MaterialRAGCurator
from .common.nim_integration import NIMPipeline, SegmentAnalysis
from .step1.pipeline import ProductConfiguratorPipeline, BatchProcessor, quick_process
from .step3.usd_pipeline import ModelToUSDConverter, USDExporter, USDImporter
from .common.nim_cli import apply_nim_cli_overrides
from .common.nim_probe import probe_nim_endpoint as _probe_nim_endpoint
from .step1.runner import run_mesh_to_multimesh as _run_mesh_to_multimesh
from .step2.runner import run_multimesh_to_mdl_json as _run_multimesh_to_mdl_json
from .step3.runner import run_curate_multimesh as _run_curate_multimesh
from .common.step_combos import (
    run_step12 as _run_step12,
    run_step23 as _run_step23,
    run_step123 as _run_step123,
)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

if RICH_AVAILABLE:
    app = typer.Typer(
        name="prodconfig",
        help="Universal 3D Product Configurator - Single mesh to Multi-mesh with curated MDL materials",
        add_completion=False
    )
    console = Console()


def create_cli():
    """Create and return the CLI app"""
    if not RICH_AVAILABLE:
        print("ERROR: This is not supported/compatible - missing required CLI dependencies (typer, rich).")
        print("Install dependencies using requirements.txt via setup.sh.")
        sys.exit(1)

    @app.command()
    def process(
        input_path: Path = typer.Argument(..., help="Input 3D model file (GLB, GLTF, OBJ, etc.)"),
        output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
        model_name: Optional[str] = typer.Option(None, "--name", "-n", help="Model name for output"),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY",
            help="NIM API key for NIM services"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url", help="Override NIM base URL"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile", help="NIM profile: cloud|local|custom"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode", help="NIM auth mode: auto|required|none"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency", help="Max concurrent NIM requests"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries", help="Max NIM retries for transient failures"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff", help="Base retry backoff seconds"),
        target_segments: Optional[int] = typer.Option(None, "--segments", "-s",
            help="Target number of segments"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    ):
        """
        Process a single 3D model.

        Converts a single-mesh 3D model to a multi-mesh USD file and
        curates MDL material references for each mesh part.
        """
        if not input_path.exists():
            console.print(f"[red]Error: Input file not found: {input_path}[/red]")
            raise typer.Exit(1)

        # Configure
        config = AppConfig(
            output_path=output_dir or Path("./output"),
            verbose=verbose
        )

        if api_key:
            config.nim.api_key = api_key

        _apply_nim_cli_overrides(
            config,
            nim_base_url=nim_base_url,
            nim_profile=nim_profile,
            nim_auth_mode=nim_auth_mode,
            nim_max_concurrency=nim_max_concurrency,
            nim_max_retries=nim_max_retries,
            nim_retry_backoff=nim_retry_backoff,
        )

        if target_segments:
            config.segmentation.target_num_segments = target_segments

        # Display configuration
        console.print(Panel.fit(
            f"[bold]Input:[/bold] {input_path}\n"
            f"[bold]Output:[/bold] {config.output_path}\n"
            f"[bold]Product Name:[/bold] {model_name or input_path.stem}\n"
            f"[bold]NIM Enabled:[/bold] {'Yes' if config.nim.is_configured_for_inference() else 'No'}\n"
            f"[bold]NIM Base URL:[/bold] {config.nim.base_url}\n"
            f"[bold]NIM Profile:[/bold] {config.nim.profile}\n"
            f"[bold]NIM Auth Mode:[/bold] {config.nim.auth_mode}",
            title="Product Configurator Pipeline",
            border_style="blue"
        ))

        # Process
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Starting pipeline...", total=None)

            class _ProgressLogHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    message = record.getMessage()
                    for line in message.splitlines():
                        match = re.search(r"\[Step \d+/\d+\].*", line)
                        if match:
                            progress.update(task, description=match.group(0))
                            return

            handler = _ProgressLogHandler()
            handler.setLevel(logging.INFO)
            root_logger = logging.getLogger()
            pipeline_logger = logging.getLogger("src.pipeline")

            root_logger.addHandler(handler)
            pipeline_logger.addHandler(handler)
            pipeline_logger.setLevel(logging.INFO)
            pipeline_logger.propagate = True

            try:
                pipeline = ProductConfiguratorPipeline(config)
                result = pipeline.process_sync(input_path, output_dir, model_name)
            finally:
                root_logger.removeHandler(handler)
                pipeline_logger.removeHandler(handler)

            progress.update(task, completed=True)

        # Display results
        if result.success:
            console.print("\n[green bold]Pipeline completed successfully![/green bold]\n")

            # Segments table
            table = Table(title="Mesh Segments")
            table.add_column("ID", style="cyan")
            table.add_column("Label", style="green")
            table.add_column("Faces", justify="right")
            table.add_column("Confidence", justify="right")

            for seg in result.segments or []:
                table.add_row(
                    str(seg.segment_id),
                    seg.label,
                    str(seg.properties.get('face_count', 0)),
                    f"{seg.confidence:.2f}"
                )

            console.print(table)

            # Output paths
            console.print(f"\n[bold]Output Files:[/bold]")
            console.print(f"  Multi-mesh USD: [cyan]{result.multi_mesh_usd_path}[/cyan]")
            curated_path = result.metadata.get("curated_materials_path")
            if curated_path:
                console.print(f"  Curated materials: [cyan]{curated_path}[/cyan]")
            console.print(f"\n[dim]Processing time: {result.processing_time_seconds:.2f}s[/dim]")

        else:
            console.print("\n[red bold]Pipeline failed![/red bold]")
            for error in result.errors:
                console.print(f"  [red]- {error}[/red]")
            raise typer.Exit(1)

    @app.command("mesh-to-multimesh")
    def mesh_to_multimesh(
        source: Path = typer.Option(..., "--source", help="Input model file (GLB/GLTF/OBJ/FBX/USD family)"),
        dest: Path = typer.Option(..., "--dest", help="Output directory for multi-mesh artifacts"),
        name: Optional[str] = typer.Option(None, "--name", help="Output model name"),
        segmentation_method: SegmentationMethod = typer.Option(
            SegmentationMethod.SAM3_ZEROSHOT,
            "--segmentation-method",
            help="Segmentation method",
        ),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url", help="Override NIM base URL"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile", help="NIM profile: cloud|local|custom"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode", help="NIM auth mode: auto|required|none"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency", help="Max concurrent NIM requests"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries", help="Max NIM retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff", help="Base retry backoff seconds"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    ):
        """Step 1: single mesh to multi-mesh USD/GLB (no material curation)."""
        if not source.exists():
            console.print(f"[red]Error: Input file not found: {source}[/red]")
            raise typer.Exit(1)

        result = _run_mesh_to_multimesh(
            source=source,
            dest=dest,
            name=name,
            segmentation_method=segmentation_method,
            api_key=api_key,
            nim_base_url=nim_base_url,
            nim_profile=nim_profile,
            nim_auth_mode=nim_auth_mode,
            nim_max_concurrency=nim_max_concurrency,
            nim_max_retries=nim_max_retries,
            nim_retry_backoff=nim_retry_backoff,
            verbose=verbose,
        )

        if not result.success:
            for error in result.errors:
                console.print(f"[red]- {error}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]USD:[/green] {result.multi_mesh_usd_path}")
        console.print(f"[green]GLB:[/green] {result.multi_mesh_glb_path}")

    @app.command("nim-smoke")
    def nim_smoke(
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY", help="NIM API key"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url", help="Override NIM base URL"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile", help="NIM profile: cloud|local|custom"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode", help="NIM auth mode: auto|required|none"),
        timeout_seconds: float = typer.Option(8.0, "--timeout", help="Per-request timeout in seconds"),
    ):
        """Probe NIM endpoint readiness without running full inference."""
        if not HTTPX_AVAILABLE:
            console.print("[red]ERROR: This is not supported/compatible - missing required dependency: httpx. Install dependencies using requirements.txt via setup.sh.[/red]")
            raise typer.Exit(1)

        config = AppConfig()
        if api_key:
            config.nim.api_key = api_key
        _apply_nim_cli_overrides(
            config,
            nim_base_url=nim_base_url,
            nim_profile=nim_profile,
            nim_auth_mode=nim_auth_mode,
            nim_max_concurrency=None,
            nim_max_retries=None,
            nim_retry_backoff=None,
        )

        if config.nim.auth_mode == "required" and not config.nim.api_key:
            console.print("[red]NIM auth mode is 'required' but no API key is configured.[/red]")
            raise typer.Exit(1)

        headers = {"Accept": "application/json"}
        if config.nim.should_send_auth_header() and config.nim.api_key:
            headers["Authorization"] = f"Bearer {config.nim.api_key}"

        result = asyncio.run(_probe_nim_endpoint(config.nim.base_url, headers, timeout_seconds))

        table = Table(title="NIM Smoke Check")
        table.add_column("Setting")
        table.add_column("Value")
        table.add_row("Profile", str(config.nim.profile))
        table.add_row("Base URL", str(config.nim.base_url))
        table.add_row("Auth Mode", str(config.nim.auth_mode))
        table.add_row("Auth Header", "sent" if "Authorization" in headers else "not sent")
        table.add_row("Probe Path", result.get("path", "-"))
        table.add_row("HTTP Status", str(result.get("status", "-")))
        table.add_row("Reachable", "yes" if result.get("reachable") else "no")
        console.print(table)

        if result.get("reachable"):
            console.print("[green]NIM endpoint is reachable.[/green]")
            return

        console.print(f"[red]NIM probe failed: {result.get('error', 'unknown error')}[/red]")
        raise typer.Exit(1)

    @app.command("step1")
    def step1_alias(
        source: Path = typer.Option(..., "--source", help="Input model file"),
        dest: Path = typer.Option(..., "--dest", help="Output directory"),
        name: Optional[str] = typer.Option(None, "--name", help="Output model name"),
        segmentation_method: SegmentationMethod = typer.Option(SegmentationMethod.SAM3_ZEROSHOT, "--segmentation-method"),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """Alias for mesh-to-multimesh."""
        result = _run_mesh_to_multimesh(
            source=source,
            dest=dest,
            name=name,
            segmentation_method=segmentation_method,
            api_key=api_key,
            nim_base_url=nim_base_url,
            nim_profile=nim_profile,
            nim_auth_mode=nim_auth_mode,
            nim_max_concurrency=nim_max_concurrency,
            nim_max_retries=nim_max_retries,
            nim_retry_backoff=nim_retry_backoff,
            verbose=verbose,
        )
        if not result.success:
            for error in result.errors:
                console.print(f"[red]- {error}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]USD:[/green] {result.multi_mesh_usd_path}")
        console.print(f"[green]GLB:[/green] {result.multi_mesh_glb_path}")

    @app.command("multimesh-to-mdl-json")
    def multimesh_to_mdl_json(
        source: Path = typer.Option(..., "--source", help="Input multi-mesh model (USD/GLB/etc.)"),
        dest: Path = typer.Option(..., "--dest", help="Output materials JSON path"),
        name: Optional[str] = typer.Option(None, "--name", help="Product name override"),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        top_k: Optional[int] = typer.Option(None, "--top-k", help="Top materials per segment"),
        nim_rerank: Optional[bool] = typer.Option(None, "--nim-rerank/--no-nim-rerank", help="Enable/disable NIM rerank"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url", help="Override NIM base URL"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile", help="NIM profile: cloud|local|custom"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode", help="NIM auth mode: auto|required|none"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency", help="Max concurrent NIM requests"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries", help="Max NIM retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff", help="Base retry backoff seconds"),
    ):
        """Step 2: multi-mesh model to curated MDL JSON."""
        if not source.exists():
            console.print(f"[red]Error: Input file not found: {source}[/red]")
            raise typer.Exit(1)

        output = asyncio.run(
            _run_multimesh_to_mdl_json(
                source=source,
                dest=dest,
                name=name,
                api_key=api_key,
                top_k=top_k,
                nim_rerank=nim_rerank,
                nim_base_url=nim_base_url,
                nim_profile=nim_profile,
                nim_auth_mode=nim_auth_mode,
                nim_max_concurrency=nim_max_concurrency,
                nim_max_retries=nim_max_retries,
                nim_retry_backoff=nim_retry_backoff,
            )
        )
        console.print(f"[green]MDL JSON:[/green] {output}")

    @app.command("step2")
    def step2_alias(
        source: Path = typer.Option(..., "--source"),
        dest: Path = typer.Option(..., "--dest"),
        name: Optional[str] = typer.Option(None, "--name"),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        top_k: Optional[int] = typer.Option(None, "--top-k"),
        nim_rerank: Optional[bool] = typer.Option(None, "--nim-rerank/--no-nim-rerank"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff"),
    ):
        """Alias for multimesh-to-mdl-json."""
        output = asyncio.run(
            _run_multimesh_to_mdl_json(
                source=source,
                dest=dest,
                name=name,
                api_key=api_key,
                top_k=top_k,
                nim_rerank=nim_rerank,
                nim_base_url=nim_base_url,
                nim_profile=nim_profile,
                nim_auth_mode=nim_auth_mode,
                nim_max_concurrency=nim_max_concurrency,
                nim_max_retries=nim_max_retries,
                nim_retry_backoff=nim_retry_backoff,
            )
        )
        console.print(f"[green]MDL JSON:[/green] {output}")

    @app.command("curate-multimesh")
    def curate_multimesh(
        source: Path = typer.Option(..., "--source", help="Input multi-mesh model (USD/GLB/etc.)"),
        materials_json: Path = typer.Option(..., "--materials-json", help="Curated materials JSON path"),
        dest: Path = typer.Option(..., "--dest", help="Output directory"),
        name: Optional[str] = typer.Option(None, "--name", help="Output model name"),
    ):
        """Step 3: apply curated MDL JSON to multi-mesh model and export bound USD/GLB."""
        usd_out, glb_out = _run_curate_multimesh(source, materials_json, dest, name)
        console.print(f"[green]Bound USD:[/green] {usd_out}")
        console.print(f"[green]Bound GLB:[/green] {glb_out}")

    @app.command("step3")
    def step3_alias(
        source: Path = typer.Option(..., "--source"),
        materials_json: Path = typer.Option(..., "--materials-json"),
        dest: Path = typer.Option(..., "--dest"),
        name: Optional[str] = typer.Option(None, "--name"),
    ):
        """Alias for curate-multimesh."""
        usd_out, glb_out = _run_curate_multimesh(source, materials_json, dest, name)
        console.print(f"[green]Bound USD:[/green] {usd_out}")
        console.print(f"[green]Bound GLB:[/green] {glb_out}")

    @app.command("mesh-to-mdl-json")
    def mesh_to_mdl_json(
        source: Path = typer.Option(..., "--source", help="Input model file"),
        dest: Path = typer.Option(..., "--dest", help="Output curated materials JSON path"),
        name: Optional[str] = typer.Option(None, "--name", help="Output model name"),
        segmentation_method: SegmentationMethod = typer.Option(
            SegmentationMethod.SAM3_ZEROSHOT,
            "--segmentation-method",
            help="Segmentation method",
        ),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        top_k: Optional[int] = typer.Option(None, "--top-k", help="Top materials per segment"),
        nim_rerank: Optional[bool] = typer.Option(None, "--nim-rerank/--no-nim-rerank", help="Enable/disable NIM rerank"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url", help="Override NIM base URL"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile", help="NIM profile: cloud|local|custom"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode", help="NIM auth mode: auto|required|none"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency", help="Max concurrent NIM requests"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries", help="Max NIM retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff", help="Base retry backoff seconds"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    ):
        """Run step1 then step2: single mesh -> curated MDL JSON."""
        output_json = asyncio.run(
            _run_step12(
                source=source,
                dest=dest,
                name=name,
                segmentation_method=segmentation_method,
                api_key=api_key,
                top_k=top_k,
                nim_rerank=nim_rerank,
                nim_base_url=nim_base_url,
                nim_profile=nim_profile,
                nim_auth_mode=nim_auth_mode,
                nim_max_concurrency=nim_max_concurrency,
                nim_max_retries=nim_max_retries,
                nim_retry_backoff=nim_retry_backoff,
                verbose=verbose,
            )
        )
        console.print(f"[green]MDL JSON:[/green] {output_json}")

    @app.command("step12")
    def step12_alias(
        source: Path = typer.Option(..., "--source"),
        dest: Path = typer.Option(..., "--dest"),
        name: Optional[str] = typer.Option(None, "--name"),
        segmentation_method: SegmentationMethod = typer.Option(SegmentationMethod.SAM3_ZEROSHOT, "--segmentation-method"),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        top_k: Optional[int] = typer.Option(None, "--top-k"),
        nim_rerank: Optional[bool] = typer.Option(None, "--nim-rerank/--no-nim-rerank"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """Alias for mesh-to-mdl-json."""
        output_json = asyncio.run(
            _run_step12(
                source=source,
                dest=dest,
                name=name,
                segmentation_method=segmentation_method,
                api_key=api_key,
                top_k=top_k,
                nim_rerank=nim_rerank,
                nim_base_url=nim_base_url,
                nim_profile=nim_profile,
                nim_auth_mode=nim_auth_mode,
                nim_max_concurrency=nim_max_concurrency,
                nim_max_retries=nim_max_retries,
                nim_retry_backoff=nim_retry_backoff,
                verbose=verbose,
            )
        )
        console.print(f"[green]MDL JSON:[/green] {output_json}")

    @app.command("mdl-json-to-bound-multimesh")
    def mdl_json_to_bound_multimesh(
        source: Path = typer.Option(..., "--source", help="Input multi-mesh model (USD/GLB/etc.)"),
        dest: Path = typer.Option(..., "--dest", help="Output directory"),
        name: Optional[str] = typer.Option(None, "--name", help="Output model name"),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        top_k: Optional[int] = typer.Option(None, "--top-k", help="Top materials per segment"),
        nim_rerank: Optional[bool] = typer.Option(None, "--nim-rerank/--no-nim-rerank", help="Enable/disable NIM rerank"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url", help="Override NIM base URL"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile", help="NIM profile: cloud|local|custom"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode", help="NIM auth mode: auto|required|none"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency", help="Max concurrent NIM requests"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries", help="Max NIM retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff", help="Base retry backoff seconds"),
    ):
        """Run step2 then step3: multi-mesh -> curated JSON -> bound USD/GLB."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[Step 1/3] Starting step23...", total=None)

                def _progress_cb(message: str) -> None:
                    progress.update(task, description=message)

                materials_json, usd_out, glb_out = asyncio.run(
                    _run_step23(
                        source=source,
                        dest=dest,
                        name=name,
                        api_key=api_key,
                        top_k=top_k,
                        nim_rerank=nim_rerank,
                        nim_base_url=nim_base_url,
                        nim_profile=nim_profile,
                        nim_auth_mode=nim_auth_mode,
                        nim_max_concurrency=nim_max_concurrency,
                        nim_max_retries=nim_max_retries,
                        nim_retry_backoff=nim_retry_backoff,
                        progress_cb=_progress_cb,
                    )
                )
                progress.update(task, description="Step23 completed", completed=True)
        except Exception as exc:
            console.print(f"[red]step23 failed:[/red] {exc}")
            raise typer.Exit(1)
        console.print(f"[green]MDL JSON:[/green] {materials_json}")
        console.print(f"[green]Bound USD:[/green] {usd_out}")
        console.print(f"[green]Bound GLB:[/green] {glb_out}")

    @app.command("step23")
    def step23_alias(
        source: Path = typer.Option(..., "--source"),
        dest: Path = typer.Option(..., "--dest"),
        name: Optional[str] = typer.Option(None, "--name"),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        top_k: Optional[int] = typer.Option(None, "--top-k"),
        nim_rerank: Optional[bool] = typer.Option(None, "--nim-rerank/--no-nim-rerank"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff"),
    ):
        """Alias for mdl-json-to-bound-multimesh."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[Step 1/3] Starting step23...", total=None)

                def _progress_cb(message: str) -> None:
                    progress.update(task, description=message)

                materials_json, usd_out, glb_out = asyncio.run(
                    _run_step23(
                        source=source,
                        dest=dest,
                        name=name,
                        api_key=api_key,
                        top_k=top_k,
                        nim_rerank=nim_rerank,
                        nim_base_url=nim_base_url,
                        nim_profile=nim_profile,
                        nim_auth_mode=nim_auth_mode,
                        nim_max_concurrency=nim_max_concurrency,
                        nim_max_retries=nim_max_retries,
                        nim_retry_backoff=nim_retry_backoff,
                        progress_cb=_progress_cb,
                    )
                )
                progress.update(task, description="Step23 completed", completed=True)
        except Exception as exc:
            console.print(f"[red]step23 failed:[/red] {exc}")
            raise typer.Exit(1)
        console.print(f"[green]MDL JSON:[/green] {materials_json}")
        console.print(f"[green]Bound USD:[/green] {usd_out}")
        console.print(f"[green]Bound GLB:[/green] {glb_out}")

    @app.command("run-all")
    def run_all(
        source: Path = typer.Option(..., "--source", help="Input model file"),
        dest: Path = typer.Option(..., "--dest", help="Output directory"),
        name: Optional[str] = typer.Option(None, "--name", help="Output model name"),
        segmentation_method: SegmentationMethod = typer.Option(
            SegmentationMethod.SAM3_ZEROSHOT,
            "--segmentation-method",
            help="Segmentation method",
        ),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        top_k: Optional[int] = typer.Option(None, "--top-k", help="Top materials per segment"),
        nim_rerank: Optional[bool] = typer.Option(None, "--nim-rerank/--no-nim-rerank", help="Enable/disable NIM rerank"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url", help="Override NIM base URL"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile", help="NIM profile: cloud|local|custom"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode", help="NIM auth mode: auto|required|none"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency", help="Max concurrent NIM requests"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries", help="Max NIM retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff", help="Base retry backoff seconds"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    ):
        """Run step1 then step2 then step3 end-to-end."""
        materials_json, usd_out, glb_out = asyncio.run(
            _run_step123(
                source=source,
                dest=dest,
                name=name,
                segmentation_method=segmentation_method,
                api_key=api_key,
                top_k=top_k,
                nim_rerank=nim_rerank,
                nim_base_url=nim_base_url,
                nim_profile=nim_profile,
                nim_auth_mode=nim_auth_mode,
                nim_max_concurrency=nim_max_concurrency,
                nim_max_retries=nim_max_retries,
                nim_retry_backoff=nim_retry_backoff,
                verbose=verbose,
            )
        )
        console.print(f"[green]MDL JSON:[/green] {materials_json}")
        console.print(f"[green]Bound USD:[/green] {usd_out}")
        console.print(f"[green]Bound GLB:[/green] {glb_out}")

    @app.command("step123")
    def step123_alias(
        source: Path = typer.Option(..., "--source"),
        dest: Path = typer.Option(..., "--dest"),
        name: Optional[str] = typer.Option(None, "--name"),
        segmentation_method: SegmentationMethod = typer.Option(SegmentationMethod.SAM3_ZEROSHOT, "--segmentation-method"),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        top_k: Optional[int] = typer.Option(None, "--top-k"),
        nim_rerank: Optional[bool] = typer.Option(None, "--nim-rerank/--no-nim-rerank"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """Alias for run-all."""
        materials_json, usd_out, glb_out = asyncio.run(
            _run_step123(
                source=source,
                dest=dest,
                name=name,
                segmentation_method=segmentation_method,
                api_key=api_key,
                top_k=top_k,
                nim_rerank=nim_rerank,
                nim_base_url=nim_base_url,
                nim_profile=nim_profile,
                nim_auth_mode=nim_auth_mode,
                nim_max_concurrency=nim_max_concurrency,
                nim_max_retries=nim_max_retries,
                nim_retry_backoff=nim_retry_backoff,
                verbose=verbose,
            )
        )
        console.print(f"[green]MDL JSON:[/green] {materials_json}")
        console.print(f"[green]Bound USD:[/green] {usd_out}")
        console.print(f"[green]Bound GLB:[/green] {glb_out}")

    @app.command()
    def batch(
        input_dir: Path = typer.Argument(..., help="Directory containing input models"),
        output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
        pattern: str = typer.Option("*.glb", "--pattern", "-p", help="File pattern to match"),
        api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="PRODCONFIG_NIM__API_KEY"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url", help="Override NIM base URL"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile", help="NIM profile: cloud|local|custom"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode", help="NIM auth mode: auto|required|none"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency", help="Max concurrent NIM requests"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries", help="Max NIM retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff", help="Base retry backoff seconds"),
    ):
        """
        Process multiple 3D models in batch.
        """
        if not input_dir.exists():
            console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
            raise typer.Exit(1)

        # Find input files
        input_files = list(input_dir.glob(pattern))
        if not input_files:
            console.print(f"[yellow]No files matching '{pattern}' found in {input_dir}[/yellow]")
            raise typer.Exit(0)

        console.print(f"Found {len(input_files)} files to process")

        # Configure
        config = AppConfig(
            output_path=output_dir or Path("./output")
        )

        if api_key:
            config.nim.api_key = api_key

        _apply_nim_cli_overrides(
            config,
            nim_base_url=nim_base_url,
            nim_profile=nim_profile,
            nim_auth_mode=nim_auth_mode,
            nim_max_concurrency=nim_max_concurrency,
            nim_max_retries=nim_max_retries,
            nim_retry_backoff=nim_retry_backoff,
        )

        # Process batch
        processor = BatchProcessor(config)

        with Progress(console=console) as progress:
            task = progress.add_task("Processing batch...", total=len(input_files))

            results = []
            for i, input_file in enumerate(input_files):
                progress.update(task, description=f"Processing {input_file.name}...")

                pipeline = ProductConfiguratorPipeline(config)
                result = pipeline.process_sync(
                    input_file,
                    (output_dir or Path("./output")) / input_file.stem
                )
                results.append(result)

                progress.advance(task)

        # Summary
        successful = sum(1 for r in results if r.success)
        console.print(f"\n[bold]Batch Complete:[/bold] {successful}/{len(results)} successful")

        # Results table
        table = Table(title="Batch Results")
        table.add_column("File", style="cyan")
        table.add_column("Status")
        table.add_column("Segments", justify="right")
        table.add_column("Time (s)", justify="right")

        for result in results:
            status = "[green]Success[/green]" if result.success else "[red]Failed[/red]"
            segments = str(len(result.segments)) if result.segments else "-"
            table.add_row(
                result.input_path.name,
                status,
                segments,
                f"{result.processing_time_seconds:.1f}"
            )

        console.print(table)

    @app.command("materials-rag")
    def materials_rag(
        input_path: Path = typer.Argument(..., help="Input result JSON or NIM material JSON"),
        output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output curated JSON path"),
        material_root: Optional[List[Path]] = typer.Option(
            None, "--root", "-r", help="Material root folder(s) to scan", show_default=False
        ),
        top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Top materials per segment"),
        rebuild: bool = typer.Option(False, "--rebuild", help="Force rebuild of material index"),
        nim_rerank: Optional[bool] = typer.Option(None, "--nim-rerank/--no-nim-rerank", help="Enable NIM rerank"),
        nim_base_url: Optional[str] = typer.Option(None, "--nim-base-url", help="Override NIM base URL"),
        nim_profile: Optional[str] = typer.Option(None, "--nim-profile", help="NIM profile: cloud|local|custom"),
        nim_auth_mode: Optional[str] = typer.Option(None, "--nim-auth-mode", help="NIM auth mode: auto|required|none"),
        nim_max_concurrency: Optional[int] = typer.Option(None, "--nim-max-concurrency", help="Max concurrent NIM requests"),
        nim_max_retries: Optional[int] = typer.Option(None, "--nim-max-retries", help="Max NIM retries"),
        nim_retry_backoff: Optional[float] = typer.Option(None, "--nim-retry-backoff", help="Base retry backoff seconds"),
    ):
        """
        Curate MDL materials from NIM JSON or pipeline results.
        """
        if not input_path.exists():
            console.print(f"[red]Error: Input file not found: {input_path}[/red]")
            raise typer.Exit(1)

        config = AppConfig()
        _apply_nim_cli_overrides(
            config,
            nim_base_url=nim_base_url,
            nim_profile=nim_profile,
            nim_auth_mode=nim_auth_mode,
            nim_max_concurrency=nim_max_concurrency,
            nim_max_retries=nim_max_retries,
            nim_retry_backoff=nim_retry_backoff,
        )
        roots = material_root or config.rag.material_roots
        if not roots:
            console.print(
                "[red]ERROR: This is not supported/compatible - no material roots configured. "
                "Set `PRODCONFIG_RAG__MATERIAL_ROOTS` or pass --root.[/red]"
            )
            raise typer.Exit(1)

        index_dir = config.rag.index_dir
        if not index_dir.is_absolute():
            index_dir = config.temp_path / index_dir

        with open(input_path, "r") as f:
            payload = json.load(f)

        segments, ai_materials, product_name = _extract_material_inputs(payload)

        resolved_top_k = top_k or config.rag.top_k
        if resolved_top_k is None:
            console.print("[red]Error: PRODCONFIG_RAG__TOP_K is required[/red]")
            raise typer.Exit(1)

        candidate_pool_size = config.rag.candidate_pool_size or resolved_top_k
        constraints = {}
        if config.rag.roughness_tolerance is not None:
            constraints["roughness_tolerance"] = config.rag.roughness_tolerance
        if config.rag.metallic_tolerance is not None:
            constraints["metallic_tolerance"] = config.rag.metallic_tolerance
        if config.rag.opacity_tolerance is not None:
            constraints["opacity_tolerance"] = config.rag.opacity_tolerance

        rerank_enabled = config.rag.nim_rerank_enabled if nim_rerank is None else nim_rerank
        if rerank_enabled is None:
            rerank_enabled = bool(config.nim.api_key)
        nim_client = None
        nim_pipeline = None
        if rerank_enabled and config.nim.api_key:
            nim_pipeline = NIMPipeline(config.nim)
            nim_client = nim_pipeline.nim_client

        curator = MaterialRAGCurator(roots, index_dir)
        curated = asyncio.run(curator.curate(
            segments=segments,
            ai_materials=ai_materials,
            product_name=product_name,
            top_k=resolved_top_k,
            candidate_pool_size=candidate_pool_size,
            constraints=constraints,
            similarity_threshold=config.rag.similarity_threshold,
            allowlist_strict=config.rag.allowlist_strict,
            nim_client=nim_client,
            nim_rerank_enabled=rerank_enabled,
            nim_rerank_temperature=config.rag.nim_rerank_temperature,
            nim_rerank_max_tokens=config.rag.nim_rerank_max_tokens,
            force_rebuild=rebuild,
        ))

        output_path = output_path or input_path.with_name(f"{input_path.stem}_materials.json")
        with open(output_path, "w") as f:
            json.dump(curated, f, indent=2)

        if nim_pipeline:
            asyncio.run(nim_pipeline.close())

        console.print(f"[green]Curated materials written:[/green] {output_path}")

    @app.command()
    def info():
        """
        Display system information and available features.
        """
        from . import __version__

        console.print(Panel.fit(
            f"[bold]Product Configurator Pipeline[/bold]\n"
            f"Version: {__version__}",
            border_style="blue"
        ))

        # Check dependencies
        deps = []

        try:
            import trimesh
            deps.append(("trimesh", trimesh.__version__, "[green]OK[/green]"))
        except ImportError:
            deps.append(("trimesh", "Not installed", "[red]MISSING[/red]"))

        try:
            import pymeshlab
            deps.append(("pymeshlab", "Installed", "[green]OK[/green]"))
        except ImportError:
            deps.append(("pymeshlab", "Not installed", "[yellow]Optional[/yellow]"))

        try:
            from pxr import Usd
            deps.append(("USD (pxr)", "Installed", "[green]OK[/green]"))
        except ImportError:
            deps.append(("USD (pxr)", "Not installed", "[red]REQUIRED[/red]"))

        try:
            import numpy
            deps.append(("numpy", numpy.__version__, "[green]OK[/green]"))
        except ImportError:
            deps.append(("numpy", "Not installed", "[red]MISSING[/red]"))

        try:
            from sklearn import __version__ as sk_version
            deps.append(("scikit-learn", sk_version, "[green]OK[/green]"))
        except ImportError:
            deps.append(("scikit-learn", "Not installed", "[red]REQUIRED[/red]"))

        try:
            from PIL import Image
            import PIL
            deps.append(("Pillow", PIL.__version__, "[green]OK[/green]"))
        except ImportError:
            deps.append(("Pillow", "Not installed", "[red]REQUIRED[/red]"))

        # Dependencies table
        table = Table(title="Dependencies")
        table.add_column("Package")
        table.add_column("Version")
        table.add_column("Status")

        for name, version, status in deps:
            table.add_row(name, version, status)

        console.print(table)

        # Product naming
        console.print("\n[bold]Product Naming:[/bold]")
        console.print("  - Uses input filename as the product name sent to NIM")

    return app


def _apply_nim_cli_overrides(
    config: AppConfig,
    nim_base_url: Optional[str],
    nim_profile: Optional[str],
    nim_auth_mode: Optional[str],
    nim_max_concurrency: Optional[int],
    nim_max_retries: Optional[int],
    nim_retry_backoff: Optional[float],
) -> None:
    apply_nim_cli_overrides(
        config=config,
        nim_base_url=nim_base_url,
        nim_profile=nim_profile,
        nim_auth_mode=nim_auth_mode,
        nim_max_concurrency=nim_max_concurrency,
        nim_max_retries=nim_max_retries,
        nim_retry_backoff=nim_retry_backoff,
    )


def main():
    """Main entry point for CLI"""
    if not RICH_AVAILABLE:
        print("CLI requires 'typer' and 'rich' packages.")
        print("ERROR: This is not supported/compatible - missing required CLI dependencies (typer, rich). Install from requirements.txt.")
        print("\nAlternatively, use the Python API:")
        print("  from src.step1.pipeline import quick_process")
        print("  result = quick_process('product.glb', './output')")
        sys.exit(1)

    cli = create_cli()
    cli()


def _extract_material_inputs(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """Extract segments and AI material recommendations from common payload shapes."""
    if "metadata" in payload and "segments" in payload:
        product_name = payload.get("metadata", {}).get("product_name") or payload.get("metadata", {}).get("model_name")
        segments = payload.get("segments", [])
        ai_materials = payload.get("metadata", {}).get("ai_material_recommendations", {})
        return segments, ai_materials, product_name or "generic"

    if "product_name" in payload and "segments" in payload:
        return payload.get("segments", []), payload.get("ai_material_recommendations", {}), payload.get("product_name")

    if "parts" in payload:
        segments = []
        ai_materials = {}
        for idx, part in enumerate(payload.get("parts", [])):
            label = part.get("part_name", f"part_{idx}")
            segments.append({"id": idx, "label": label, "part_type": part.get("part_type", "unknown")})
            if "materials" in part:
                ai_materials[label] = part.get("materials", {})
        return segments, ai_materials, payload.get("product_name", "generic")

    return [], {}, payload.get("product_name", "generic")


if __name__ == "__main__":
    main()

