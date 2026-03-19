#!/usr/bin/env python3
"""CLI entry point for the photogrammetry pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="photogrammetry",
    help="Modular photogrammetry pipeline with AliceVision, MILo, and NeuRodin backends.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    input: Path = typer.Option(
        ..., "--input", "-i",
        help="Directory containing input images.",
        exists=True, file_okay=False,
    ),
    output: Path = typer.Option(
        "./output", "--output", "-o",
        help="Output directory for results.",
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset", "-p",
        help="Pipeline preset: classical, neural, hybrid, quality.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to a custom YAML config file.",
    ),
    set_param: Optional[list[str]] = typer.Option(
        None, "--set",
        help="Override config params (e.g. --set sfm.vggt.confidence_threshold=0.7).",
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level",
        help="Logging level: DEBUG, INFO, WARNING, ERROR.",
    ),
    gpu: int = typer.Option(
        0, "--gpu",
        help="GPU device ID to use.",
    ),
) -> None:
    """Run the photogrammetry pipeline."""
    from photogrammetry.pipeline.config import load_config
    from photogrammetry.pipeline.runner import PipelineRunner
    from photogrammetry.utils.logging import setup_logging

    setup_logging(level=log_level)

    overrides: dict = {
        "input_dir": str(input),
        "output_dir": str(output),
        "device": {"gpu_id": gpu},
    }

    if set_param:
        for param in set_param:
            key, _, value = param.partition("=")
            parts = key.strip().split(".")
            d = overrides
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            try:
                d[parts[-1]] = float(value) if "." in value else int(value)
            except ValueError:
                d[parts[-1]] = value

    cfg = load_config(
        config_path=config,
        preset=preset,
        overrides=overrides,
    )

    console.print(f"\n[bold]Photogrammetry Pipeline[/bold]")
    console.print(f"  Preset:  [cyan]{cfg.pipeline.preset}[/cyan]")
    console.print(f"  Input:   {input}")
    console.print(f"  Output:  {output}")
    console.print(f"  Surface: [green]{cfg.surface.backend}[/green]")
    console.print()

    runner = PipelineRunner(cfg)
    result = runner.run()

    console.print("\n[bold green]Pipeline complete![/bold green]")
    if result.mesh_path:
        console.print(f"  Mesh:        {result.mesh_path}")
    if result.point_cloud_path:
        console.print(f"  Point cloud: {result.point_cloud_path}")
    if result.texture_path:
        console.print(f"  Textures:    {result.texture_path}")
    console.print()


@app.command()
def info() -> None:
    """Show information about available presets and backends."""
    table = Table(title="Pipeline Presets")
    table.add_column("Preset", style="cyan")
    table.add_column("Features")
    table.add_column("SfM")
    table.add_column("Dense")
    table.add_column("Surface", style="green")
    table.add_column("Speed")

    table.add_row(
        "classical", "—", "AliceVision", "AliceVision", "AliceVision", "Fast"
    )
    table.add_row(
        "neural", "—", "VGGT", "VGGT", "MILo", "Minutes"
    )
    table.add_row(
        "hybrid", "SP+LG", "VGGT + COLMAP BA", "VGGT", "MILo", "~30 min"
    )
    table.add_row(
        "quality", "SP+LG", "VGGT + COLMAP BA", "—", "NeuRodin", "Hours"
    )

    console.print(table)
    console.print()

    backends = Table(title="Surface Reconstruction Backends")
    backends.add_column("Backend", style="cyan")
    backends.add_column("Paradigm")
    backends.add_column("Quality")
    backends.add_column("Paper")

    backends.add_row(
        "AliceVision", "Classical (Delaunay + Graph Cut)",
        "High", "Griwodz et al., MMSys 2021",
    )
    backends.add_row(
        "MILo", "Gaussian Splatting + Mesh-in-the-Loop",
        "High", "Guédon & Lepetit, SIGGRAPH Asia 2025",
    )
    backends.add_row(
        "NeuRodin", "Neural Implicit (SDF + Density)",
        "Highest", "Wang et al., NeurIPS 2024",
    )

    console.print(backends)


@app.command()
def export(
    mesh: Path = typer.Option(
        ..., "--mesh", "-m",
        help="Path to the mesh file to export.",
        exists=True,
    ),
    output: Path = typer.Option(
        "./export", "--output", "-o",
        help="Output directory.",
    ),
    formats: str = typer.Option(
        "ply,obj", "--formats", "-f",
        help="Comma-separated list of formats: ply, obj, glb, stl.",
    ),
) -> None:
    """Export a mesh to various formats."""
    from photogrammetry.export.formats import export_mesh

    fmt_list = [f.strip() for f in formats.split(",")]
    results = export_mesh(mesh, output, formats=fmt_list)

    for fmt, path in results.items():
        console.print(f"  [green]{fmt}[/green]: {path}")


if __name__ == "__main__":
    app()
