# Photogrammetry Pipeline

A modular, quality-focused photogrammetry pipeline with three best-in-class surface
reconstruction backends, YAML-driven configuration, and a unified CLI.

## Overview

This project provides a complete photogrammetry pipeline — from input images to
textured 3D meshes — with swappable backends at every stage. It is optimized for
**mesh geometric quality** and supports classical, Gaussian-based, and neural implicit
reconstruction methods under a single Python 3.10 environment.

### Surface Reconstruction Backends

| Backend | Paradigm | Quality | Speed | Key Paper |
|---------|----------|---------|-------|-----------|
| **AliceVision/Meshroom** | Classical (Delaunay + Graph Cut) | High | Fast (no training) | Griwodz et al., MMSys 2021 |
| **MILo** | Gaussian Splatting + Mesh-in-the-Loop | High | Minutes | Guédon & Lepetit, SIGGRAPH Asia 2025 |
| **NeuRodin** | Neural Implicit (SDF + Density) | Highest | Hours | Wang et al., NeurIPS 2024 |

### Pipeline Presets

| Preset | Pipeline | Best For |
|--------|----------|----------|
| `classical` | AliceVision end-to-end | Quick, reliable results with full texturing |
| `neural` | VGGT → MILo | Rapid iteration (minutes) |
| `hybrid` | SuperPoint+LightGlue → VGGT+COLMAP BA → MILo | Balanced quality and speed |
| `quality` | SuperPoint+LightGlue → VGGT+COLMAP BA → NeuRodin | Maximum geometric accuracy |

## Architecture

```
                         ┌─────────────────────┐
                         │    Input Images      │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │     Preset Router     │
                         └──┬─────┬──────────┬──┘
                            │     │          │
              ┌─────────────▼─┐ ┌─▼────────┐ ┌▼───────────────┐
              │  classical    │ │ neural /  │ │    quality      │
              │               │ │ hybrid    │ │                 │
              │ AliceVision   │ │           │ │ SP+LG → VGGT   │
              │ (full pipe)   │ │ SP+LG →   │ │ + COLMAP BA    │
              │               │ │ VGGT →    │ │       ↓         │
              │ Depth maps    │ │ VGGT      │ │ NeuRodin        │
              │ Delaunay mesh │ │ dense →   │ │ (two-stage      │
              │ Graph cut     │ │ MILo      │ │  SDF+density)   │
              │ Texturing     │ │           │ │                 │
              └──────┬────────┘ └────┬──────┘ └───────┬─────────┘
                     │               │                │
                     └───────────────┼────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Export (PLY/OBJ/GLB) │
                          └──────────────────────┘
```

### Stage Breakdown (Modular Mode)

1. **Preprocessing** — Image loading, EXIF extraction, resizing, optional masking
2. **Feature Extraction & Matching** — SuperPoint + LightGlue (learned features, 8x faster than dense matchers)
3. **Structure from Motion** — VGGT feed-forward (CVPR 2025 Best Paper) with optional COLMAP bundle adjustment refinement
4. **Dense Reconstruction** — VGGT dense depth maps and 3D point clouds
5. **Surface Reconstruction** — MILo (Gaussian + mesh) or NeuRodin (neural implicit SDF)
6. **Export** — PLY, OBJ, GLB formats via trimesh

### Classical Mode

When the `classical` preset is selected, AliceVision/Meshroom handles the entire
pipeline end-to-end: feature extraction, SfM, depth map estimation, depth filtering,
3D Delaunay tetrahedralization, graph-cut surface extraction, mesh filtering/denoising,
and multi-band UV texturing.

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04 WSL2)
- NVIDIA GPU with CUDA 12.1+ support
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/)
- Git

### 1. Create the Conda Environment

```bash
cd photogrammetry
conda env create -f environment.yml
conda activate photogrammetry
```

### 2. Install Pip Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the Package

```bash
pip install -e .
```

### 4. Install External Tools

#### AliceVision/Meshroom (Classical Backend)

AliceVision is already available in the workspace at `../AliceVision/`. To use it,
either build from source or install a binary release:

```bash
# Option A: Use pre-built binaries
# Download from https://github.com/alicevision/Meshroom/releases

# Option B: Build from source (already cloned)
cd ../AliceVision
# Follow INSTALL.md for build instructions
```

Ensure `meshroom_batch` is on your PATH, or set the `MESHROOM_BIN` environment
variable to point to the Meshroom binary directory.

#### MILo (Gaussian Backend)

```bash
git clone https://github.com/Anttwo/MILo.git third_party/milo
cd third_party/milo
pip install -e .
cd ../..
```

#### NeuRodin / SDFStudio (Neural Implicit Backend)

SDFStudio is vendored in `third_party/sdfstudio/` and patched for Python 3.10 +
PyTorch 2.4 compatibility. Install it as an editable package:

```bash
cd third_party/sdfstudio
pip install -e .
ns-install-cli
cd ../..
```

You also need tiny-cuda-nn v2.0+:

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

See `third_party/sdfstudio/README_PATCHES.md` for details on patches applied.

## Usage

### CLI

The pipeline is controlled via a single CLI entry point:

```bash
# Run with a preset
python scripts/run.py --preset classical --input ./data/my_scene/images --output ./output/my_scene

# Run with the neural preset (fast, VGGT + MILo)
python scripts/run.py --preset neural --input ./data/my_scene/images --output ./output/my_scene

# Run with the hybrid preset (balanced quality)
python scripts/run.py --preset hybrid --input ./data/my_scene/images --output ./output/my_scene

# Run with the quality preset (NeuRodin, highest quality)
python scripts/run.py --preset quality --input ./data/my_scene/images --output ./output/my_scene

# Use a custom config file
python scripts/run.py --config configs/custom.yaml --input ./data/my_scene/images --output ./output/my_scene

# Override specific parameters
python scripts/run.py --preset hybrid --input ./data/images --output ./output \
    --set sfm.vggt.confidence_threshold=0.7 \
    --set surface.milo.iterations=30000
```

### Python API

```python
from photogrammetry.pipeline.config import load_config
from photogrammetry.pipeline.runner import PipelineRunner

config = load_config("configs/hybrid.yaml", overrides={
    "input_dir": "./data/my_scene/images",
    "output_dir": "./output/my_scene",
})

runner = PipelineRunner(config)
result = runner.run()

print(f"Mesh saved to: {result.mesh_path}")
print(f"Point cloud: {result.point_cloud_path}")
```

## Configuration Reference

Configurations use YAML files processed by OmegaConf. Each preset overrides the
`configs/default.yaml` base configuration.

### Key Configuration Sections

```yaml
# Pipeline selection
pipeline:
  preset: hybrid                    # classical | neural | hybrid | quality

# Preprocessing
preprocessing:
  max_image_size: 1600              # Resize longest edge (0 = no resize)
  mask_dir: null                    # Optional mask directory

# Feature extraction and matching
features:
  backend: superpoint_lightglue     # superpoint_lightglue
  superpoint:
    max_num_keypoints: 4096
  lightglue:
    depth_confidence: 0.95
    width_confidence: 0.99

# Structure from Motion
sfm:
  backend: vggt                     # vggt | colmap
  vggt:
    model: facebook/VGGT-1B
    confidence_threshold: 0.5
  colmap:
    bundle_adjustment: true         # Refine VGGT poses with COLMAP BA

# Dense reconstruction
dense:
  backend: vggt                     # vggt
  vggt:
    use_depth_maps: true
    use_point_maps: true
    point_confidence_threshold: 0.5

# Surface reconstruction
surface:
  backend: milo                     # alicevision | milo | neurodin
  alicevision:
    meshroom_bin: null              # Auto-detect from PATH or MESHROOM_BIN
    pipeline: photogrammetry        # photogrammetry | object | turntable
  milo:
    iterations: 15000
    imp_metric: indoor              # indoor | outdoor
    rasterizer: radegs
  neurodin:
    stage1_config: indoor-small     # indoor-small | indoor-large | outdoor-large
    stage2_config: indoor-small
    resolution: 2048                # Mesh extraction resolution

# Export
export:
  formats:
    - ply
    - obj
  include_textures: true
  include_point_cloud: true
```

## Backend Details

### AliceVision/Meshroom

- **Version**: v2025.1.0+
- **Pipeline**: Feature extraction → SfM → Depth maps → Depth filtering → Meshing (Delaunay + Graph Cut) → Mesh filtering → Texturing
- **Strengths**: Production-proven, full UV texturing, specialized object/turntable pipelines
- **Interface**: Wrapped via `meshroom_batch` CLI with parameter overrides

### MILo (Mesh-in-the-Loop Gaussian Splatting)

- **Paper**: SIGGRAPH Asia 2025 (ACM Transactions on Graphics)
- **Author**: Same as SuGaR (Antoine Guédon)
- **Key innovation**: Differentiable mesh extraction at every training iteration
- **Input**: COLMAP-format scene (images + sparse reconstruction)
- **Output**: Lightweight mesh (60-400MB), order of magnitude fewer vertices than GOF/RaDe-GS
- **Interface**: Python API wrapping MILo's `train.py` and `mesh_extract_sdf.py`

### NeuRodin (Neural Implicit Surfaces)

- **Paper**: NeurIPS 2024
- **Key innovation**: Two-stage SDF + density framework surpassing Neuralangelo
- **Built on**: SDFStudio (vendored and patched for PyTorch 2.4)
- **Input**: Posed images in transforms.json or COLMAP format
- **Output**: High-fidelity watertight mesh via Marching Cubes
- **Interface**: Python API wrapping `ns-train` with NeuRodin configs

## Project Structure

```
photogrammetry/
├── README.md                          # This file
├── environment.yml                    # Conda environment (Python 3.10, CUDA 12.1)
├── requirements.txt                   # Pip dependencies
├── pyproject.toml                     # Package metadata
├── configs/                           # Pipeline configuration presets
│   ├── default.yaml
│   ├── classical.yaml
│   ├── neural.yaml
│   ├── hybrid.yaml
│   └── quality.yaml
├── photogrammetry/                    # Main Python package
│   ├── pipeline/                      # Pipeline orchestration and config
│   ├── preprocessing/                 # Image loading, EXIF, resizing
│   ├── features/                      # Feature extraction and matching
│   ├── sfm/                           # Structure from Motion backends
│   ├── dense/                         # Dense reconstruction
│   ├── surface/                       # Surface reconstruction backends
│   ├── export/                        # Mesh/point cloud export
│   └── utils/                         # Geometry, visualization, logging
├── third_party/
│   └── sdfstudio/                     # Vendored SDFStudio (patched for PyTorch 2.4)
├── scripts/
│   └── run.py                         # CLI entry point
└── tests/
```

## Key References

- **VGGT**: Wang et al., "Visual Geometry Grounded Transformer", CVPR 2025 (Best Paper)
- **SuperPoint**: DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description", CVPRW 2018
- **LightGlue**: Lindenberger et al., "LightGlue: Local Feature Matching at Light Speed", ICCV 2023
- **AliceVision**: Griwodz et al., "AliceVision Meshroom: An open-source 3D reconstruction pipeline", MMSys 2021
- **MILo**: Guédon & Lepetit, "MILo: Mesh-In-the-Loop Gaussian Splatting", SIGGRAPH Asia 2025
- **NeuRodin**: Wang et al., "NeuRodin: A Two-stage Framework for High-Fidelity Neural Surface Reconstruction", NeurIPS 2024
- **COLMAP**: Schönberger & Frahm, "Structure-from-Motion Revisited", CVPR 2016

## License

This project integrates multiple open-source components, each with their own licenses:

- AliceVision: MPLv2
- MILo: See [Anttwo/MILo](https://github.com/Anttwo/MILo) license
- NeuRodin/SDFStudio: Apache 2.0
- LightGlue: Apache 2.0
- VGGT: See [facebookresearch/vggt](https://github.com/facebookresearch/vggt) license
