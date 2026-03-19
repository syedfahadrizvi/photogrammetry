# SDFStudio Patches for Python 3.10 + PyTorch 2.4

This directory contains a vendored copy of [SDFStudio](https://github.com/autonomousvision/sdfstudio)
with [NeuRodin](https://github.com/Open3DVLab/NeuRodin) model code, patched
for compatibility with Python 3.10 and PyTorch 2.4 + CUDA 12.1.

## Setup

Run the setup script to clone and patch SDFStudio:

```bash
cd photogrammetry
bash scripts/setup_sdfstudio.sh
```

This will:
1. Clone SDFStudio and NeuRodin into `third_party/sdfstudio/`
2. Apply all patches listed below
3. Install SDFStudio as an editable package

## Patches Applied

### 1. Remove `torch._six` references

**Files**: `nerfstudio/utils/misc.py`, `nerfstudio/utils/decorators.py`

`torch._six` was removed in PyTorch 2.0. All references are replaced with
standard Python equivalents:

```python
# BEFORE (PyTorch 1.x)
from torch._six import string_classes

# AFTER (PyTorch 2.x)
string_classes = (str,)
```

### 2. Update `torch.cuda.amp` to `torch.amp`

**Files**: `nerfstudio/engine/trainer.py`, `nerfstudio/models/*.py`

PyTorch 2.4 deprecates the `torch.cuda.amp` namespace:

```python
# BEFORE
from torch.cuda.amp import GradScaler, autocast

# AFTER
from torch.amp import GradScaler, autocast
# GradScaler("cuda") instead of GradScaler()
# autocast("cuda") instead of autocast()
```

### 3. Fix deprecated `torch.nn.utils.clip_grad_norm`

**Files**: `nerfstudio/engine/trainer.py`

```python
# BEFORE
torch.nn.utils.clip_grad_norm(parameters, max_norm)

# AFTER
torch.nn.utils.clip_grad_norm_(parameters, max_norm)
```

### 4. Update typing imports for Python 3.10

**Files**: Various

Python 3.10 allows using built-in types directly in type hints:

```python
# BEFORE
from typing import Dict, List, Optional, Tuple

# AFTER (Python 3.10+)
# Use dict, list, tuple, X | None directly
```

### 5. Loosen version pins in `setup.py` / `pyproject.toml`

**Files**: `setup.py`, `pyproject.toml`

Original pins `torch==1.12.1` are relaxed to `torch>=2.0`:

```python
# BEFORE
install_requires=["torch==1.12.1", ...]

# AFTER
install_requires=["torch>=2.0", ...]
```

### 6. Update tiny-cuda-nn binding

**Files**: `setup.py`

Ensure compatibility with `tiny-cuda-nn>=2.0`:

```python
# BEFORE
"tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/...#..."

# AFTER
# Remove the pin; user installs tiny-cuda-nn separately
```

### 7. Copy NeuRodin model code

**Files**: `nerfstudio/models/neurodin*.py`, `nerfstudio/configs/method_configs.py`

NeuRodin model definitions and config registrations from the
[Open3DVLab/NeuRodin](https://github.com/Open3DVLab/NeuRodin) repository
are merged into the SDFStudio source tree.

## Validation

After patching, verify the setup works:

```bash
# Check ns-train recognizes NeuRodin configs
ns-train --help | grep neurodin

# Quick smoke test (will fail without data but validates imports)
python -c "from nerfstudio.models.neurodin import NeuRodinModel; print('OK')"
```

## Known Limitations

- Some nerfstudio viewer features may not work (they require nerfstudio >=1.0)
- The `ns-extract-mesh` command from modern nerfstudio is not available;
  use `zoo/extract_surface.py` instead
- Mixed-precision (fp16) training may need `torch.amp` dtype adjustments
  depending on GPU architecture
