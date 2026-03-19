#!/usr/bin/env bash
#
# Setup script for vendoring and patching SDFStudio + NeuRodin
# for Python 3.10 + PyTorch 2.4 + CUDA 12.1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SDFSTUDIO_DIR="$PROJECT_DIR/third_party/sdfstudio"

echo "=== Setting up SDFStudio + NeuRodin ==="
echo "Target: $SDFSTUDIO_DIR"

# -------------------------------------------------------
# 1. Clone SDFStudio
# -------------------------------------------------------
if [ ! -d "$SDFSTUDIO_DIR/.git" ]; then
    echo "Cloning SDFStudio..."
    git clone https://github.com/autonomousvision/sdfstudio.git "$SDFSTUDIO_DIR"
else
    echo "SDFStudio already cloned."
fi

cd "$SDFSTUDIO_DIR"

# -------------------------------------------------------
# 2. Clone NeuRodin and copy model code
# -------------------------------------------------------
NEURODIN_TMP="/tmp/neurodin_$$"
echo "Cloning NeuRodin model code..."
git clone --depth 1 https://github.com/Open3DVLab/NeuRodin.git "$NEURODIN_TMP"

# Copy NeuRodin model files into SDFStudio
cp -r "$NEURODIN_TMP"/nerfstudio/models/neurodin* nerfstudio/models/ 2>/dev/null || true
cp -r "$NEURODIN_TMP"/nerfstudio/configs/* nerfstudio/configs/ 2>/dev/null || true
cp -r "$NEURODIN_TMP"/zoo . 2>/dev/null || true

rm -rf "$NEURODIN_TMP"
echo "NeuRodin model code merged."

# -------------------------------------------------------
# 3. Patch torch._six references
# -------------------------------------------------------
echo "Patching torch._six references..."
find . -name "*.py" -exec grep -l "torch._six\|from torch._six" {} \; | while read -r f; do
    sed -i 's/from torch._six import string_classes/string_classes = (str,)/g' "$f"
    sed -i 's/from torch._six import/# Removed: from torch._six import/g' "$f"
    echo "  Patched: $f"
done

# -------------------------------------------------------
# 4. Patch torch.cuda.amp -> torch.amp
# -------------------------------------------------------
echo "Patching torch.cuda.amp -> torch.amp..."
find . -name "*.py" -exec grep -l "torch.cuda.amp" {} \; | while read -r f; do
    sed -i 's/from torch.cuda.amp import GradScaler/from torch.amp import GradScaler/g' "$f"
    sed -i 's/from torch.cuda.amp import autocast/from torch.amp import autocast/g' "$f"
    sed -i 's/torch.cuda.amp.GradScaler()/torch.amp.GradScaler("cuda")/g' "$f"
    sed -i 's/torch.cuda.amp.autocast()/torch.amp.autocast("cuda")/g' "$f"
    echo "  Patched: $f"
done

# -------------------------------------------------------
# 5. Fix clip_grad_norm (missing underscore)
# -------------------------------------------------------
echo "Patching clip_grad_norm..."
find . -name "*.py" -exec grep -l "clip_grad_norm(" {} \; | while read -r f; do
    # Only replace the non-underscore version, avoid double-replacing
    sed -i 's/clip_grad_norm(/clip_grad_norm_(/g' "$f"
    # Fix any that got double-underscored
    sed -i 's/clip_grad_norm__(/clip_grad_norm_(/g' "$f"
    echo "  Patched: $f"
done

# -------------------------------------------------------
# 6. Loosen version pins in setup files
# -------------------------------------------------------
echo "Loosening version pins..."
if [ -f setup.py ]; then
    sed -i 's/"torch==1.\([0-9]*\)\.[0-9]*"/"torch>=2.0"/g' setup.py
    sed -i 's/"torchvision==0.\([0-9]*\)\.[0-9]*"/"torchvision>=0.15"/g' setup.py
    sed -i 's/"torch==1.\([0-9]*\)\.[0-9]*+cu[0-9]*"/"torch>=2.0"/g' setup.py
    echo "  Patched: setup.py"
fi

if [ -f pyproject.toml ]; then
    sed -i 's/torch==1\.[0-9]*\.[0-9]*/torch>=2.0/g' pyproject.toml
    echo "  Patched: pyproject.toml"
fi

# -------------------------------------------------------
# 7. Install SDFStudio as editable
# -------------------------------------------------------
echo "Installing SDFStudio (editable)..."
pip install -e .

# Install CLI
if command -v ns-install-cli &>/dev/null; then
    ns-install-cli
    echo "nerfstudio CLI installed."
fi

echo ""
echo "=== SDFStudio + NeuRodin setup complete ==="
echo "Verify with: ns-train --help | grep neurodin"
echo "See $SDFSTUDIO_DIR/../README_PATCHES.md for patch details."
