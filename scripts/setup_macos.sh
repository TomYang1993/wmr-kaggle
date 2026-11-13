#!/usr/bin/env bash
# macOS (Apple Silicon). Creates ./.venv and installs wmr into it.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
python3 -m venv "$HERE/.venv"
source "$HERE/.venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$HERE/requirements/macos.txt"
pip install -q "$HERE"
if [ "$1" = "propainter" ]; then
  pip install -q torch torchvision einops timm addict yapf imageio imageio-ffmpeg scikit-image
  python -m wmr setup-propainter
fi
python -m wmr doctor
echo "Activate with: source $HERE/.venv/bin/activate"
