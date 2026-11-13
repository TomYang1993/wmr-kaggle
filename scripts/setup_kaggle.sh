#!/usr/bin/env bash
# Run inside a Kaggle notebook with Accelerator = GPU T4 x2 and Internet = On.
#   !bash wmr/scripts/setup_kaggle.sh            (add "propainter" to also fetch ProPainter)
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
pip install -q -r "$HERE/requirements/kaggle.txt"
# GPU build of ONNX Runtime for OCR (rapidocr 3.x does not pull in an ONNX Runtime itself).
pip uninstall -y -q onnxruntime 2>/dev/null || true   # CPU build would shadow onnxruntime-gpu; usually absent on Kaggle
pip install -q onnxruntime-gpu || pip install -q onnxruntime
# cuDNN 9.19 (pinned by torch cu130) has no NCHW engine for the OCR det convs on T4; 9.28 does,
# is ~1.7x faster than the NHWC workaround, and torch still runs on it (probed 2026-11-10).
# pip warns that torch pins 9.19; expected.
pip install -q -U "nvidia-cudnn-cu13>=9.28" 2>&1 | grep -v "which is incompatible" || true
pip install -q --no-deps "$HERE"
# OCR models from the yxyxtx/wmr-ocr-models dataset when attached (RapidOCR checks their sha256
# and skips its download: 10-24 s per run on Kaggle's network)
M=$(python -c "import rapidocr, os; print(os.path.join(os.path.dirname(rapidocr.__file__), 'models'))")
find -L /kaggle/input -path '*wmr-ocr-models*' -name '*.onnx' -exec cp -n {} "$M"/ \; 2>/dev/null || true
if [ "$1" = "propainter" ]; then
  python -m wmr setup-propainter
  pip install -q einops timm addict yapf imageio imageio-ffmpeg scikit-image
fi
python -m wmr doctor
