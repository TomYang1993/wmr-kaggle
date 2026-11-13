# Windows + NVIDIA. Run in PowerShell from the wmr folder:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 [-ProPainter]
param([switch]$ProPainter)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
python -m venv "$here\.venv"
& "$here\.venv\Scripts\Activate.ps1"
python -m pip install -q --upgrade pip
pip install -q -r "$here\requirements\windows.txt"
# CUDA build of PyTorch: also supplies the CUDA/cuDNN libraries ONNX Runtime needs, so its CUDA
# major must match onnxruntime-gpu's (1.31 on PyPI is built for CUDA 13; 2026-11-12). CUDA 13
# needs NVIDIA driver 580 or newer. `wmr doctor` must list CUDAExecutionProvider, else OCR runs on CPU.
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip uninstall -y -q onnxruntime
pip install -q onnxruntime-gpu
pip install -q "$here"
if ($ProPainter) {
  pip install -q einops timm addict yapf imageio imageio-ffmpeg scikit-image
  python -m wmr setup-propainter
}
python -m wmr doctor
Write-Host "Activate with: $here\.venv\Scripts\Activate.ps1"
