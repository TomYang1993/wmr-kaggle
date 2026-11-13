"""Platform / accelerator selection. Nothing else in the package checks the platform."""
import os
import platform
import shutil
import subprocess


def torch_device():
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def ort_providers():
    """Available ONNX Runtime providers. Importing torch first lets onnxruntime-gpu
    find the CUDA/cuDNN libraries that ship inside the torch wheel (Kaggle, Windows)."""
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    return list(ort.get_available_providers())


def gpu_ids():
    """Indices of visible NVIDIA GPUs (empty on macOS / CPU-only machines)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip() != "":
        return [int(x) for x in vis.split(",") if x.strip().isdigit()]
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout
            return [int(x) for x in out.split() if x.strip().isdigit()]
        except Exception:
            return []
    return []


def describe():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine(), "torch_device": torch_device(), "gpus": gpu_ids()}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        info["torch"] = None
    try:
        info["ort_providers"] = ort_providers()
    except Exception as e:  # pragma: no cover
        info["ort_providers"] = f"unavailable: {e}"
    for mod in ("av", "cv2", "numpy", "scipy", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info
