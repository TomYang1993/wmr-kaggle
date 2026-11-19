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


def cpu_quota(cgroup="/sys/fs/cgroup"):
    """CPUs this process may really use. A container renting a share of its host (a RunPod pod)
    still shows every host core in os.cpu_count(), so thread pools sized from it get throttled;
    the cgroup quota is the real limit (v2 cpu.max, else v1 cfs quota / period)."""
    from pathlib import Path
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        q, p = Path(cgroup, "cpu.max").read_text().split()[:2]
    except (OSError, ValueError):
        try:
            q = Path(cgroup, "cpu", "cpu.cfs_quota_us").read_text().strip()
            p = Path(cgroup, "cpu", "cpu.cfs_period_us").read_text().strip()
        except OSError:
            return n
    if q in ("max", "-1"):
        return n
    return max(1, min(n, -(-int(q) // int(p))))


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
            "machine": platform.machine(), "torch_device": torch_device(), "gpus": gpu_ids(),
            "cpu_count": os.cpu_count(), "cpus_usable": cpu_quota(),
            "WMR_THREADS": os.environ.get("WMR_THREADS")}
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
