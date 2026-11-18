"""LaMa inpainting (big-lama TorchScript, Apache-2.0, the build IOPaint uses) for the holes that
borrowing from nearby frames cannot fill. Telea smears letter-shaped blurs into texture; LaMa
synthesises texture from the surrounding picture.
Model file: $WMR_LAMA, else <torch hub>/checkpoints/big-lama.pt, downloaded once (196 MB)."""
import hashlib
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
# Half precision (autocast) runs the model's FFTs in fp16, which cuFFT takes only at power-of-two
# sizes, and which overflowed at 512x1024 on a T4 (2026-11-17). Up to this padded area it matched
# fp32 (mean error ~3 levels, as much as a different padding gives) at 2.3x the speed.
FP16_MAX_AREA = 256 * 512


def model_path():
    import torch
    p = os.environ.get("WMR_LAMA") or os.path.join(torch.hub.get_dir(), "checkpoints", "big-lama.pt")
    if os.path.exists(p):
        return p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".part"
    torch.hub.download_url_to_file(URL, tmp, progress=False)
    h = hashlib.md5()
    with open(tmp, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != MD5:
        os.remove(tmp)
        raise RuntimeError(f"big-lama.pt download has md5 {h.hexdigest()}, expected {MD5}")
    os.replace(tmp, p)
    return p


def _pow2(n):
    return 1 << (int(n) - 1).bit_length()


class Lama:
    """One model per device; `many` runs a frame's crops on all of them at once."""

    def __init__(self, devices):
        import torch
        self.torch = torch
        self.devs = [torch.device(d) for d in ([devices] if isinstance(devices, str) else devices)]
        self.dev = ", ".join(str(d) for d in self.devs)
        p = model_path()
        with warnings.catch_warnings():   # torch >= 2.15 flags jit.load as deprecated; still loads
            warnings.simplefilter("ignore", FutureWarning)
            self.ms = [torch.jit.load(p, map_location=d).eval() for d in self.devs]
        self.pool = ThreadPoolExecutor(len(self.ms)) if len(self.ms) > 1 else None
        self.calls = self.fp16 = self.fp16_overflow = 0

    def __call__(self, bgr, mask):
        return self.many([(bgr, mask)])[0]

    def many(self, items):
        """items: [(bgr HxWx3 uint8, mask HxW bool, True = fill)] -> inpainted bgr crops.
        Several devices: crop k goes to device k % n, each device in its own thread (torch
        releases the GIL inside CUDA calls). ponytail: no batching on one device (+15% at most)."""
        n = len(self.ms)
        if self.pool is None or len(items) < 2:
            return [self._run(0, b, m) for b, m in items]
        parts = list(self.pool.map(lambda k: [self._run(k, b, m) for b, m in items[k::n]], range(n)))
        out = [None] * len(items)
        for k, p in enumerate(parts):
            out[k::n] = p
        return out

    def _run(self, k, bgr, mask):
        torch = self.torch
        h, w = mask.shape
        half = self.devs[k].type == "cuda" and _pow2(h) * _pow2(w) <= FP16_MAX_AREA
        # fp32 needs only multiples of 8 (three 2x downsamplings); smaller pads run faster
        H, W = (_pow2(h), _pow2(w)) if half else (-(-h // 8) * 8, -(-w // 8) * 8)
        img = np.pad(bgr[..., ::-1], ((0, H - h), (0, W - w), (0, 0)), mode="symmetric")
        m = np.pad(mask, ((0, H - h), (0, W - w)), mode="symmetric")
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(img)).to(self.devs[k]).permute(2, 0, 1)[None].float().div(255)
            y = torch.from_numpy(m).to(self.devs[k])[None, None].float()
            o = None
            if half:
                with torch.autocast("cuda", dtype=torch.float16):
                    o = self.ms[k](x, y)
                self.fp16 += 1
                if not bool(torch.isfinite(o).all()):   # fp16 overflow: redo this crop in fp32
                    o = None
                    self.fp16_overflow += 1
            if o is None:
                o = self.ms[k](x, y)
            o = o[0].float().permute(1, 2, 0).clamp(0, 1).mul(255).round().byte()[:h, :w].cpu().numpy()
        self.calls += 1
        return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
