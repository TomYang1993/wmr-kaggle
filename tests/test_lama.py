"""LaMa must load (downloading big-lama.pt once) and continue texture into a hole better than
Telea: diagonal stripes with a box hole. CUDA when available, else the CPU.
    python tests/test_lama.py"""
import os
import sys
import time
import types

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.modules.setdefault("av", types.ModuleType("av"))
from wmr.lama import Lama  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 200, 330   # not multiples of 64: exercises the padding
yy, xx = np.mgrid[0:H, 0:W]
s = (127 + 100 * np.sin((xx + 0.5 * yy) / 5.0)).astype(np.uint8)
img = np.dstack([s, s[:, ::-1], np.full_like(s, 90)])
m = np.zeros((H, W), bool)
m[70:120, 100:220] = True
lama = Lama(dev)
lama(img, m)   # warm-up
t = time.perf_counter()
out = lama(img, m)
ms = 1000 * (time.perf_counter() - t)
tel = cv2.inpaint(img, m.astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA)
el = np.abs(out[m].astype(int) - img[m]).mean()
et = np.abs(tel[m].astype(int) - img[m]).mean()
print(f"LaMa on {dev}: {ms:.0f} ms for {W}x{H}; mean error in hole: LaMa {el:.1f}, Telea {et:.1f}")
assert out.shape == img.shape and out.dtype == np.uint8
assert el < 0.6 * et, (el, et)
print("ok")
