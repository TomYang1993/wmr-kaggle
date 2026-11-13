"""The torch borrow loop must match the OpenCV one. Panning shot with a static opaque box;
runs on CUDA when available, else torch on the CPU.  python tests/test_fill_torch.py"""
import os
import sys
import types

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.modules.setdefault("av", types.ModuleType("av"))   # remove.py imports vio; no video I/O here
from wmr import remove as rm  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W, N, DX = 240, 400, 41, 6.5
rng = np.random.default_rng(0)
bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W + 2 * N * 2, 3), dtype=np.uint8), (0, 0), 3)
bg = cv2.normalize(bg, None, 0, 255, cv2.NORM_MINMAX)


def frame(i):   # background moves right by DX px per frame
    A = np.float32([[1, 0, i * DX], [0, 1, 0]])
    return cv2.warpAffine(bg, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def case(box_dx, flicker, second=False):
    """box_dx: overlay moves this many px per frame (source masks partly cover the target hole);
    flicker: every 3rd frame brightened by 40, which the photometric check must reject;
    second: a static second overlay far from the first, which must be filled as its own region."""
    buf = []
    for i in range(N):
        C = np.eye(3)
        C[0, 2] = i * DX
        clean = frame(i)
        if flicker and i % 3 == 0:
            clean = cv2.add(clean, 40)
        m = np.zeros((H, W), bool)
        x = 150 + int((i - N // 2) * box_dx)
        m[100:130, x:x + 110] = True
        if second:
            m[195:220, 300:390] = True
        img = clean.copy()
        img[m] = 255
        buf.append({"i": i, "img": img, "mask": m, "C": C, "shot": 0, "clean": clean})
    t = buf[N // 2]
    res_cv, st_cv = rm.fill_frame(dict(t), buf, None)
    for e in buf:
        e["g"] = torch.from_numpy(e["img"]).to(dev)
        e["gm"] = torch.from_numpy(e["mask"]).to(dev)
    res_t, st_t = rm.fill_frame(t, buf, None)
    m = t["mask"]
    assert len(rm._regions(m, 24)) == (2 if second else 1), rm._regions(m, 24)
    d_impl = np.abs(res_cv[m].astype(int) - res_t[m].astype(int)).mean()
    err_t = np.abs(res_t[m].astype(int) - t["clean"][m].astype(int)).mean()
    print(f"box_dx {box_dx}, flicker {flicker}: cv {st_cv}, torch {st_t}; |cv - torch| {d_impl:.2f}, torch error {err_t:.2f}")
    # cv and torch may disagree on a few edge pixels (bilinear vs nearest at the mask border)
    assert abs(st_t["borrowed"] - st_cv["borrowed"]) <= 0.01 * m.sum(), (st_cv, st_t)
    assert d_impl < 1.5 and err_t < 3.0, (d_impl, err_t)


print("device", dev)
case(0, False)
case(-DX, False)    # overlay rides with the background: nearest sources are masked where it matters
case(0, True)
case(-DX, True)
case(-DX, False, second=True)
print("ok")
