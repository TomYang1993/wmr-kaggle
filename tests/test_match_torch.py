"""Pass B's torch template search (_TorchMatch) must pick what cv2 picks: grad() of each window and
TM_CCOEFF_NORMED, best over the candidate windows. CUDA when available, else torch on the CPU.
    python tests/test_match_torch.py"""
import os
import sys
import types

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.modules.setdefault("av", types.ModuleType("av"))   # track imports vio; no video I/O here
from wmr import track as tk  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
H, W = 400, 640
bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (0, 0), 2)
img = bg.copy()
for x, y in ((60, 80), (300, 90), (420, 300)):   # three identical marks
    cv2.putText(img, "Huohua Studio", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 235, 2)
B = 40
gp = cv2.copyMakeBorder(img, B, B, B, B, cv2.BORDER_REPLICATE)
tmpl = tk.grad(img[52:92, 50:270])
t = types.SimpleNamespace(id=0, tmpl=tmpl, TH=tmpl.shape[0], TW=tmpl.shape[1], R=24)
m = tk._TorchMatch([t], dev)
ww, wh = t.TW + 2 * t.R, t.TH + 2 * t.R
worst = 0.0
for trial in range(60):
    origins = [(int(rng.integers(0, gp.shape[1] - ww)), int(rng.integers(0, gp.shape[0] - wh)))
               for _ in range(int(rng.integers(1, 6)))]
    if trial % 3 == 0:   # include windows around each mark
        origins[0] = (60 - 10 - t.R + B + int(rng.integers(-8, 8)), 52 - t.R + B + int(rng.integers(-8, 8)))
    best, bpos = -1.0, None
    for ox, oy in origins:
        res = cv2.matchTemplate(tk.grad(gp[oy:oy + wh, ox:ox + ww]), t.tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best:
            best, bpos = mx, (ox + loc[0], oy + loc[1])
    m.frame(gp)
    m.search(t, origins)
    (_, best_t, bpos_t), = m.results()
    worst = max(worst, abs(best - best_t))
    assert abs(best - best_t) < 1e-3, (trial, best, best_t)
    assert bpos == bpos_t or best > 0.99 or abs(best - best_t) < 1e-4, (trial, bpos, bpos_t, best, best_t)
print(f"ok on {dev}: max |score diff| {worst:.2e}")
