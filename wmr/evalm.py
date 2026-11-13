"""Evaluation against synthetic ground truth, plus a ground-truth-free OCR leak check."""
import json
import os

import cv2
import numpy as np

from . import track as tk
from . import vio
from .detect import OCRPool, match_text


def _psnr(a, b):
    if a.size == 0:
        return None
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-6 else float(10 * np.log10(255.0 ** 2 / mse))


def verify(video, texts, sample_fps=2.0, use_gpu=True, thr=0.7, enhance=True, ocr=None):
    """Frames (sampled) where an expected string is still readable. Pass `ocr` (an OCRPool) to
    reuse the engines analyze already built."""
    info = vio.probe(video)
    step = max(1, int(round(info["fps"] / sample_fps)))
    if ocr is None:
        ocr = OCRPool(use_gpu=use_gpu)
    ocr.enhance = enhance
    leaks, checked = [], 0
    frames = ((i, f.to_ndarray(format="bgr24")) for i, _, f in vio.read_frames(video, convert=False) if i % step == 0)
    for i, _, found in ocr.imap(frames):
        checked += 1
        hit = [m for m in (match_text(d["text"], texts, thr)[0] for d in found) if m]
        if hit:
            leaks.append({"frame": i, "texts": sorted(set(hit))})
    return {"checked_frames": checked, "leak_frames": len(leaks), "leaks": leaks[:200]}


def evaluate(gt_dir, output, tracks_json=None, marked=None):
    with open(os.path.join(gt_dir, "gt.json")) as fh:
        gt = json.load(fh)
    W, H, n = gt["width"], gt["height"], gt["frames"]
    covs = [cv2.imread(os.path.join(gt_dir, o["cov_file"]), cv2.IMREAD_GRAYSCALE) / 255.0 * o["alpha"] > 0.08
            for o in gt["overlays"]]
    doc = tk.load(tracks_json) if tracks_json else None

    def gt_mask(i):
        m = np.zeros((H, W), bool)
        for o, c in zip(gt["overlays"], covs):
            x, y = o["pos"][i]
            x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + o["w"]), min(H, y + o["h"])
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] |= c[y0 - y:y1 - y, x0 - x:x1 - x]
        return m

    def pred_mask(i):
        m = np.zeros((H, W), bool)
        for t in doc["tracks"]:
            if not t["enabled"] or i >= len(t["pos"]) or t["pos"][i, 0] <= tk.ABSENT:
                continue
            x, y = int(t["pos"][i, 0]), int(t["pos"][i, 1])
            x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + t["w"]), min(H, y + t["h"])
            if x1 <= x0 or y1 <= y0:
                continue
            tm = t["mask_img"] if t["tier"] == "fill" else (t["alpha_img"] > 0)
            m[y0:y1, x0:x1] |= tm[y0 - y:y1 - y, x0 - x:x1 - x]
        return m

    clean = vio.read_frames(os.path.join(gt_dir, "clean.mp4"))
    outp = vio.read_frames(output)
    mk = vio.read_frames(marked) if marked else None
    pin, pout, pbefore, recalls, flick_o, flick_c = [], [], [], [], [], []
    leaked = 0
    prev = None
    for (i, _, c), (_, _, o) in zip(clean, outp):
        if i >= n:
            break
        g = gt_mask(i)
        gd = cv2.dilate(g.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if g.any():
            pin.append(_psnr(c[gd], o[gd]))
        pout.append(_psnr(c[~gd], o[~gd]))
        if mk is not None:
            _, _, mfr = next(mk)
            if g.any():
                pbefore.append(_psnr(c[gd], mfr[gd]))
        if doc is not None and g.any():
            r = float((pred_mask(i) & g).sum()) / float(g.sum())
            recalls.append(r)
            leaked += r < 0.95
        if prev is not None and g.any() and prev[2].any():
            both = gd & prev[2]
            if both.sum() > 50:
                flick_o.append(float(np.abs(o[both].astype(np.int16) - prev[1][both]).mean()))
                flick_c.append(float(np.abs(c[both].astype(np.int16) - prev[0][both]).mean()))
        prev = (c.astype(np.int16), o.astype(np.int16), gd)
    res = {
        "psnr_in_mask_after": round(float(np.mean(pin)), 2) if pin else None,
        "psnr_in_mask_before": round(float(np.mean(pbefore)), 2) if pbefore else None,
        "psnr_outside_mask": round(float(np.mean(pout)), 2) if pout else None,
        "flicker_in_mask_output": round(float(np.mean(flick_o)), 2) if flick_o else None,
        "flicker_in_mask_clean": round(float(np.mean(flick_c)), 2) if flick_c else None,
    }
    if doc is not None:
        res["mask_recall_mean"] = round(float(np.mean(recalls)), 4) if recalls else None
        res["frames_with_mask_recall_below_95pct"] = int(leaked)
    return res
