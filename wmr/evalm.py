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


def _edge_err(o, c, area):
    """Mean |gradient magnitude(output) - gradient magnitude(clean)| over `area`: high for hard
    seams (edges that should not exist) and for smears (texture that went missing)."""
    ys, xs = np.nonzero(area)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    g = []
    for im in (o, c):
        y = cv2.cvtColor(im[sl], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g.append(cv2.magnitude(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1)))
    return float(np.abs(g[0] - g[1])[area[sl]].mean())


def _residue(img, doc, i):
    """Per track at frame i: best normalised match of its gradient template within 2 px of where
    the track says the mark is. Marks score ~0.5-0.9 before removal; clean picture ~0-0.2. Sees
    what OCR cannot read: ghost outlines, inverted text, partial letters."""
    gray = None
    out = {}
    for t in doc["tracks"]:
        if not t["enabled"] or t.get("tmpl_img") is None or i >= len(t["pos"]) or t["pos"][i, 0] <= tk.ABSENT:
            continue
        x, y = int(t["pos"][i, 0]) - 2, int(t["pos"][i, 1]) - 2
        if x < 0 or y < 0 or x + t["w"] + 4 > img.shape[1] or y + t["h"] + 4 > img.shape[0]:
            continue
        if gray is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = tk.grad(gray[y:y + t["h"] + 4, x:x + t["w"] + 4])
        out[t["id"]] = float(cv2.matchTemplate(g, t["tmpl_img"], cv2.TM_CCOEFF_NORMED).max())
    return out


def verify(video, texts, sample_fps=2.0, use_gpu=True, thr=0.7, enhance=True, ocr=None, tracks=None):
    """Frames (sampled) where an expected string is still readable. Pass `ocr` (an OCRPool) to
    reuse the engines analyze already built; `tracks` (tracks.json) adds a per-track residue
    score (see _residue)."""
    doc = tk.load(tracks) if tracks else None
    res = {}
    info = vio.probe(video)
    step = max(1, int(round(info["fps"] / sample_fps)))
    if ocr is None:
        ocr = OCRPool(use_gpu=use_gpu)
    ocr.enhance = enhance
    leaks, checked = [], 0
    frames = ((i, f.to_ndarray(format="bgr24")) for i, _, f in vio.read_frames(video, convert=False) if i % step == 0)
    for i, img, found in ocr.imap(frames):
        checked += 1
        if doc is not None:
            for k, v in _residue(img, doc, i).items():
                res.setdefault(k, []).append(v)
        hit = [m for m in (match_text(d["text"], texts, thr)[0] for d in found) if m]
        if hit:
            leaks.append({"frame": i, "texts": sorted(set(hit))})
    out = {"checked_frames": checked, "leak_frames": len(leaks), "leaks": leaks[:200]}
    if res:   # ponytail: 0.35 = pass B's lowest accept score (track.T_LO)
        out["residue"] = {k: {"median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3),
                              "share_ge_0.35": round(float(np.mean(np.array(v) >= tk.T_LO)), 3)}
                          for k, v in sorted(res.items())}
    return out


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
    pin, pout, pbefore, recalls, flick_o, flick_c, pring, eerr = [], [], [], [], [], [], [], []
    leaked = 0
    prev = None
    for (i, _, c), (_, _, o) in zip(clean, outp):
        if i >= n:
            break
        g = gt_mask(i)
        gd = cv2.dilate(g.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if g.any():
            pin.append(_psnr(c[gd], o[gd]))
            near = cv2.dilate(gd.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
            ring = near & ~(cv2.dilate(gd.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            pring.append(_psnr(c[ring], o[ring]))
            eerr.append(_edge_err(o, c, near))
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
        "psnr_ring_10px": round(float(np.mean(pring)), 2) if pring else None,
        "edge_err_near_mask": round(float(np.mean(eerr)), 2) if eerr else None,
        "flicker_in_mask_output": round(float(np.mean(flick_o)), 2) if flick_o else None,
        "flicker_in_mask_clean": round(float(np.mean(flick_c)), 2) if flick_c else None,
    }
    if doc is not None:
        res["mask_recall_mean"] = round(float(np.mean(recalls)), 4) if recalls else None
        res["frames_with_mask_recall_below_95pct"] = int(leaked)
    return res
