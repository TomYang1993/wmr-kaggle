"""Opacity (alpha) and colour estimation for a rigid overlay from a stack of aligned crops.

A blended overlay obeys J = a*W + (1-a)*I. Over many frames with a changing background:
  std_t(J) = (1-a) * std_t(I)      -> a from the dip in temporal std under the glyphs
  mean_t(J) = a*W + (1-a)*mean_t(I) -> W, then a refined from the means
"""
import cv2
import numpy as np
from scipy.ndimage import percentile_filter


def estimate(sum_, sumsq, n, text_h, tmpl=None):
    out = {"ok": False, "n": int(n)}
    if n < 12:
        out["why"] = "too few aligned samples"
        return out
    M = (sum_ / n).astype(np.float32)
    var = np.maximum(sumsq / n - (sum_ / n) ** 2, 0)
    S = np.sqrt(var).mean(2).astype(np.float32)
    k = int(max(9, round(0.9 * text_h))) | 1
    B = percentile_filter(S, 80, size=k, mode="nearest")
    B = cv2.GaussianBlur(B, (0, 0), k / 4.0)
    out["bg_std"] = float(np.median(B))
    if out["bg_std"] < 4.0:
        out["why"] = "background barely changes under the overlay"
        return out
    a_var = np.clip(1 - S / np.maximum(B, 1e-3), 0, 1)
    a_var = cv2.medianBlur((a_var * 255).astype(np.uint8), 3).astype(np.float32) / 255
    zone = np.ones(a_var.shape, bool)
    if tmpl is not None:      # glyphs can only sit near the persistent edges of the template
        kz = int(max(5, round(text_h / 3.0))) | 1
        zone = cv2.dilate((tmpl > 0.2 * float(tmpl.max())).astype(np.uint8), np.ones((kz, kz), np.uint8)) > 0
    thr = max(0.08, 0.5 * float(np.percentile(a_var[zone], 85)))
    g0 = (a_var > thr) & zone
    support = cv2.dilate(g0.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    Mbg = cv2.inpaint(np.clip(M, 0, 255).astype(np.uint8), support.astype(np.uint8) * 255, 5,
                      cv2.INPAINT_TELEA).astype(np.float32)
    sup_vals = a_var[support]
    core = support & (a_var >= 0.9 * np.percentile(sup_vals, 97))
    if core.sum() < 20:
        out["why"] = "no stable glyph pixels found"
        return out
    Mc, Bc = M[core], Mbg[core]
    # A colour channel cannot exceed 0..255, which bounds the opacity from below; the bound is
    # tight for white, black and other saturated colours. The std-dip estimate is biased low by
    # codec noise, so take the larger of the two.
    room = np.where(Mc >= Bc, 255.0 - Bc, Bc)
    bound = float(np.median((np.abs(Mc - Bc) / np.maximum(room, 1.0)).max(axis=1)))
    a0 = float(np.clip(max(bound, float(np.median(a_var[core]))), 0.05, 1.0))
    Wc = np.clip(np.median((Mc - (1 - a0) * Bc) / a0, axis=0), 0, 255)
    mad = float(np.median(np.abs((Mc - (1 - a0) * Bc) / a0 - Wc)))
    d = Wc[None, None, :] - Mbg
    den = (d * d).sum(2)
    a_mean = ((M - Mbg) * d).sum(2) / np.maximum(den, 1e-3)
    a = np.where(den > 2500, a_mean, a_var)
    a = np.clip(a, 0, 1)
    a[~support] = 0
    a[a < max(0.03, 0.15 * a0)] = 0
    out.update(ok=True, alpha=a.astype(np.float32), color=[float(c) for c in Wc],
               color_mad=mad, median=a0)
    return out
