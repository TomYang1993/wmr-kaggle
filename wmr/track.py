"""Analyze pass: find text overlays, track them on every frame, estimate opacity, write tracks.json."""
import bisect
import json
import os
import time
from collections import Counter
from difflib import SequenceMatcher

import cv2
import numpy as np

from . import alpha as alpha_mod
from . import prof, vio
from .detect import OCRPool, match_text, norm_text

ABSENT = -10 ** 6
CUT_THR = 22.0
T_HI, T_LO, T_ACC = 0.5, 0.35, 0.4


def grad(gray):
    g = gray.astype(np.float32)
    return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _match_device(use_gpu):
    """Torch device for pass B's template search, or None for OpenCV on the CPU.
    WMR_MATCH_DEVICE overrides: "none" (OpenCV), or a torch device such as "cpu" (tests)."""
    env = os.environ.get("WMR_MATCH_DEVICE")
    if env:
        return None if env == "none" else env
    if not use_gpu:
        return None
    try:
        import torch
    except ImportError:
        return None
    return "cuda" if torch.cuda.is_available() else None


class _TorchMatch:
    """Pass B's template search on a torch device: grad() of each candidate window (reflect
    border, as cv2.Sobel) and TM_CCOEFF_NORMED against the track's template (FFT correlation,
    float64 window sums), all candidates of a track in one batch; one host sync per frame
    (results()). Scores match cv2 to ~1e-5."""

    def __init__(self, tracks, dev):
        import torch
        import torch.nn.functional as F
        self.torch, self.F, self.dev = torch, F, dev
        k = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=dev)
        self.sobel = torch.stack([k, k.t()])[:, None]
        self.tz = {}
        for t in tracks:   # zero-mean template, its FFT at the track's window size, its energy
            T = torch.from_numpy(t.tmpl).to(dev)
            Tz = T - T.mean()
            pad = torch.zeros(t.TH + 2 * t.R, t.TW + 2 * t.R, device=dev)
            pad[:t.TH, :t.TW] = Tz
            self.tz[t.id] = (torch.fft.rfft2(pad).conj(), float((Tz * Tz).sum()))

    def frame(self, gp):
        self.gp = self.torch.from_numpy(gp).to(self.dev).float()
        self.pending = []

    def search(self, t, origins):
        """Queue track t's windows at origins [(ox, oy), ...] of the padded frame."""
        F = self.F
        ww, wh = t.TW + 2 * t.R, t.TH + 2 * t.R
        win = self.torch.stack([self.gp[oy:oy + wh, ox:ox + ww] for ox, oy in origins])[:, None]
        g = F.conv2d(F.pad(win, (1, 1, 1, 1), mode="reflect"), self.sobel).abs().sum(1, keepdim=True)
        tzf, tn = self.tz[t.id]
        n, TH, TW = t.TW * t.TH, t.TH, t.TW
        # correlation by FFT (a direct conv with a ~60x300 kernel kept a T4 86% busy for 390 s
        # on 072q-1); the window is larger than template + shift, so the valid part never wraps
        gc = g - g.mean((2, 3), keepdim=True)   # the template sums to 0: same values, smaller numbers
        num = self.torch.fft.irfft2(self.torch.fft.rfft2(gc) * tzf, s=gc.shape[-2:])[..., :wh - TH + 1, :ww - TW + 1]
        # window sums of g and g^2 from float64 integral images (as cv2 does)
        g = g.double()
        I = F.pad(self.torch.cat([g, g * g], 1).cumsum(2).cumsum(3), (1, 0, 1, 0))
        S = I[..., TH:, TW:] - I[..., :-TH, TW:] - I[..., TH:, :-TW] + I[..., :-TH, :-TW]
        var = (S[:, 1:] - S[:, :1] * S[:, :1] / n).clamp_min(0)
        r = num / (var * tn).sqrt().clamp_min(1e-12).float()
        r = self.torch.where(r.abs() <= 1.125, r.clamp(-1, 1), self.torch.zeros_like(r))   # cv2's guard
        mx, idx = r.flatten(1).max(1)
        self.pending.append((t, origins, r.shape[3], mx, idx))

    def results(self):
        """[(t, best, bpos)] for the queued searches, in order; best is the first maximum."""
        if not self.pending:
            return []
        out = self.torch.cat([self.torch.stack([mx, idx.float()]) for _, _, _, mx, idx in self.pending], 1).cpu().numpy()
        res, c = [], 0
        for t, origins, ow, _, _ in self.pending:
            k = int(np.argmax(out[0, c:c + len(origins)]))
            best, flat = float(out[0, c + k]), int(out[1, c + k])
            ox, oy = origins[k]
            res.append((t, best, (ox + flat % ow, oy + flat // ow)))
            c += len(origins)
        return res


class _T:
    def __init__(self, tid, label, src, dets):
        self.id, self.label, self.src, self.dets = tid, label, src, dets
        self.enabled = True
        self.note = ""


def _cut(d, x, y, w, h):
    """Region [x, y, w, h] (frame coords) out of the grey crop stored with detection d."""
    x0, y0 = x - d["cx"], y - d["cy"]
    c = d["crop"]
    if x0 < 0 or y0 < 0 or x0 + w > c.shape[1] or y0 + h > c.shape[0]:
        return None
    return c[y0:y0 + h, x0:x0 + w]


def _cut_pad(d, x, y, w, h):
    """Like _cut, but regions running past the stored crop (text at the frame edge) are
    filled by repeating the border instead of failing."""
    c = d["crop"]
    x0, y0 = x - d["cx"], y - d["cy"]
    l, t = max(0, -x0), max(0, -y0)
    r, b = max(0, x0 + w - c.shape[1]), max(0, y0 + h - c.shape[0])
    if l or t or r or b:
        if l >= w or t >= h or r >= w or b >= h:
            return None
        c = cv2.copyMakeBorder(c, t, b, l, r, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + l, y0 + t
    return c[y0:y0 + h, x0:x0 + w]


def _group(dets):
    tracks = []
    for d in sorted(dets, key=lambda d: (d["f"], -d["score"])):
        for t in tracks:
            if t[-1]["f"] == d["f"]:
                continue
            mw = np.median([e["w"] for e in t])
            mh = np.median([e["h"] for e in t])
            if abs(d["w"] - mw) / mw < 0.3 and abs(d["h"] - mh) / mh < 0.35:
                t.append(d)
                break
        else:
            tracks.append([d])
    return tracks


def _build_template(t):
    # OCR boxes are often a little short on faint text, so take a generous width and pad well;
    # the opacity map later finds the exact glyph pixels inside the padded box.
    ws = int(np.percentile([d["w"] for d in t.dets], 75))
    hs = int(np.median([d["h"] for d in t.dets]))
    px, py = max(8, int(round(1.2 * hs))), max(6, int(round(0.35 * hs)))
    TW, TH = ws + 2 * px, hs + 2 * py
    # Template = median gradient over many detections. Start from the OCR boxes' own alignment,
    # then re-align every crop to the current median and repeat. A single frame is never used
    # as the reference, because a faint overlay's edges are weaker than the background's.
    good = [d for d in t.dets if abs(d["w"] - ws) <= 0.12 * ws + 2]
    if len(good) < 3:
        good = list(t.dets)
    good = sorted(good, key=lambda d: -d["score"])[:40]
    m = 10
    crops = []
    for d in good:
        C = _cut_pad(d, d["x"] - px - m, d["y"] - py - m, TW + 2 * m, TH + 2 * m)
        if C is not None:
            crops.append(grad(C))
    if not crops:
        return False
    tmpl = np.median(np.stack([c[m:m + TH, m:m + TW] for c in crops]), axis=0)
    for _ in range(3):
        aligned = []
        for gC in crops:
            res = cv2.matchTemplate(gC, tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > 0.15:
                aligned.append(gC[loc[1]:loc[1] + TH, loc[0]:loc[0] + TW])
        if len(aligned) < 3:
            break
        tmpl = np.median(np.stack(aligned), axis=0)
    # recentre: shift the template so the text's edge mass sits in the middle of the padded box
    prof_x, prof_y = tmpl.sum(0), tmpl.sum(1)
    hot_x = np.flatnonzero(prof_x > 0.35 * prof_x.max())
    hot_y = np.flatnonzero(prof_y > 0.35 * prof_y.max())
    if len(hot_x) and len(hot_y):
        sx = int(round((hot_x[0] + hot_x[-1]) / 2.0 - (TW - 1) / 2.0))
        sy = int(round((hot_y[0] + hot_y[-1]) / 2.0 - (TH - 1) / 2.0))
        sx, sy = int(np.clip(sx, -m, m)), int(np.clip(sy, -m, m))
        if sx or sy:
            stack = []
            for gC in crops:
                res = cv2.matchTemplate(gC, tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                x0, y0 = loc[0] + sx, loc[1] + sy
                if mx > 0.15 and 0 <= x0 and 0 <= y0 and x0 + TW <= gC.shape[1] and y0 + TH <= gC.shape[0]:
                    stack.append(gC[y0:y0 + TH, x0:x0 + TW])
            if len(stack) >= 3:
                tmpl = np.median(np.stack(stack), axis=0)
    if float(tmpl.std()) < 1e-3:
        return False
    t.px, t.py, t.TW, t.TH, t.text_h, t.tmpl = px, py, TW, TH, hs, tmpl.astype(np.float32)
    # refine every coarse detection against the template -> consistent coarse positions.
    # Boxes of a different width (truncated reads) are searched over a wider range.
    cf, cp = [], []
    for d in sorted(t.dets, key=lambda d: d["f"]):
        m = 12 + abs(d["w"] - ws)
        C = _cut_pad(d, d["x"] - px - m, d["y"] - py - 12, TW + 2 * m, TH + 24)
        if C is None:
            m = 12
            C = _cut_pad(d, d["x"] - px - m, d["y"] - py - 12, TW + 2 * m, TH + 24)
        p = None
        if C is not None:
            res = cv2.matchTemplate(grad(C), t.tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > 0.3:
                p = (d["x"] - px - m + loc[0], d["y"] - py - 12 + loc[1])
        if p is None:
            if abs(d["w"] - ws) > 0.12 * ws + 2:
                continue
            p = (d["x"] - px, d["y"] - py)
        cf.append(d["f"])
        cp.append(p)
    if not cf:
        return False
    t.cf, t.cp = cf, np.array(cp, dtype=np.int64)
    sp = [np.abs(t.cp[k + 1] - t.cp[k]).max() / max(1, cf[k + 1] - cf[k]) for k in range(len(cf) - 1)]
    t.speed = float(np.median(sp)) if sp else 0.0
    t.R = int(min(48, max(10, round(2.5 * t.speed))))
    return True


def _discover(dets, cuts, fps, step, min_seconds):
    clusters = []
    for d in dets:
        key = norm_text(d["text"])
        if len(key) < 3:
            continue
        for c in clusters:
            if abs(len(c["key"]) - len(key)) <= 3 and SequenceMatcher(None, c["key"], key).ratio() >= 0.8:
                c["dets"].append(d)
                break
        else:
            clusters.append({"key": key, "dets": [d]})
    out = []
    for c in clusters:
        fr = sorted({d["f"] for d in c["dets"]})
        if len(fr) < 4:
            continue
        dur = (fr[-1] - fr[0]) / fps
        cov = len(fr) / ((fr[-1] - fr[0]) / step + 1)
        ncuts = sum(1 for x in cuts if fr[0] < x <= fr[-1])
        accept = (ncuts >= 1 and cov >= 0.5) or (dur >= min_seconds and cov >= 0.6)
        if not accept and dur < 3.0:
            continue
        label = Counter(d["text"] for d in c["dets"]).most_common(1)[0][0]
        for d in c["dets"]:
            d["label"] = label
        why = f"seen {dur:.0f}s, across {ncuts} scene cut(s), coverage {cov:.0%}"
        out.append((label, c["dets"], accept, why))
    return out


def _merge_static_sums(t, mode):
    """Opacity statistics were split by the jittered positions; shift each part onto the mode."""
    if len(t.key_sums) <= 1:
        return
    S = np.zeros_like(t.sum)
    Q = np.zeros_like(t.sum)
    C = np.zeros(t.sum.shape[:2], np.float64)
    for (x, y), (s_, q_, c_) in t.key_sums.items():
        dx, dy = x - int(mode[0]), y - int(mode[1])
        if abs(dx) > 4 or abs(dy) > 4:
            continue
        # crop taken at (x, y): pixel (u, v) of it is frame (x+u, y+v) = mode crop pixel (u+dx, v+dy)
        H, W = C.shape
        u0, u1 = max(0, -dx), min(W, W - dx)
        v0, v1 = max(0, -dy), min(H, H - dy)
        S[v0 + dy:v1 + dy, u0 + dx:u1 + dx] += s_[v0:v1, u0:u1]
        Q[v0 + dy:v1 + dy, u0 + dx:u1 + dx] += q_[v0:v1, u0:u1]
        C[v0 + dy:v1 + dy, u0 + dx:u1 + dx] += c_
    full = C >= C.max() * 0.999
    if C.max() < 12:
        return
    n = C.max()
    # border pixels that miss some shifted parts: rescale to the full count
    scale = np.where(C > 0, n / np.maximum(C, 1), 0)[..., None]
    t.sum, t.sumsq, t.nacc = S * scale, Q * scale, int(n)
    t.sum[~full & (C == 0)] = 0


def _box_iou(p, t, q, o):
    x0, y0 = max(p[0] + t.px, q[0] + o.px), max(p[1] + t.py, q[1] + o.py)
    x1 = min(p[0] + t.TW - t.px, q[0] + o.TW - o.px)
    y1 = min(p[1] + t.TH - t.py, q[1] + o.TH - o.py)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    a = (t.TW - 2 * t.px) * (t.TH - 2 * t.py)
    b = (o.TW - 2 * o.px) * (o.TH - 2 * o.py)
    return inter / float(min(a, b) + 1e-6)


def _dedupe(tracks, log):
    """OCR boxes of one overlay can differ in size (a truncated read, two words read apart), which
    can split it into two tracks that follow the same text. Keep the better one."""
    drop = set()
    for a in range(len(tracks)):
        for b in range(a + 1, len(tracks)):
            t, o = tracks[a], tracks[b]
            if a in drop or b in drop:
                continue
            both = np.flatnonzero((t.pos[:, 0] > ABSENT) & (o.pos[:, 0] > ABSENT))
            if len(both) < 3:
                continue
            ious = [_box_iou(t.pos[i], t, o.pos[i], o) for i in both[:: max(1, len(both) // 200)]]
            if np.mean(np.array(ious) > 0.5) < 0.8:
                continue
            def quality(x):
                pres = x.pos[:, 0] > ABSENT
                # the one that follows the text through more of the video wins (in 5% steps);
                # then match strength, then box width
                return (int(20 * pres.sum() / len(pres)), round(float(x.score[pres].mean()), 2),
                        x.TW - 2 * x.px)
            ka, kb = quality(t), quality(o)
            loser, winner = (b, a) if ka >= kb else (a, b)
            w, l = tracks[winner], tracks[loser]
            fill = (w.pos[:, 0] <= ABSENT) & (l.pos[:, 0] > ABSENT)
            if fill.any():      # frames only the duplicate saw: map them across by the median offset
                off = np.round(np.median(w.pos[both] - l.pos[both], axis=0)).astype(np.int64)
                w.pos[fill] = l.pos[fill] + off
                w.low[fill] = True
            drop.add(loser)
            if t.label == o.label:
                log(f"[analyze] merged duplicate tracks of {w.label!r}")
            else:
                log(f"[analyze] {l.label!r} and {w.label!r} follow the same text; kept {w.label!r}")
    kept = [t for k, t in enumerate(tracks) if k not in drop]
    for k, t in enumerate(kept):
        t.id = k
    return kept


def _occlusion_hold(t, tracks, n):
    """Keep an overlay masked while another tracked overlay sits on top of it."""
    def covered(i, q):
        for o in tracks:
            if o is t or o.label == t.label or o.pos[i, 0] <= ABSENT:
                continue
            if q[0] < o.pos[i, 0] + o.TW and o.pos[i, 0] < q[0] + t.TW and \
                    q[1] < o.pos[i, 1] + o.TH and o.pos[i, 1] < q[1] + t.TH:
                return True
        return False

    for rng, d in ((range(1, n), -1), (range(n - 2, -1, -1), 1)):
        for i in rng:
            if t.pos[i, 0] <= ABSENT and t.pos[i + d, 0] > ABSENT and covered(i, t.pos[i + d]):
                t.pos[i], t.low[i] = t.pos[i + d], True


def _extrapolate_edges(t, W, H):
    """A moving overlay entering or leaving the frame is only partly visible and the template
    stops matching; continue it at constant velocity until it is fully outside. Weak matches
    at the end of a run are replaced by the extrapolation, but only next to a frame border."""
    n = len(t.pos)
    present = t.pos[:, 0] > ABSENT
    if not present.any():
        return
    weak = min(T_HI, 0.75 * float(np.median(t.score[present])))
    size, lim, pd = (t.TW - 2 * t.px, t.TH - 2 * t.py), (W, H), (t.px, t.py)

    def partly_out(q):      # judged on the text itself, not the padded box
        return q[0] + t.px < 0 or q[1] + t.py < 0 or q[0] + t.TW - t.px > W or q[1] + t.TH - t.py > H

    for direction in (1, -1):
        ends = [i for i in range(n) if present[i] and not (0 <= i + direction < n and present[i + direction])]
        for e0 in ends:
            e, steps = e0, 0
            while 0 <= e < n and present[e] and steps < 3 * t.TW and (partly_out(t.pos[e]) or t.score[e] < weak):
                e -= direction
                steps += 1
            if not (0 <= e < n) or not present[e]:
                continue
            back = [e - direction * k for k in range(6) if 0 <= e - direction * k < n and present[e - direction * k]]
            if len(back) < 3:
                continue
            v = (t.pos[back[0]] - t.pos[back[-1]]) / float(abs(back[0] - back[-1]))   # per step in `direction`
            if np.abs(v).max() < 0.3:
                continue
            p = t.pos[e].astype(np.float64)
            near = False
            for a in (0, 1):   # only when the overlay is heading for a nearby frame border
                if abs(v[a]) > 0.3:
                    dist = p[a] + pd[a] if v[a] < 0 else lim[a] - (p[a] + pd[a] + size[a])
                    near |= dist < 0.6 * size[a]
            if not near:
                continue
            for k in range(1, 4 * max(t.TW, t.TH)):
                i = e + direction * k
                if not (0 <= i < n):
                    break
                if present[i] and steps <= 0:
                    break
                steps -= 1
                q = np.round(p + v * k).astype(np.int64)
                visible = q[0] + t.TW - t.px > 0 and q[1] + t.TH - t.py > 0 and q[0] + t.px < W and q[1] + t.py < H
                if not visible:
                    if present[i]:
                        t.pos[i] = ABSENT
                        present[i] = False
                    continue
                t.pos[i], t.low[i] = q, True
                present[i] = True


def _coast(t, W, H, limit, bounce):
    """Carry a moving overlay through stretches where it cannot be matched (for example white
    text over a white background) using its motion model: constant velocity, reflecting at the
    walls it has been seen to bounce off."""
    n = len(t.pos)
    present = t.pos[:, 0] > ABSENT
    idx = np.flatnonzero(present)
    if len(idx) < 4:
        return
    lo = np.array([-t.px, -t.py], dtype=np.float64)
    hi = np.array([W - t.TW + t.px, H - t.TH + t.py], dtype=np.float64)
    weak = min(T_HI, 0.75 * float(np.median(t.score[idx])))
    good = idx[t.score[idx] >= weak]          # walls are learned from reliable frames only
    if len(good) < 4:
        return
    P = t.pos[good].astype(np.float64)
    v_all = np.diff(P, axis=0)[np.diff(good) == 1]
    P_adj = P[1:][np.diff(good) == 1]
    for a in (0, 1):
        va = v_all[:, a]
        nz = np.flatnonzero(np.abs(va) > 0)
        for k0, k1 in zip(nz[:-1], nz[1:]):
            if va[k0] < 0 < va[k1]:
                lo[a] = max(lo[a], P_adj[k0:k1 + 1, a].min()) if P_adj[k0:k1 + 1, a].min() > lo[a] + 2 else lo[a]
            elif va[k0] > 0 > va[k1]:
                hi[a] = min(hi[a], P_adj[k0:k1 + 1, a].max()) if P_adj[k0:k1 + 1, a].max() < hi[a] - 2 else hi[a]
    for direction in (1, -1):
        ends = [i for i in range(n) if present[i] and not (0 <= i + direction < n and present[i + direction])]
        for e in ends:
            steps = 0      # weak matches at the end of the run are re-derived from the model
            while 0 <= e - direction < n and present[e - direction] and t.score[e] < weak and steps < limit // 2:
                t.pos[e] = ABSENT
                present[e] = False
                e -= direction
                steps += 1
            back = [e - direction * k for k in range(6) if 0 <= e - direction * k < n and present[e - direction * k]]
            if len(back) < 3:
                continue
            v = (t.pos[back[0]] - t.pos[back[-1]]) / float(abs(back[0] - back[-1]))
            if np.abs(v).max() < 0.3:
                continue
            p = t.pos[e].astype(np.float64)
            for k in range(1, limit + steps + 1):
                i = e + direction * k
                if not (0 <= i < n) or present[i]:
                    break
                p = p + v
                if bounce:
                    for a in (0, 1):
                        if p[a] < lo[a]:
                            p[a], v[a] = 2 * lo[a] - p[a], -v[a]
                        elif p[a] > hi[a]:
                            p[a], v[a] = 2 * hi[a] - p[a], -v[a]
                q = np.round(p).astype(np.int64)
                if not (q[0] + t.TW - t.px > 0 and q[1] + t.TH - t.py > 0 and q[0] + t.px < W and q[1] + t.py < H):
                    break
                t.pos[i], t.low[i] = q, True
                present[i] = True


def _spans(pos):
    spans = []
    for i in range(len(pos)):
        x, y = int(pos[i, 0]), int(pos[i, 1])
        if x <= ABSENT:
            continue
        if spans and spans[-1][1] == i - 1 and spans[-1][2] == x and spans[-1][3] == y:
            spans[-1][1] = i
        else:
            spans.append([i, i, x, y])
    return spans


def positions_from_spans(spans, n):
    pos = np.full((n, 2), ABSENT, dtype=np.int64)
    for f0, f1, x, y in spans:
        pos[f0:f1 + 1] = (x, y)
    return pos


def analyze(video, out_json, texts=None, discover=False, sample_fps=3.0, enhance=False,
            use_gpu=True, match_thr=0.7, min_seconds=8.0, ocr=None, log=print):
    info = vio.probe(video)
    if info["high_bit_depth"]:
        log("warning: source is high bit depth; output will be 8-bit yuv420p")
    W, H, fps = info["width"], info["height"], info["fps"]
    step = max(1, int(round(fps / sample_fps)))
    texts = [t for t in (texts or []) if t.strip()]
    if not texts:
        discover = True
    t0 = time.perf_counter()
    if ocr is None:
        ocr = OCRPool(use_gpu=use_gpu, enhance=enhance)
    ocr.enhance = enhance
    log(f"[analyze] {os.path.basename(video)} {W}x{H} @ {fps:.2f} fps, OCR {ocr.model} on {ocr.provider}"
        f" x{len(ocr.engines)}, every {step} frames (OCR ready in {time.perf_counter() - t0:.1f} s)")

    # ---- pass A: sparse OCR + scene cuts -------------------------------------------------
    dets, cuts, n = [], [], 0
    prof.phase("analyze: pass A")
    t_a = time.perf_counter()

    def sampled():   # decode + cut detection here; OCR of the sampled frames runs on every GPU
        nonlocal n
        prev = None
        for i, _, f in vio.read_frames(video, convert=False):
            n = i + 1
            # cut detection on the luma plane, every 4th pixel: only sampled frames need BGR
            small = cv2.resize(vio.luma(f)[::4, ::4], (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
            if prev is not None and float(np.abs(small - prev).mean()) > CUT_THR:
                cuts.append(i)
            prev = small
            if i % step == 0:
                img = f.to_ndarray(format="bgr24")
                yield (i, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), img

    for (i, gray), _, found in ocr.imap(sampled()):
        for d in found:
            label, src = None, None
            if texts:
                label, _ = match_text(d["text"], texts, match_thr)
                src = "list" if label else None
            if label is None and discover:
                src = "discovered"
            if src is None:
                continue
            E = int(1.3 * d["h"]) + 40
            cx0, cy0 = max(0, d["x"] - E), max(0, d["y"] - E)
            cx1, cy1 = min(W, d["x"] + d["w"] + E), min(H, d["y"] + d["h"] + E)
            d.update(f=i, label=label, src=src, crop=gray[cy0:cy1, cx0:cx1].copy(), cx=cx0, cy=cy0)
            dets.append(d)
    log(f"[analyze] pass A: {n} frames, {len(dets)} candidate detections, {len(cuts)} scene cuts "
        f"({time.perf_counter() - t_a:.1f} s)")

    groups = []
    by_label = {}
    for d in dets:
        if d["src"] == "list":
            by_label.setdefault(d["label"], []).append(d)
    for label, ds in by_label.items():
        groups.append((label, "list", ds, True, ""))
    if discover:
        disc = [d for d in dets if d["src"] == "discovered"]
        for label, ds, accept, why in _discover(disc, cuts, fps, step, min_seconds):
            groups.append((label, "discovered", ds, accept, why))

    min_dets = 3 if n / step >= 10 else 2
    tracks = []
    for label, src, ds, enabled, why in groups:
        for g in _group(ds):
            if len(g) < min_dets:
                continue
            t = _T(len(tracks), label, src, g)
            t.enabled, t.note = enabled, why
            if _build_template(t):
                tracks.append(t)
    for k, t in enumerate(tracks):
        t.id = k
        t.pos = np.full((n, 2), ABSENT, dtype=np.int64)
        t.score = np.zeros(n, dtype=np.float32)
        t.low = np.zeros(n, dtype=bool)
        t.cpos = np.full((n, 2), ABSENT, dtype=np.int64)
        t.cscore = np.zeros(n, dtype=np.float32)
        t.sum = np.zeros((t.TH, t.TW, 3), np.float64)
        t.sumsq = np.zeros((t.TH, t.TW, 3), np.float64)
        t.nacc, t.seen, t.stable = 0, 0, None
        t.acc_pos = Counter()
        t.key_sums = {}
        t.acc_every = max(1, (t.cf[-1] - t.cf[0] + step) // 500)
        th = cv2.resize(t.tmpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        t.tmpl_half = th if min(th.shape) >= 6 and float(th.std()) > 1e-3 else None
    log(f"[analyze] {len(tracks)} track(s): " + ", ".join(repr(t.label) for t in tracks))

    # ---- pass B: dense template tracking + alpha statistics ------------------------------
    prof.phase("analyze: pass B")
    t_b = time.perf_counter()
    if tracks:
        B = max(t.R + max(t.px, t.py) for t in tracks) + 8
        def plan(t):
            """Track t's candidate windows on frame i: (j, p1, [(ox, oy), ...] in the padded frame)."""
            j = bisect.bisect_right(t.cf, i) - 1
            cands = []
            p1 = t.pos[i - 1] if i >= 1 and t.pos[i - 1, 0] > ABSENT else None
            if p1 is not None:
                if i >= 2 and t.pos[i - 2, 0] > ABSENT:
                    cands.append(2 * p1 - t.pos[i - 2])
                cands.append(p1)
            if j >= 0:
                cands.append(t.cp[j])
            if j + 1 < len(t.cf):
                cands.append(t.cp[j + 1])
                if j >= 0:
                    u = (i - t.cf[j]) / float(t.cf[j + 1] - t.cf[j])
                    cands.append(np.round(t.cp[j] + u * (t.cp[j + 1] - t.cp[j])).astype(np.int64))
            if p1 is None and i % step == 0 and not (j >= 0 and t.cf[j] == i):
                # not tracked here and OCR did not see it: look for the template anywhere
                if t.tmpl_half is not None and t.tmpl_half.shape[0] <= ghalf.shape[0] \
                        and t.tmpl_half.shape[1] <= ghalf.shape[1]:
                    res = cv2.matchTemplate(ghalf, t.tmpl_half, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, loc = cv2.minMaxLoc(res)
                    if mx >= T_LO:
                        cands.append(np.array([2 * loc[0], 2 * loc[1]], dtype=np.int64))
            uniq = []
            for c in cands:
                if all(np.abs(c - q).max() > 3 for q in uniq):
                    uniq.append(c)
            ww, wh = t.TW + 2 * t.R, t.TH + 2 * t.R
            if ww > gp.shape[1] or wh > gp.shape[0]:
                return j, p1, []
            # every candidate: identical marks (same text, several places) need the max over all
            return j, p1, [(int(np.clip(c[0] - t.R + B, 0, gp.shape[1] - ww)),
                            int(np.clip(c[1] - t.R + B, 0, gp.shape[0] - wh))) for c in uniq]

        def search(t):
            """CPU (OpenCV): best template match for track t over its candidate windows."""
            j, p1, origins = plan(t)
            best, bpos = -1.0, None
            ww, wh = t.TW + 2 * t.R, t.TH + 2 * t.R
            for ox, oy in origins:
                res = cv2.matchTemplate(grad(gp[oy:oy + wh, ox:ox + ww]), t.tmpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                if mx > best:
                    best, bpos = mx, np.array([ox - B + loc[0], oy - B + loc[1]], dtype=np.int64)
            return j, p1, best, bpos

        def search_all_gpu():
            plans = [plan(t) for t in tracks]
            gm.frame(gp)
            for t, (_, _, origins) in zip(tracks, plans):
                if origins:
                    gm.search(t, origins)
            res = iter(gm.results())
            out = []
            for j, p1, origins in plans:
                if origins:
                    _, best, (x, y) = next(res)
                    out.append((j, p1, best, np.array([x - B, y - B], dtype=np.int64)))
                else:
                    out.append((j, p1, -1.0, None))
            return out

        def prepared():   # background thread: decode, gray, padding
            for i, _, img in vio.read_frames(video):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                yield i, img, gray, cv2.copyMakeBorder(gray, B, B, B, B, cv2.BORDER_REPLICATE)

        # GPU: every track's windows in one batch per frame (539 s of CPU matchTemplate on 072q-1
        # on Kaggle's CPU). CPU: tracks side by side in threads (OpenCV releases the GIL). Track
        # state is updated below, in track order.
        dev = _match_device(use_gpu)
        gm = _TorchMatch(tracks, dev) if dev else None
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(min(len(tracks), os.cpu_count() or 1))
        log(f"[analyze] pass B: template search on {dev or 'cpu (OpenCV)'}")
        for i, img, gray, gp in vio.prefetch(prepared()):
            ip, ghalf, found = None, None, []
            if i % step == 0 and any(i < 1 or t.pos[i - 1, 0] <= ABSENT for t in tracks):
                ghalf = grad(cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
            if gm is not None:
                results = search_all_gpu()
            else:
                results = ex.map(search, tracks) if len(tracks) > 1 else map(search, tracks)
            for t, (j, p1, best, bpos) in zip(tracks, results):
                if bpos is None:
                    continue
                inside = j >= 0 and (t.cf[j] == i or (j + 1 < len(t.cf) and t.cf[j + 1] - t.cf[j] <= 2 * step + 1))
                present = best >= T_HI or inside or (best >= T_LO and p1 is not None)
                if not present:
                    t.stable = None
                    t.cpos[i], t.cscore[i] = bpos, best
                    continue
                if t.stable is not None and np.abs(bpos - t.stable).max() <= 1:
                    bpos = t.stable
                else:
                    t.stable = bpos
                t.pos[i], t.score[i], t.low[i] = bpos, best, best < T_LO
                if best >= T_ACC:
                    found.append((t, bpos))
            # opacity statistics: skip frames where another overlay overlaps this one
            for t, bpos in found:
                clash = any(o is not t and o.label != t.label and o.pos[i, 0] > ABSENT
                            and abs(o.pos[i, 0] - bpos[0]) < (o.TW + t.TW) / 2 + abs(o.TW - t.TW) / 2
                            and bpos[0] < o.pos[i, 0] + o.TW and o.pos[i, 0] < bpos[0] + t.TW
                            and bpos[1] < o.pos[i, 1] + o.TH and o.pos[i, 1] < bpos[1] + t.TH
                            for o in tracks)
                if clash:
                    continue
                t.seen += 1
                if t.seen % t.acc_every == 0:
                    if ip is None:
                        ip = cv2.copyMakeBorder(img, B, B, B, B, cv2.BORDER_REPLICATE)
                    x, y = int(bpos[0]) + B, int(bpos[1]) + B
                    if 0 <= x and 0 <= y and x + t.TW <= ip.shape[1] and y + t.TH <= ip.shape[0]:
                        crop = ip[y:y + t.TH, x:x + t.TW].astype(np.float64)
                        t.sum += crop
                        t.sumsq += crop * crop
                        t.nacc += 1
                        key = (int(bpos[0]), int(bpos[1]))
                        t.acc_pos[key] += 1
                        if key in t.key_sums or len(t.key_sums) < 8:
                            ks = t.key_sums.setdefault(key, [np.zeros_like(t.sum), np.zeros_like(t.sum), 0])
                            ks[0] += crop
                            ks[1] += crop * crop
                            ks[2] += 1
        ex.shutdown()

    if tracks:
        log(f"[analyze] pass B: {time.perf_counter() - t_b:.1f} s")
    prof.phase("analyze: finish")
    tracks = _dedupe(tracks, log)

    # ---- finish: bridge gaps, classify motion, estimate opacity --------------------------
    side = os.path.splitext(out_json)[0] + ".d"
    os.makedirs(side, exist_ok=True)
    out_tracks = []
    for t in tracks:
        for i in range(n - 2, -1, -1):   # extend backwards where the template still matched weakly
            if t.pos[i, 0] <= ABSENT and t.pos[i + 1, 0] > ABSENT and t.cscore[i] >= T_LO \
                    and np.abs(t.cpos[i] - t.pos[i + 1]).max() <= t.R:
                near = np.abs(t.cpos[i] - t.pos[i + 1]).max() <= 1
                t.pos[i] = t.pos[i + 1] if near else t.cpos[i]
                t.score[i] = t.cscore[i]
        _occlusion_hold(t, tracks, n)
        idx = np.flatnonzero(t.pos[:, 0] > ABSENT)
        if len(idx) == 0:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if 0 < gap <= 2 * step:
                for k in range(a + 1, b):
                    u = (k - a) / float(b - a)
                    same = np.abs(t.pos[a] - t.pos[b]).max() <= 2
                    t.pos[k] = t.pos[a] if same else np.round(t.pos[a] + u * (t.pos[b] - t.pos[a]))
                    t.low[k] = True
        idx = np.flatnonzero(t.pos[:, 0] > ABSENT)
        P = t.pos[idx]
        mode = np.array(t.acc_pos.most_common(1)[0][0] if t.acc_pos else
                        np.round(np.median(P, axis=0)), dtype=np.int64)
        near_mode = float((np.abs(P - mode).max(1) <= 4).mean())
        if (P.max(0) - P.min(0)).max() <= 2 or near_mode >= 0.9:
            # static; small jitter is template ambiguity on faint text, not motion
            motion = "static"
            t.pos[idx] = mode
            _merge_static_sums(t, mode)
        else:
            adj = np.diff(idx) == 1
            dd = np.abs(np.diff(P, axis=0)).max(1)[adj]
            frac = float((dd > 0).mean()) if len(dd) else 0.0
            if frac < 0.08:
                motion = "teleport"
            else:
                _extrapolate_edges(t, W, H)
                idx = np.flatnonzero(t.pos[:, 0] > ABSENT)
                P = t.pos[idx]
                v = np.diff(P, axis=0)[np.diff(idx) == 1]
                flips = 0
                if len(v) > 2:
                    v = v[np.abs(v).max(1) <= 3 * np.median(np.abs(v).max(1)) + 2]
                    flips = sum(int((np.diff(np.sign(v[np.abs(v[:, a]) > 0, a])) != 0).sum()) for a in (0, 1)
                                if np.median(np.abs(v[:, a])) > 0.5)
                motion = "bounce" if flips else "linear"
                _coast(t, W, H, int(round(2 * fps)), motion == "bounce")
                idx = np.flatnonzero(t.pos[:, 0] > ABSENT)
        est = alpha_mod.estimate(t.sum, t.sumsq, t.nacc, t.text_h, t.tmpl)
        base = os.path.join(side, f"t{t.id}")
        d = 2 + int(round(t.text_h / 40.0))
        if est["ok"]:
            a = est["alpha"]
            mask = cv2.dilate((a > 0.1 * est["median"]).astype(np.uint8), np.ones((2 * d + 1, 2 * d + 1), np.uint8))
            opaque = est["median"] >= 0.7 or est["color_mad"] > 40
            tier = "fill" if opaque else "alpha"
            cv2.imwrite(base + "_alpha.png", (a * 255).round().astype(np.uint8))
        else:
            mask = np.zeros((t.TH, t.TW), np.uint8)
            ix, iy = max(0, t.px // 2), max(0, t.py - 3)
            mask[iy:t.TH - iy, ix:t.TW - ix] = 1
            tier = "fill"
        cv2.imwrite(base + "_mask.png", mask * 255)
        tv = t.tmpl / (t.tmpl.max() + 1e-6) * 255
        cv2.imwrite(base + "_template.png", tv.astype(np.uint8))
        out_tracks.append({
            "id": t.id, "label": t.label, "source": t.src, "enabled": bool(t.enabled),
            "motion": motion, "tier": tier, "w": t.TW, "h": t.TH, "pad": [t.px, t.py],
            "frames_present": int(len(idx)), "first": int(idx[0]), "last": int(idx[-1]),
            "confidence": float(t.score[idx].mean()), "low_conf_frames": int(t.low[idx].sum()),
            "alpha": {"ok": bool(est["ok"]), "median": est.get("median"), "samples": est["n"],
                      "color": est.get("color"), "why": est.get("why", "")},
            "alpha_gain": 1.0, "note": t.note,
            "files": {"alpha": os.path.basename(base) + "_alpha.png" if est["ok"] else None,
                      "mask": os.path.basename(base) + "_mask.png"},
            "spans": _spans(t.pos),
        })
    doc = {"version": 1, "video": os.path.abspath(video), "info": info, "frames": n, "cuts": cuts,
           "sample_step": step, "texts": texts, "sidecar": os.path.basename(side), "tracks": out_tracks}
    with open(out_json, "w") as fh:
        json.dump(doc, fh)
    for tr in out_tracks:
        am = tr["alpha"]["median"]
        log(f"  track {tr['id']}: {tr['label']!r} [{tr['source']}{'' if tr['enabled'] else ', DISABLED - review'}] "
            f"{tr['motion']}, tier={tr['tier']}, frames {tr['first']}-{tr['last']} ({tr['frames_present']}), "
            f"conf {tr['confidence']:.2f}, opacity {'%.2f' % am if am is not None else 'n/a'}"
            + (f" | {tr['note']}" if tr["note"] else ""))
    return doc


def load(tracks_json):
    with open(tracks_json) as fh:
        doc = json.load(fh)
    side = os.path.join(os.path.dirname(os.path.abspath(tracks_json)), doc["sidecar"])
    for t in doc["tracks"]:
        t["pos"] = positions_from_spans(t["spans"], doc["frames"])
        t["mask_img"] = cv2.imread(os.path.join(side, t["files"]["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        t["alpha_img"] = None
        if t["files"].get("alpha"):
            a = cv2.imread(os.path.join(side, t["files"]["alpha"]), cv2.IMREAD_GRAYSCALE)
            t["alpha_img"] = a.astype(np.float32) / 255.0
    return doc


def preview(video, tracks_json, out, max_side=960):
    doc = load(tracks_json)
    info = doc["info"]
    s = min(1.0, max_side / float(max(info["width"], info["height"])))
    ow, oh = int(info["width"] * s) // 2 * 2, int(info["height"] * s) // 2 * 2
    wr = vio.VideoWriter(out, ow, oh, info["fps"], info["time_base"], crf=27, preset="veryfast")
    for i, pts, img in vio.read_frames(video):
        for t in doc["tracks"]:
            if i >= len(t["pos"]) or t["pos"][i, 0] <= ABSENT:
                continue
            x, y = int(t["pos"][i, 0]), int(t["pos"][i, 1])
            col = (0, 0, 255) if t["enabled"] else (0, 220, 255)
            cv2.rectangle(img, (x, y), (x + t["w"], y + t["h"]), col, 2)
            cv2.putText(img, f"{t['id']}:{t['tier']}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        wr.write(cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA) if s < 1 else img, pts)
    wr.close()
    return out
