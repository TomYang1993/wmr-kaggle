"""Remove pass: one decode, one encode. Per frame:
  tier "alpha": invert the blend with the estimated opacity map
  tier "fill" : borrow real pixels from nearby frames of the same shot (global-motion
                compensated), then fall back to spatial inpainting kept stable over time.
Optionally, fill-tier regions are pre-computed by ProPainter (see propainter.py)."""
import time
from collections import deque

import cv2
import numpy as np

from . import track as tk
from . import vio

OFFSETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def clip_box(x, y, w, h, W, H):
    fx0, fy0, fx1, fy1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    return (slice(fy0, fy1), slice(fx0, fx1)), (slice(fy0 - y, fy1 - y), slice(fx0 - x, fx1 - x))


def frame_masks(doc, i, W, H, tiers=("fill",)):
    """Union fill mask for frame i (bool HxW) or None."""
    m = None
    for t in doc["tracks"]:
        if not t["enabled"] or i >= len(t["pos"]) or t["pos"][i, 0] <= tk.ABSENT:
            continue
        if t["tier"] not in tiers:
            continue
        b = clip_box(int(t["pos"][i, 0]), int(t["pos"][i, 1]), t["w"], t["h"], W, H)
        if b is None:
            continue
        if m is None:
            m = np.zeros((H, W), dtype=bool)
        m[b[0]] |= t["mask_img"][b[1]]
    return m


def apply_alpha(img, mask, doc, i, a_max=0.85):
    """Invert semi-transparent overlays in place; pixels too opaque to invert join the fill mask."""
    H, W = img.shape[:2]
    for t in doc["tracks"]:
        if not t["enabled"] or t["tier"] != "alpha" or i >= len(t["pos"]) or t["pos"][i, 0] <= tk.ABSENT:
            continue
        b = clip_box(int(t["pos"][i, 0]), int(t["pos"][i, 1]), t["w"], t["h"], W, H)
        if b is None:
            continue
        a = np.clip(t["alpha_img"][b[1]] * float(t.get("alpha_gain", 1.0)), 0, 1)
        Wc = np.array(t["alpha"]["color"], dtype=np.float32)
        J = img[b[0]].astype(np.float32)
        inv = (J - a[..., None] * Wc) / np.maximum(1 - a[..., None], 0.1)
        use = (a > 0) & (a <= a_max)
        J[use] = np.clip(inv[use] + 0.5, 0, 255)
        img[b[0]] = J.astype(np.uint8)
        hard = a > a_max
        if hard.any():
            if mask is None:
                mask = np.zeros((H, W), dtype=bool)
            mask[b[0]] |= cv2.dilate(hard.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return mask


class Motion:
    """Global frame-to-frame similarity transform from sparse optical flow (masked areas ignored)."""

    def __init__(self, W, H):
        self.s = min(1.0, 480.0 / max(W, H))
        self.prev = None
        self.C = np.eye(3)

    def step(self, img, mask, cut):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.s < 1:
            g = cv2.resize(g, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
        fm = None
        if mask is not None:
            fm = cv2.resize(mask.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            fm = (cv2.dilate(fm, np.ones((5, 5), np.uint8)) == 0).astype(np.uint8) * 255
        T = np.eye(3)
        if self.prev is not None and not cut:
            pg, pm = self.prev
            pts = cv2.goodFeaturesToTrack(pg, 200, 0.01, 8, mask=pm)
            if pts is not None and len(pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, pts, None)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    A, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                                       ransacReprojThreshold=1.5)
                    if A is not None:
                        T[:2] = A
                        T[:2, 2] /= self.s
        self.C = np.eye(3) if cut else T @ self.C
        self.prev = (g, fm)
        return self.C.copy()


def _warp_to(src_img, src_mask, M, roi):
    """Bring the roi of the target frame from a source frame through the 3x3 map M (src->dst)."""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    if np.abs(M - np.eye(3)).max() < 0.02:
        return src_img[y0:y1, x0:x1], (None if src_mask is None else src_mask[y0:y1, x0:x1])
    Tr = np.eye(3)
    Tr[0, 2], Tr[1, 2] = -x0, -y0
    A = (Tr @ M)[:2]
    wi = cv2.warpAffine(src_img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    sm = np.zeros(src_img.shape[:2], np.uint8) if src_mask is None else src_mask.astype(np.uint8) * 255
    wm = cv2.warpAffine(sm, A, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=255) > 0
    return wi, wm


def _sources(t, buf):
    """Other frames of the same shot, nearest first (the order borrowing prefers them in)."""
    by_index = {e["i"]: e for e in buf}
    for off in OFFSETS:
        for sgn in (-1, 1):
            s = by_index.get(t["i"] + sgn * off)
            if s is not None and s["shot"] == t["shot"]:
                yield s


def _borrow_cv(t, buf, roi, out, remaining, ring, photo_thr, stats):
    for s in _sources(t, buf):
        M = t["C"] @ np.linalg.inv(s["C"])
        wi, wm = _warp_to(s["img"], s["mask"], M, roi)
        if wm is not None:
            wm = cv2.dilate(wm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            usable = remaining & ~wm
            chk = ring & ~wm
        else:
            usable, chk = remaining, ring
        if not usable.any() or chk.sum() < 30:
            continue
        diff = np.abs(wi[chk].astype(np.int16) - out[chk].astype(np.int16)).mean()
        if diff > photo_thr:
            continue
        out[usable] = wi[usable]
        stats["borrowed"] += int(usable.sum())
        remaining &= ~usable
        if not remaining.any():
            break
    return out, remaining


def _borrow_torch(t, buf, roi, out, remaining, ring, photo_thr, stats):
    """_borrow_cv on the GPU, all sources at once: same sources, order and photometric check.
    The greedy loop's accept/reject test reads only the ring, which borrowing never writes, so
    every source is judged together and each pixel takes the first accepted source that covers
    it: the same result as the sequential loop. One batched warp over a shared source region
    (bbox of where the roi maps in every source, plus a pixel for bilinear neighbours) and a
    fixed number of host<->device copies per frame, however many sources there are.
    Frames of the window are already on the device (e["g"], e["gm"]); bilinear for pixels,
    nearest for masks, outside the frame counts as masked.
    ponytail: the shared region grows with motion inside the window; at worst K full frames
    in float32 (~0.5 GB at 1080p for 20 sources). Per-source regions if that ever matters."""
    import torch
    import torch.nn.functional as F
    srcs = list(_sources(t, buf))
    if not srcs:
        return out, remaining
    x0, y0, x1, y1 = roi
    h, w = y1 - y0, x1 - x0
    H, W = t["mask"].shape
    K = len(srcs)
    Ai = np.stack([np.linalg.inv(t["C"] @ np.linalg.inv(s["C"]))[:2] for s in srcs])   # K,2,3 dst -> src
    c = Ai @ np.array([[x0, x1 - 1, x0, x1 - 1], [y0, y0, y1 - 1, y1 - 1], [1, 1, 1, 1]], float)
    cx0, cy0 = max(0, int(np.floor(c[:, 0].min())) - 1), max(0, int(np.floor(c[:, 1].min())) - 1)
    cx1, cy1 = min(W, int(np.ceil(c[:, 0].max())) + 2), min(H, int(np.ceil(c[:, 1].max())) + 2)
    if cx1 - cx0 < 2 or cy1 - cy0 < 2:   # the roi maps outside every source frame
        return out, remaining
    dev = t["g"].device
    pk = torch.from_numpy(np.concatenate([out, remaining[..., None], ring[..., None]], -1)).to(dev)
    o, rem, rng = pk[..., :3].float(), pk[..., 3].bool(), pk[..., 4].bool()
    A = torch.from_numpy(Ai.astype(np.float32)).to(dev)
    yy, xx = torch.meshgrid(torch.arange(y0, y1, device=dev, dtype=torch.float32),
                            torch.arange(x0, x1, device=dev, dtype=torch.float32), indexing="ij")
    P = torch.stack([xx, yy, torch.ones_like(xx)]).reshape(3, -1)
    sx, sy = (A @ P).reshape(K, 2, h, w).unbind(1)
    grid = torch.stack([2 * (sx - cx0) / (cx1 - cx0 - 1) - 1, 2 * (sy - cy0) / (cy1 - cy0 - 1) - 1], -1)
    S = torch.stack([s["g"][cy0:cy1, cx0:cx1] for s in srcs])                   # K,ch,cw,3 uint8
    wi = F.grid_sample(S.permute(0, 3, 1, 2).float(), grid, mode="bilinear", padding_mode="zeros",
                       align_corners=True).permute(0, 2, 3, 1).round()       # K,h,w,3
    ok_px = torch.stack([torch.ones((cy1 - cy0, cx1 - cx0), dtype=torch.bool, device=dev) if s["gm"] is None
                         else ~s["gm"][cy0:cy1, cx0:cx1] for s in srcs])
    wm = F.grid_sample(ok_px[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                       align_corners=True)[:, 0] < 0.5
    wm = F.max_pool2d(wm.float()[:, None], 3, 1, 1)[:, 0] > 0                 # K,h,w
    chk = rng & ~wm
    n_chk = chk.sum((1, 2))
    diff = ((wi - o).abs().sum(-1) * chk).sum((1, 2)) / (3 * n_chk.clamp(min=1))
    ok = (n_chk >= 30) & (diff <= photo_thr)
    cand = (ok[:, None, None] & rem & ~wm).to(torch.uint8)
    first = cand.argmax(0)                    # first accepted source per pixel (argmax takes the first max)
    take = cand.amax(0).bool()
    pick = wi.gather(0, first[None, ..., None].expand(1, h, w, 3))[0]
    o = torch.where(take[..., None], pick, o)
    res = torch.cat([o.to(torch.uint8), (rem & ~take).to(torch.uint8)[..., None]], -1).cpu().numpy()
    o8, rem_np = res[..., :3], res[..., 3].astype(bool)
    stats["borrowed"] += int(remaining.sum() - rem_np.sum())
    return o8, rem_np


def _regions(mask, pad):
    """Padded boxes (x0, y0, x1, y1) around the separate parts of the mask. Parts closer than
    2 * pad end up in one box; overlapping boxes are merged, so the boxes are disjoint.
    Works on an 8x smaller mask (a full-size dilate + labelling cost ~11 ms per 1080p frame on
    Kaggle's CPU), so boxes are rounded out to 8 px blocks."""
    H, W = mask.shape
    F = 8   # block any(): 8 bools along x are one uint64, then any() over 8 rows
    m = np.ascontiguousarray(mask, dtype=bool)
    if W % F:
        m = np.pad(m, ((0, 0), (0, -W % F)))
    cols = m.view(np.uint64) != 0
    if H % F:
        cols = np.pad(cols, ((0, -H % F), (0, 0)))
    small = cols.reshape(-1, F, cols.shape[1]).any(1)
    k = -(-pad // F)
    grown = cv2.dilate(small.view(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    boxes = [[int(x) * F, int(y) * F, int(x + w) * F, int(y + h) * F] for x, y, w, h, _ in st[1:]]
    merged = True
    while merged:   # ponytail: O(n^2) per pass; n is a handful of overlays
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                p, q = boxes[i], boxes[j]
                if p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]:
                    boxes[i] = [min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [(max(0, x0), max(0, y0), min(W, x1), min(H, y1)) for x0, y0, x1, y1 in boxes]


def fill_frame(t, buf, prev_out, photo_thr=10.0):
    """Fill the masked pixels of buffer entry `t` from other entries of the same shot, one region
    per separate part of the mask (a corner logo and a URL at the far edge are warped and judged
    apart, not as one frame-wide box). Borrowing runs on the GPU when the window's frames were
    uploaded (e["g"]), else with OpenCV; pixels no frame can supply are inpainted on the CPU and
    kept stable over time."""
    img, mask = t["img"], t["mask"]
    stats = {"borrowed": 0, "carried": 0, "inpainted": 0, "secs_borrow": 0.0, "secs_inpaint": 0.0}
    borrow = _borrow_torch if t.get("g") is not None else _borrow_cv
    res = img.copy()
    for roi in _regions(mask, pad=24):
        x0, y0, x1, y1 = roi
        out = img[y0:y1, x0:x1].copy()
        remaining = mask[y0:y1, x0:x1].copy()
        ring = (cv2.dilate(remaining.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~remaining
        t0 = time.perf_counter()
        out, remaining = borrow(t, buf, roi, out, remaining, ring, photo_thr, stats)
        t1 = time.perf_counter()
        stats["secs_borrow"] += t1 - t0
        if remaining.any():
            rem8 = remaining.astype(np.uint8) * 255
            tel = cv2.inpaint(out, rem8, 4, cv2.INPAINT_TELEA)
            carried = False
            if prev_out is not None and prev_out["shot"] == t["shot"]:
                M = t["C"] @ np.linalg.inv(prev_out["C"])
                wi, wm = _warp_to(prev_out["img"], None, M, roi)
                chk = ring if wm is None else ring & ~wm
                ok = remaining if wm is None else remaining & ~wm
                if chk.sum() >= 30 and ok.all() and \
                        np.abs(wi[chk].astype(np.int16) - out[chk].astype(np.int16)).mean() <= photo_thr:
                    blend = cv2.addWeighted(wi, 0.75, tel, 0.25, 0)
                    out[remaining] = blend[remaining]
                    carried = True
            if not carried:
                out[remaining] = tel[remaining]
            stats["carried" if carried else "inpainted"] += int(remaining.sum())
        stats["secs_inpaint"] += time.perf_counter() - t1
        res[y0:y1, x0:x1] = out
    return res, stats


def _fill_device(device):
    """"auto": CUDA when torch sees it, else None (OpenCV on the CPU). ponytail: MPS not enabled
    until measured on a Mac."""
    if device in (None, "cpu"):
        return None
    try:
        import torch
    except ImportError:
        return None
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else None
    return device


def remove(video, tracks_json, out, window=40, crf=21, preset="medium", cap_bitrate=True, encoder="auto", device="auto",
           patches=None, force_fill=False, log=print):
    doc = tk.load(tracks_json)
    if force_fill:
        for t in doc["tracks"]:
            t["tier"] = "fill"
    info = doc["info"]
    W, H = info["width"], info["height"]
    cuts = set(doc["cuts"])
    tmp = out + ".video.mp4"
    maxrate = int(info["bit_rate"] * 1.15) if cap_bitrate and info["bit_rate"] else None
    wr = vio.VideoWriter(tmp, W, H, info["fps"], info["time_base"], encoder=encoder, crf=crf, preset=preset,
                         max_bitrate=maxrate)
    motion = Motion(W, H)
    dev = _fill_device(device) if any(t["enabled"] for t in doc["tracks"]) else None
    if dev is not None:
        import torch
    buf, pending = deque(), deque()
    prev_out = None
    tot = {"frames": 0, "alpha_frames": 0, "fill_frames": 0, "borrowed": 0, "carried": 0, "inpainted": 0,
           "propainter_frames": 0}
    secs = {"decode": 0.0, "alpha+motion": 0.0, "fill": 0.0, "fill: borrow": 0.0, "fill: inpaint": 0.0, "encode": 0.0}
    clock = time.perf_counter

    def emit(e):
        nonlocal prev_out
        t0 = clock()
        if e["mask"] is not None and e["mask"].any():
            res, st = fill_frame(e, buf, prev_out)
            for k, v in st.items():
                if k.startswith("secs_"):
                    secs["fill: " + k[5:]] += v
                else:
                    tot[k] += v
            tot["fill_frames"] += 1
        else:
            res = e["img"]
        prev_out = {"img": res, "C": e["C"], "shot": e["shot"]}
        t1 = clock()
        wr.write(res, e["pts"])
        secs["fill"] += t1 - t0
        secs["encode"] += clock() - t1

    def prepared():   # runs in a background thread (vio.prefetch): decode, alpha, motion, upload
        shot = 0
        frames = vio.read_frames(video)
        while True:
            t0 = clock()
            nxt = next(frames, None)
            secs["decode"] += clock() - t0
            if nxt is None:
                return
            i, pts, img = nxt
            t0 = clock()
            if i in cuts:
                shot += 1
            mask = frame_masks(doc, i, W, H)
            if patches is not None and mask is not None:
                if patches.paste(i, img, mask):
                    tot["propainter_frames"] += 1
                    mask = None
            m2 = apply_alpha(img, mask, doc, i)
            if any(t["enabled"] and t["tier"] == "alpha" and i < len(t["pos"]) and t["pos"][i, 0] > tk.ABSENT
                   for t in doc["tracks"]):
                tot["alpha_frames"] += 1
            mask = m2
            C = motion.step(img, mask, i in cuts)
            e = {"i": i, "pts": pts, "img": img, "mask": mask, "C": C, "shot": shot}
            if dev is not None:   # the window's frames live on the GPU for borrowing
                e["g"] = torch.from_numpy(img).to(dev)
                e["gm"] = None if mask is None else torch.from_numpy(mask).to(dev)
            secs["alpha+motion"] += clock() - t0
            yield e

    for e in vio.prefetch(prepared()):
        i = e["i"]
        buf.append(e)
        pending.append(e)
        tot["frames"] += 1
        while pending and pending[0]["i"] <= i - window:
            emit(pending.popleft())
            while buf and buf[0]["i"] < pending[0]["i"] - window if pending else False:
                buf.popleft()
    while pending:
        emit(pending.popleft())
        while buf and pending and buf[0]["i"] < pending[0]["i"] - window:
            buf.popleft()
    t0 = clock()
    wr.close()
    secs["encode"] += clock() - t0
    vio.finalize(tmp, video, out)
    tot["seconds"] = {k: round(v, 2) for k, v in secs.items()}
    tot["encoder"] = wr.codec
    filled = tot["borrowed"] + tot["carried"] + tot["inpainted"]
    tot["borrowed_share"] = round(tot["borrowed"] / filled, 3) if filled else None
    log(f"[remove] {tot['frames']} frames; alpha-inverted {tot['alpha_frames']}, filled {tot['fill_frames']}"
        + (f" (real pixels borrowed for {tot['borrowed_share']:.0%} of filled area)" if filled else "")
        + (f", ProPainter {tot['propainter_frames']}" if tot["propainter_frames"] else ""))
    log(f"[remove] {wr.codec}, fill on {dev or 'cpu'}; seconds (decode and alpha+motion run in a "
        "background thread, overlapping fill and encode): " + ", ".join(f"{k} {v:.1f}" for k, v in tot["seconds"].items()))
    return tot
