"""Synthetic ground truth: render known text overlays onto clean footage (or a procedural
clip) with a chosen motion and opacity, and record exact per-frame positions."""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import vio


def _scene(rng, W, H, margin):
    bh, bw = H + 2 * margin, W + 2 * margin
    base = rng.random((bh // 24 + 2, bw // 24 + 2, 3)).astype(np.float32)
    img = cv2.resize(base, (bw, bh), interpolation=cv2.INTER_CUBIC) * 255
    for _ in range(40):
        col = tuple(int(c) for c in rng.integers(20, 235, 3))
        x, y = int(rng.integers(0, bw)), int(rng.integers(0, bh))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(20, 160)), y + int(rng.integers(20, 120))), col, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(10, 70)), col, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def procedural_frames(n, W, H, scene_len, static_scene, seed=0, blobs=1, blob_r=26, exposure=0.0):
    """Panning scenes; `blobs` discs of radius `blob_r` move on their own (local motion, which a
    global camera model cannot align); `exposure`: brightness swings by +-that much (period 50)."""
    rng = np.random.default_rng(seed)
    margin = 0 if static_scene else int(min(400, 3 * scene_len + 16))
    i = 0
    while i < n:
        big = _scene(rng, W, H, margin)
        v = np.zeros(2) if static_scene else rng.uniform(-2.5, 2.5, 2)
        bl = [(rng.uniform([0, 0], [W, H]), rng.uniform(-4, 4, 2), tuple(int(c) for c in rng.integers(0, 255, 3)))
              for _ in range(blobs)]
        for k in range(min(scene_len, n - i)):
            ox = int(np.clip(margin + v[0] * k, 0, 2 * margin))
            oy = int(np.clip(margin + v[1] * k, 0, 2 * margin))
            f = big[oy:oy + H, ox:ox + W].copy()
            if not static_scene:
                for p, bv, col in bl:
                    c = p + bv * k
                    cv2.circle(f, (int(c[0]) % W, int(c[1]) % H), blob_r, col, -1)
            noise = rng.normal(0, 2.0, f.shape) + exposure * np.sin(2 * np.pi * (i + k) / 50)
            yield np.clip(f + noise, 0, 255).astype(np.uint8)
        i += scene_len


def render_text(text, size, color):
    """Return (coverage float32 HxW in 0..1, color BGR)."""
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    x0, y0, x1, y1 = font.getbbox(text)
    im = Image.new("L", (x1 - x0 + 4, y1 - y0 + 4), 0)
    ImageDraw.Draw(im).text((2 - x0, 2 - y0), text, fill=255, font=font)
    return np.asarray(im, dtype=np.float32) / 255.0, color


def simulate(motion, n, W, H, w, h, rng, dwell=60, speed=3.0):
    pos = np.zeros((n, 2), dtype=np.int32)
    xmax, ymax = max(1, W - w - 1), max(1, H - h - 1)
    if motion == "static":
        pos[:] = [rng.integers(0, xmax), rng.integers(0, ymax)]
    elif motion == "teleport":
        for s in range(0, n, dwell):
            pos[s:s + dwell] = [rng.integers(0, xmax), rng.integers(0, ymax)]
    elif motion == "linear":
        y = rng.integers(0, ymax)
        for t in range(n):
            pos[t] = [int(speed * t) % (W + w) - w, y]
    elif motion == "bounce":
        p = np.array([rng.integers(0, xmax), rng.integers(0, ymax)], dtype=np.float64)
        v = np.array([speed, speed * 0.7]) * rng.choice([-1, 1], 2)
        for t in range(n):
            pos[t] = p.round()
            p += v
            for a, lim in ((0, xmax), (1, ymax)):
                if p[a] < 0 or p[a] > lim:
                    v[a] = -v[a]
                    p[a] = np.clip(p[a], 0, lim)
    else:
        raise ValueError(f"unknown motion {motion}")
    return pos


def parse_overlay(spec):
    parts = spec.split(":")
    text = parts[0]
    motion = parts[1] if len(parts) > 1 else "static"
    alpha = float(parts[2]) if len(parts) > 2 else 1.0
    size = int(parts[3]) if len(parts) > 3 else 0
    return {"text": text, "motion": motion, "alpha": alpha, "size": size,
            "shadow": len(parts) > 4 and parts[4] == "shadow"}


def synth(out_dir, overlays, clean=None, frames=240, size=(640, 360), fps=30, scene_len=80,
          static_scene=False, seed=0, dwell=60, speed=3.0, crf=18, blobs=1, blob_r=26, exposure=0.0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    if clean:
        info = vio.probe(clean)
        W, H, fps = info["width"], info["height"], info["fps"]
        src = (f for _, _, f in vio.read_frames(clean))
        n = min(frames, info["frames"]) if frames else info["frames"]
    else:
        W, H = size
        n = frames
        src = procedural_frames(n, W, H, scene_len, static_scene, seed, blobs, blob_r, exposure)
    specs = []
    palette = [(255, 255, 255), (40, 230, 250), (250, 250, 60), (30, 30, 30)]
    for k, o in enumerate(overlays):
        o = parse_overlay(o) if isinstance(o, str) else dict(o)
        fsize = o.get("size") or max(18, H // 14)
        cov, col = render_text(o["text"], fsize, palette[k % len(palette)])
        layers = [(cov, col, o["alpha"])]
        if o.get("shadow"):   # soft dark drop shadow under the text: a two-tone mark
            sh = max(2, fsize // 18)
            ct = np.pad(cov, ((0, sh), (0, sh)))
            cs = cv2.GaussianBlur(np.pad(cov, ((sh, 0), (sh, 0))), (0, 0), sh / 2)
            layers = [(cs, (16, 16, 16), o["alpha"]), (ct, col, o["alpha"])]
            cov = np.maximum(ct, cs)
        h, w = cov.shape
        pos = simulate(o["motion"], n, W, H, w, h, rng, dwell=dwell, speed=speed)
        cv2.imwrite(os.path.join(out_dir, f"gt_cov{k}.png"), (cov * 255).astype(np.uint8))
        specs.append({"text": o["text"], "motion": o["motion"], "alpha": o["alpha"], "color": col,
                      "w": w, "h": h, "pos": pos, "layers": layers, "cov_file": f"gt_cov{k}.png",
                      "shadow": bool(o.get("shadow"))})
    tb = f"1/{int(round(fps * 1000))}"
    step = 1000
    wm = vio.VideoWriter(os.path.join(out_dir, "marked.mp4"), W, H, fps, tb, encoder="x264", crf=crf)
    wc = vio.VideoWriter(os.path.join(out_dir, "clean.mp4"), W, H, fps, tb, encoder="x264", crf=crf)
    count = 0
    for t, frame in enumerate(src):
        if t >= n:
            break
        wc.write(frame, t * step)
        f = frame.astype(np.float32)
        for s in specs:
            x, y = s["pos"][t]
            x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + s["w"]), min(H, y + s["h"])
            if x1 <= x0 or y1 <= y0:
                continue
            for lc, lcol, la in s["layers"]:
                a = (lc[y0 - y:y1 - y, x0 - x:x1 - x] * la)[..., None]
                f[y0:y1, x0:x1] = a * np.array(lcol, np.float32) + (1 - a) * f[y0:y1, x0:x1]
        wm.write(np.clip(f + 0.5, 0, 255).astype(np.uint8), t * step)
        count += 1
    wm.close()
    wc.close()
    gt = {"frames": count, "width": W, "height": H, "fps": fps,
          "overlays": [{"text": s["text"], "motion": s["motion"], "alpha": s["alpha"],
                        "color": list(s["color"]), "w": s["w"], "h": s["h"],
                        "cov_file": s["cov_file"], "shadow": s["shadow"], "pos": s["pos"][:count].tolist()}
                       for s in specs]}
    with open(os.path.join(out_dir, "gt.json"), "w") as fh:
        json.dump(gt, fh)
    with open(os.path.join(out_dir, "texts.txt"), "w") as fh:
        fh.write("\n".join(s["text"] for s in specs) + "\n")
    return gt
