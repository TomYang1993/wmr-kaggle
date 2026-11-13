"""Optional higher-quality fill for opaque overlays using ProPainter (flow-guided video
inpainting). ProPainter is NOT bundled: it is cloned separately and is under the NTU S-Lab
License 1.0 (non-commercial use only). Only cropped regions around the overlays are processed,
and jobs are spread across the available GPUs."""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import cv2
import numpy as np

from . import backend
from . import track as tk
from . import vio
from .remove import frame_masks

REPO = "https://github.com/sczhou/ProPainter.git"


def default_dir():
    return os.environ.get("WMR_PROPAINTER", os.path.join(os.path.expanduser("~"), ".cache", "wmr", "ProPainter"))


def setup(dest=None, log=print):
    dest = dest or default_dir()
    if not os.path.exists(os.path.join(dest, "inference_propainter.py")):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        log(f"[propainter] cloning into {dest}")
        subprocess.run(["git", "clone", "--depth", "1", REPO, dest], check=True)
    log("[propainter] ready (model weights download on first run). Licence: NTU S-Lab 1.0, non-commercial.")
    return dest


def plan_jobs(doc, chunk=240, overlap=8, ctx=8, pad=None):
    """Split the frames that need filling into jobs: [start, end) + a fixed crop rectangle."""
    n, W, H = doc["frames"], doc["info"]["width"], doc["info"]["height"]
    fills = [t for t in doc["tracks"] if t["enabled"] and t["tier"] == "fill"]
    if not fills:
        return []
    need = np.zeros(n, bool)
    for t in fills:
        need |= t["pos"][:n, 0] > tk.ABSENT
    cuts = sorted(set(doc["cuts"]) | {0, n})
    jobs = []
    for a, b in zip(cuts[:-1], cuts[1:]):          # one shot at a time
        idx = np.flatnonzero(need[a:b]) + a
        if len(idx) == 0:
            continue
        runs, s, p = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - p > 2 * ctx:
                runs.append((s, p))
                s = i
            p = i
        runs.append((s, p))
        for s, e in runs:
            s, e = max(a, s - ctx), min(b, e + 1 + ctx)
            k = s
            while k < e:
                je = min(e, k + chunk)
                if e - je < overlap * 2:
                    je = e
                jobs.append({"start": int(k), "end": int(je)})
                if je >= e:
                    break
                k = je - overlap
    for j in jobs:
        x0 = y0 = 10 ** 9
        x1 = y1 = -10 ** 9
        hmax = 0
        for t in fills:
            p = t["pos"][j["start"]:j["end"]]
            p = p[p[:, 0] > tk.ABSENT]
            if len(p):
                x0, y0 = min(x0, int(p[:, 0].min())), min(y0, int(p[:, 1].min()))
                x1, y1 = max(x1, int(p[:, 0].max()) + t["w"]), max(y1, int(p[:, 1].max()) + t["h"])
                hmax = max(hmax, t["h"])
        P = pad if pad is not None else max(32, hmax)
        x0, y0, x1, y1 = max(0, x0 - P), max(0, y0 - P), min(W, x1 + P), min(H, y1 + P)
        for _ in range(2):                           # grow to at least 128 px and a multiple of 8
            w, h = x1 - x0, y1 - y0
            tw, th = max(128, (w + 7) // 8 * 8), max(128, (h + 7) // 8 * 8)
            x0 = max(0, min(x0 - (tw - w) // 2, W - tw))
            y0 = max(0, min(y0 - (th - h) // 2, H - th))
            x1, y1 = min(W, x0 + tw), min(H, y0 + th)
        x1 -= (x1 - x0) % 8
        y1 -= (y1 - y0) % 8
        j["roi"] = [int(x0), int(y0), int(x1), int(y1)]
    return jobs


def build(video, tracks_json, workdir, pp_dir=None, gpus=None, chunk=240, overlap=8, fp16=True,
          subvideo=80, force_fill=False, log=print):
    """Dump crops, run ProPainter on each job, and return a Patches store (or None)."""
    pp_dir = pp_dir or default_dir()
    if not os.path.exists(os.path.join(pp_dir, "inference_propainter.py")):
        raise FileNotFoundError(f"ProPainter not found at {pp_dir}; run `wmr setup-propainter` first")
    doc = tk.load(tracks_json)
    if force_fill:
        for t in doc["tracks"]:
            t["tier"] = "fill"
    W, H = doc["info"]["width"], doc["info"]["height"]
    jobs = plan_jobs(doc, chunk=chunk, overlap=overlap)
    if not jobs:
        return None
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    for k, j in enumerate(jobs):
        j["dir"] = os.path.join(workdir, f"job{k:04d}")
        os.makedirs(os.path.join(j["dir"], "in"))
        os.makedirs(os.path.join(j["dir"], "mask"))
    last = max(j["end"] for j in jobs)
    for i, _, img in vio.read_frames(video):
        if i >= last:
            break
        act = [j for j in jobs if j["start"] <= i < j["end"]]
        if not act:
            continue
        m = frame_masks(doc, i, W, H)
        for j in act:
            x0, y0, x1, y1 = j["roi"]
            name = f"{i - j['start']:05d}.png"
            cv2.imwrite(os.path.join(j["dir"], "in", name), img[y0:y1, x0:x1])
            mm = np.zeros((y1 - y0, x1 - x0), np.uint8) if m is None else m[y0:y1, x0:x1].astype(np.uint8) * 255
            cv2.imwrite(os.path.join(j["dir"], "mask", name), mm)
    gpus = gpus if gpus is not None else backend.gpu_ids()
    slots = Queue()
    for g in (gpus or [None]):
        slots.put(g)
    use_fp16 = fp16 and bool(gpus)

    def run(j):
        g = slots.get()
        try:
            env = dict(os.environ)
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            cmd = [sys.executable, "inference_propainter.py", "-i", os.path.join(j["dir"], "in"),
                   "-m", os.path.join(j["dir"], "mask"), "-o", os.path.join(j["dir"], "out"),
                   "--save_frames", "--mask_dilation", "0", "--subvideo_length", str(subvideo)]
            if use_fp16:
                cmd.append("--fp16")
            r = subprocess.run(cmd, cwd=pp_dir, env=env, capture_output=True, text=True)
            res = os.path.join(j["dir"], "out", "in", "frames")
            n_out = len(os.listdir(res)) if os.path.isdir(res) else 0
            j["ok"] = r.returncode == 0 and n_out == j["end"] - j["start"]
            j["frames_dir"] = res
            if not j["ok"]:
                with open(os.path.join(j["dir"], "error.log"), "w") as fh:
                    fh.write(r.stdout[-4000:] + "\n" + r.stderr[-8000:])
                log(f"[propainter] job {j['start']}-{j['end']} failed (see {j['dir']}/error.log); "
                    "those frames fall back to the built-in fill")
            else:
                log(f"[propainter] job {j['start']}-{j['end']} roi {j['roi']} done" + (f" on GPU {g}" if g is not None else ""))
        finally:
            slots.put(g)

    with ThreadPoolExecutor(max_workers=max(1, len(gpus or [1]))) as ex:
        list(ex.map(run, jobs))
    with open(os.path.join(workdir, "index.json"), "w") as fh:
        json.dump(jobs, fh)
    return Patches(jobs, overlap)


class Patches:
    def __init__(self, jobs, overlap=8):
        self.jobs = [j for j in jobs if j.get("ok")]
        self.overlap = overlap

    def paste(self, i, img, mask):
        """Write ProPainter pixels into img where mask is set; clears those mask pixels.
        Returns True when nothing is left to fill."""
        act = [j for j in self.jobs if j["start"] <= i < j["end"]]
        if not act:
            return False
        H, W = mask.shape
        acc = np.zeros((H, W, 3), np.float32)
        ws = np.zeros((H, W), np.float32)
        for j in act:
            patch = cv2.imread(os.path.join(j["frames_dir"], f"{i - j['start']:04d}.png"))
            if patch is None:
                continue
            x0, y0, x1, y1 = j["roi"]
            if patch.shape[:2] != (y1 - y0, x1 - x0):
                patch = cv2.resize(patch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
            w = min(1.0, (i - j["start"] + 1) / float(self.overlap), (j["end"] - i) / float(self.overlap)) \
                if len(act) > 1 else 1.0
            sub = mask[y0:y1, x0:x1]
            acc[y0:y1, x0:x1][sub] += w * patch[sub]
            ws[y0:y1, x0:x1][sub] += w
        done = mask & (ws > 0)
        if done.any():
            img[done] = np.clip(acc[done] / ws[done][:, None] + 0.5, 0, 255).astype(np.uint8)
            mask[done] = False
        return not mask.any()
