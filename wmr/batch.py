"""Run many videos in parallel worker processes, one per GPU (or per CPU worker)."""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from . import backend

EXTS = (".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi")


def batch(in_dir, out_dir, extra_args, gpus=None, workers=None, log=print):
    vids = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(EXTS))
    os.makedirs(out_dir, exist_ok=True)
    gpus = gpus if gpus is not None else backend.gpu_ids()
    slots = Queue()
    pool = list(gpus) if gpus else [None] * (workers or 1)
    if workers and gpus:
        pool = [gpus[k % len(gpus)] for k in range(workers)]
    for g in pool:
        slots.put(g)
    results = {}

    def run(name):
        g = slots.get()
        try:
            env = dict(os.environ)
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            stem = os.path.splitext(name)[0]
            out = os.path.join(out_dir, stem + "_clean.mp4")
            cmd = [sys.executable, "-m", "wmr", "run", os.path.join(in_dir, name), "-o", out] + list(extra_args)
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            with open(os.path.join(out_dir, stem + ".log"), "w") as fh:
                fh.write(r.stdout + "\n" + r.stderr)
            results[name] = r.returncode
            log(f"[batch] {name}: {'ok' if r.returncode == 0 else 'FAILED (see log)'}"
                + (f" [GPU {g}]" if g is not None else ""))
        finally:
            slots.put(g)

    with ThreadPoolExecutor(max_workers=len(pool)) as ex:
        list(ex.map(run, vids))
    return results
