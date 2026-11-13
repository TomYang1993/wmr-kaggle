"""Where an OCR call spends its time, stage by stage, on sampled frames of a video.
    python scripts/ocr_profile.py VIDEO [frames]
Runs one engine alone, then the pool (one engine per GPU) as analyze and verify use it, with one
and with two worker threads per engine."""
import collections
import os
import sys
import threading
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wmr import vio  # noqa: E402
from wmr.detect import OCRPool, _DetPre  # noqa: E402

import rapidocr.main as rmain  # noqa: E402
from rapidocr.ch_ppocr_det.utils import DBPostProcess  # noqa: E402
from rapidocr.ch_ppocr_rec.main import TextRecognizer  # noqa: E402
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode  # noqa: E402
from rapidocr.inference_engine.onnxruntime.main import OrtInferSession  # noqa: E402

T, C, LOCK = collections.defaultdict(float), collections.Counter(), threading.Lock()


def timed(cls, name, key):
    f = getattr(cls, name)

    def w(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            dt = time.perf_counter() - t
            kk = key(a) if callable(key) else key
            with LOCK:
                T[kk] += dt
                C[kk] += 1
    setattr(cls, name, w)


timed(rmain.RapidOCR, "__call__", "call total")
timed(rmain.LoadImage, "__call__", "load_img")
timed(_DetPre, "__call__", "det pre")
timed(OrtInferSession, "__call__", lambda a: "rec infer" if a[1].shape[2] == 48 else "det infer")
timed(DBPostProcess, "__call__", "det post")
timed(rmain.RapidOCR, "crop_text_regions", "crop")
timed(TextRecognizer, "resize_norm_img", "rec pre (per crop)")
timed(CTCLabelDecode, "__call__", "rec post")
timed(rmain.RapidOCR, "build_final_output", "final output")


def report(tag, n_frames, wall, cpu):
    calls = C["call total"] or 1
    print(f"\n== {tag}: {n_frames} frames, {calls} calls, {wall:.2f} s wall, {1000 * wall / n_frames:.0f} ms/frame,"
          f" process CPU {cpu:.1f} s ({100 * cpu / wall:.0f}% of one core)")
    for k in sorted(T, key=lambda k: -T[k]):
        print(f"   {k:20s} {1000 * T[k] / calls:7.1f} ms/call  ({C[k]} runs)")
    T.clear()
    C.clear()


def main():
    video = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    info = vio.probe(video)
    step = max(1, info["frames"] // n)
    frames = [img for i, _, img in vio.read_frames(video) if i % step == 0][:n]
    print(f"{video}: {info['width']}x{info['height']}, {len(frames)} frames; os.cpu_count {os.cpu_count()},"
          f" cv2 threads {cv2.getNumThreads()}")
    pool = OCRPool()
    print(f"OCR {pool.model} on {pool.provider} x{len(pool.engines)}; init {pool.init_secs}")
    one = pool.engines[0]
    for f in frames[:2]:
        one(f)
    for _ in pool.imap(enumerate(frames)):   # warm every engine (rec bucket sessions are built on first use)
        pass
    T.clear()
    C.clear()
    w, c = time.perf_counter(), time.process_time()
    for f in frames:
        one(f)
    report("one engine", len(frames), time.perf_counter() - w, time.process_time() - c)
    for per in (1, 2):
        for enh in (False, True):
            pool.enhance = enh
            w, c = time.perf_counter(), time.process_time()
            for _ in pool.imap(enumerate(frames), per_engine=per):
                pass
            report(f"pool x{len(pool.engines)}, {per} thread(s) per engine" + (", enhance (as verify)" if enh else ""),
                   len(frames), time.perf_counter() - w, time.process_time() - c)
    pool.enhance = False


if __name__ == "__main__":
    main()
