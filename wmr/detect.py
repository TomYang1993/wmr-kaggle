"""Text detection + recognition behind one interface (RapidOCR 3.x / PP-OCRv6 on ONNX Runtime)."""
import os
import sys
import threading
import time
from difflib import SequenceMatcher

import cv2
import numpy as np

from . import backend


def norm_text(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def match_text(text, expected, thr=0.7):
    """Best fuzzy match of an OCR string against the expected overlay strings."""
    a = norm_text(text)
    best, score = None, 0.0
    if not a:
        return best, score
    for e in expected:
        b = norm_text(e)
        if not b:
            continue
        r = SequenceMatcher(None, a, b).ratio()
        if len(a) >= 4 and a in b and len(a) >= 0.5 * len(b):
            r = max(r, 0.9)      # truncated read
        if len(b) >= 4 and b in a:
            r = max(r, 0.9)      # read merged with neighbouring text
        if r > score:
            best, score = e, r
    return (best, score) if score >= thr else (None, score)


def _iou(a, b):
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / float(a["w"] * a["h"] + b["w"] * b["h"] - inter + 1e-6)


def _warmup_image():
    """Wide frame with text at two sizes, so detection and a multi-crop recognition batch both run."""
    img = np.full((720, 1280, 3), 90, np.uint8)
    cv2.putText(img, "warmup 2026 www.example.com", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    cv2.putText(img, "@sample_mark", (700, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    return img


def _sessions(obj, depth=0, seen=None):
    """(owner, key, session) for every onnxruntime session reachable from obj; owner[key] or
    getattr(owner, key) is where it lives. Finds them whatever the RapidOCR release calls them."""
    import onnxruntime as ort
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 6:
        return []
    seen.add(id(obj))
    if isinstance(obj, dict):
        items, is_map = list(obj.items()), True
    elif isinstance(obj, list):
        items, is_map = list(enumerate(obj)), True
    else:
        items, is_map = list(getattr(obj, "__dict__", {}).items()), False
    out = []
    for k, v in items:
        if isinstance(v, ort.InferenceSession):
            out.append((obj, k, v, is_map))
        elif v is not None and not isinstance(v, (str, bytes, int, float, bool, np.ndarray, tuple)):
            out += _sessions(v, depth + 1, seen)
    return out


def _put(owner, key, value, is_map):
    if is_map:
        owner[key] = value
    else:
        setattr(owner, key, value)


class _DetPre:
    """RapidOCR's det preprocessing (resize to a multiple of 32, (x/255 - 0.5)/0.5, NCHW) with one
    OpenCV pass for the normalise step instead of numpy float64 temporaries: same values, ~8x less
    CPU on a 1080p frame (38 -> 5 ms on an M-series Mac)."""

    def __init__(self, op):
        self.op = op

    def __call__(self, img):
        r = self.op.resize(img)
        return None if r is None else cv2.dnn.blobFromImage(r, 1 / 127.5, mean=(127.5, 127.5, 127.5))


def _fast_det_pre(engine):
    det = getattr(engine, "text_det", None)
    op = det.get_preprocess() if det is not None and hasattr(det, "get_preprocess") else None
    if op is not None and np.allclose(op.mean, 0.5) and np.allclose(op.std, 0.5) and op.scale == 1 / 255.0:
        fast = _DetPre(op)
        det.get_preprocess = lambda: fast   # RapidOCR rebuilds it per call from fixed settings


_FP16_LOCK = threading.Lock()


def _fp16(path):
    """An OCR model in fp16 for CUDA (T4 tensor cores, 1080p, 2026-11-11: det 163 -> 89 ms, rec at a
    fixed shape 19.5 -> 11 ms, same reads): the path of a converted copy next to the original
    (bytes if that folder is read-only), or None without the `onnx` and `onnxconverter-common`
    packages. ReduceMean stays fp32: the det model's squeeze-excite pools sum a whole 1080p feature
    map, which overflows fp16 to NaN."""
    out = path[:-5] + ".fp16.onnx"   # ponytail: delete these files after changing the conversion below
    with _FP16_LOCK:   # engines share one conversion (~2 s of CPU per model, once per machine)
        if os.path.exists(out):
            return out
        try:
            import warnings
            import onnx
            from onnxconverter_common import float16
        except ImportError:
            return None
        with warnings.catch_warnings():   # tiny weights truncated to fp16's range: expected
            warnings.simplefilter("ignore")
            data = float16.convert_float_to_float16(
                onnx.load(path), keep_io_types=True,
                op_block_list=float16.DEFAULT_OP_BLOCK_LIST + ["ReduceMean"]).SerializeToString()
        try:
            with open(out + ".tmp", "wb") as fh:
                fh.write(data)
            os.replace(out + ".tmp", out)
            return out
        except OSError:
            return data


class _ShapeBuckets:
    """Stands in for the rec InferenceSession on CUDA. ORT's cuDNN Conv rebuilds its state whenever
    a run's input shape differs from the session's previous run (~75 ms for the rec model on T4,
    2026-11-11), and RapidOCR's crop batches change width and count almost every call (100 ms per
    batch vs 20 at a fixed shape). So the batch is zero-padded to RapidOCR's batch size and the
    width up to a bucket (as RapidOCR already pads narrower crops to the widest in a batch), and
    each bucket has its own session, which then always sees one shape."""
    BUCKETS = (320, 480, 640, 800, 960, 1120, 1280, 1600, 1920, 2560)   # ponytail: wider runs as is

    def __init__(self, make, batch):
        self.make, self.batch, self.by_w, self.lock = make, batch, {}, threading.Lock()
        self.first = make()

    def __getattr__(self, k):   # metadata (inputs, outputs, providers) from the first session
        return getattr(self.first, k)

    def run(self, names, feed):
        (k, x), = feed.items()
        n, c, h, w = x.shape
        W = next((b for b in self.BUCKETS if b >= w), w)
        xp = np.zeros((max(n, self.batch), c, h, W), x.dtype)
        xp[:n, :, :, :w] = x
        with self.lock:
            s = self.by_w.get(W) or self.by_w.setdefault(W, self.first if not self.by_w else self.make())
        return [y[:n] for y in s.run(names, {k: xp})]


_CPU_PHASE = threading.Lock()   # model download + CPU session build: one engine at a time


class OCR:
    """PP-OCRv6 (RapidOCR 3.x) on ONNX Runtime. On CUDA the medium models (best reads of small,
    faint text on T4: 2026-11-10 probe), elsewhere the small ones (CPU cost)."""

    # CUDA EP option sets, tried in order; the first that passes the warm-up wins. On T4 (sm_75)
    # cuDNN 9.19 (pinned by Kaggle's torch 2.11+cu130) has no NCHW engine for the det convs at
    # inputs >= ~736 px; NHWC works there. setup_kaggle.sh upgrades to cuDNN 9.28, where plain
    # NCHW works and is ~1.7x faster than NHWC (probed 2026-11-10). fp32 det at 1088x1920 takes
    # 160 ms of T4 compute (2026-11-11 profile), hence fp16 (_fp16).
    CUDA_OPTS = [{}, {"prefer_nhwc": "1"}]

    def __init__(self, use_gpu=True, enhance=False, device_id=0, fp16=None):
        from rapidocr import ModelType, OCRVersion, RapidOCR
        self.enhance = enhance
        self.device_id = device_id
        # det and rec models in fp16 on CUDA; WMR_OCR_FP16=0 keeps fp32
        self.fp16 = os.environ.get("WMR_OCR_FP16", "1") != "0" if fp16 is None else fp16
        self.fp16_used = False
        self.provider = "CPUExecutionProvider"
        self.provider_opts = {}
        self._cpu_sessions = []
        prov = backend.ort_providers() if use_gpu else []
        want = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")
                if p in prov][:1]
        size = ModelType.MEDIUM if want == ["CUDAExecutionProvider"] else ModelType.SMALL
        self.model = f"PP-OCRv6 {size.name.lower()}"
        t0 = time.perf_counter()
        with _CPU_PHASE:
            self.engine = RapidOCR(params={
                "Global.log_level": "error", "Global.use_cls": False,
                "Det.ocr_version": OCRVersion.PPOCRV6, "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Det.model_type": size, "Rec.model_type": size,
                # WMR_THREADS: CPU threads per process, for several wmr processes on one machine
                "EngineConfig.onnxruntime.intra_op_num_threads": int(os.environ.get("WMR_THREADS", -1))})
            # RapidOCR downloads models and builds sessions lazily, on the first call; its loaders do
            # that without an inference (a medium-model CPU run took ~16 s on Kaggle's CPU)
            for load in ("_load_det_model", "_load_rec_model"):
                getattr(self.engine, load, lambda: None)()
            if not _sessions(self.engine):   # a RapidOCR without those loaders
                self._run_once(_warmup_image())
            _fast_det_pre(self.engine)
        t1 = time.perf_counter()
        self.init_secs = {"load": t1 - t0}
        if not want:
            return
        err = None
        for opts in (self.CUDA_OPTS if want[0] == "CUDAExecutionProvider" else [{}]):
            try:
                if not self._move_sessions(want[0], opts):
                    raise RuntimeError("a session did not build on the GPU")
                self._run_once(_warmup_image())   # a session can build on GPU and still fail on first run
                self.provider, self.provider_opts = want[0], opts
                if self.fp16_used:
                    self.model += " fp16"
                break
            except Exception as e:
                err = e
                self._to_cpu(None)
        else:
            self.provider = want[0]
            self._to_cpu(err)
        self.init_secs["gpu build+warm-up"] = time.perf_counter() - t1

    def _move_sessions(self, provider, cuda_opts=None):
        """Rebuild the engine's ONNX sessions on a GPU provider. Returns True only if every
        session actually runs on that provider."""
        import onnxruntime as ort
        self._cpu_sessions = moved = []   # (owner, key, CPU session, is_map), so _to_cpu can undo
        found = _sessions(self.engine)
        batch = getattr(getattr(self.engine, "text_rec", None), "rec_batch_num", 6)
        for owner, key, old, is_map in found:
            src = getattr(old, "_model_path", None) or getattr(old, "_model_bytes", None)
            if src is None:
                return False
            cuda = provider == "CUDAExecutionProvider"
            if cuda and self.fp16 and isinstance(src, str):
                s16 = _fp16(src)
                src, self.fp16_used = (s16, True) if s16 else (src, False)
            p = (provider, {**(cuda_opts or {}), "device_id": str(self.device_id)}) if cuda else provider

            def build(src=src, p=p):
                opts = ort.SessionOptions()
                opts.log_severity_level = 4
                s = ort.InferenceSession(src, sess_options=opts, providers=[p, "CPUExecutionProvider"])
                # without this ORT quietly rebuilds the session on CPU after a GPU kernel error
                s.disable_fallback()
                return s
            rec = old.get_inputs()[0].shape[2:3] == [48]   # rec input is 48 px high; det takes any size
            new = _ShapeBuckets(build, batch) if cuda and rec else build()
            if new.get_providers()[0] != provider:
                return False
            moved.append((owner, key, old, is_map))
            _put(owner, key, new, is_map)
        return bool(moved)

    def _to_cpu(self, err):
        for owner, key, old, is_map in self._cpu_sessions:
            _put(owner, key, old, is_map)
        self._cpu_sessions = []
        if err is not None and self.provider != "CPUExecutionProvider":
            msg = str(err).strip().splitlines()[0][:200] if str(err).strip() else ""
            print(f"note: OCR moved to CPU after a {self.provider} error ({type(err).__name__}: {msg})",
                  file=sys.stderr)
        self.provider = "CPUExecutionProvider"

    def _run(self, img):
        try:
            return self._run_once(img)
        except Exception as e:
            if self.provider == "CPUExecutionProvider":
                raise
            self._to_cpu(e)
            return self._run_once(img)

    def _run_once(self, img):
        # flags on every call: RapidOCR keeps per-call flags as engine state
        r = self.engine(img, use_det=True, use_cls=False, use_rec=True)
        if r.boxes is None or r.txts is None:
            return []
        dets = []
        for quad, text, score in zip(r.boxes, r.txts, r.scores):
            q = np.asarray(quad, dtype=np.float32)
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            dets.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0),
                         "text": str(text), "score": float(score)})
        return dets

    def __call__(self, bgr):
        dets = self._run(bgr)
        if self.enhance:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            # a CLAHE object per call: OCRPool may run one engine from two threads
            lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
            for d in self._run(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)):
                if all(_iou(d, e) < 0.5 for e in dets):
                    dets.append(d)
        return dets


class OCRPool:
    """One OCR engine per visible NVIDIA GPU, each fed by its own thread (ONNX Runtime releases
    the GIL), so sampled frames are read on all GPUs at once. Elsewhere a single engine."""

    def __init__(self, use_gpu=True, enhance=False):
        n = len(backend.gpu_ids()) if use_gpu and "CUDAExecutionProvider" in backend.ort_providers() else 0
        if n > 1:   # GPU warm-ups (CUDA context, cuDNN algorithm search) run side by side
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(n) as ex:
                self.engines = list(ex.map(lambda k: OCR(use_gpu=True, enhance=enhance, device_id=k), range(n)))
            if self.engines[0].provider != "CUDAExecutionProvider":
                self.engines = self.engines[:1]
        else:
            self.engines = [OCR(use_gpu=use_gpu, enhance=enhance)]
        first = self.engines[0]
        self.model, self.provider = first.model, first.provider
        self.init_secs = {k: round(max(e.init_secs.get(k, 0.0) for e in self.engines), 1) for k in first.init_secs}

    @property
    def enhance(self):
        return self.engines[0].enhance

    @enhance.setter
    def enhance(self, v):
        for e in self.engines:
            e.enhance = v

    def __call__(self, bgr):
        return self.engines[0](bgr)

    def imap(self, items, per_engine=None):
        """items: iterable of (key, bgr). Yields (key, bgr, detections) in input order, with at
        most two frames per worker in flight. On a GPU each engine runs from two threads, so one
        frame's CPU pre/post-processing overlaps another's GPU run (ONNX Runtime sessions take
        concurrent runs): T4 x2, 1080p, 82 -> 65 ms per frame (2026-11-11)."""
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        n = len(self.engines)
        per_engine = per_engine or (1 if self.provider == "CPUExecutionProvider" else 2)
        if n == 1 and per_engine == 1:
            for k, img in items:
                yield k, img, self.engines[0](img)
            return
        exs = [ThreadPoolExecutor(per_engine) for _ in self.engines]
        q = deque()
        try:
            for j, (k, img) in enumerate(items):
                q.append((k, img, exs[j % n].submit(self.engines[j % n], img)))
                if len(q) >= 2 * n * per_engine:
                    k0, i0, f = q.popleft()
                    yield k0, i0, f.result()
            while q:
                k0, i0, f = q.popleft()
                yield k0, i0, f.result()
        finally:
            for e in exs:
                e.shutdown(wait=True, cancel_futures=True)
