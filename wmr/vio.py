"""Video I/O on PyAV: frame-accurate decode with original timestamps, H.264 encode (NVENC on
NVIDIA, else libx264), audio stream copy. Output is 8-bit yuv420p.
Decode stays in software: on Kaggle T4, 1080p h264 decodes at 231 fps in software vs 133 fps with
PyAV's CUDA hwaccel (GPU->CPU frame copies). Encode is the other way round: libx264 medium 21 fps,
h264_nvenc p7 134 fps at 1080p (probed 2026-11-11)."""
import heapq
import os
import sys
from fractions import Fraction

import av
import cv2
import numpy as np


def probe(path):
    with av.open(path) as c:
        v = c.streams.video[0]
        fps = float(v.average_rate or v.guessed_rate or 30)
        dur = float(c.duration / av.time_base) if c.duration else 0.0
        frames = int(v.frames or round(dur * fps))
        bit_rate = int(v.bit_rate or 0)
        if not bit_rate and c.bit_rate:
            bit_rate = int(c.bit_rate)
        if not bit_rate and dur > 0:
            bit_rate = int(os.path.getsize(path) * 8 / dur)
        pix = v.codec_context.pix_fmt or ""
        return {
            "width": v.codec_context.width, "height": v.codec_context.height, "fps": fps,
            "frames": frames, "duration": dur, "codec": v.codec_context.name,
            "bit_rate": bit_rate, "pix_fmt": pix,
            "time_base": str(v.time_base) if v.time_base else "1/90000",
            "has_audio": len(c.streams.audio) > 0,
            "high_bit_depth": any(t in pix for t in ("10", "12", "16")),
        }


def luma(frame):
    """Gray uint8 (H, W) of a decoded frame, close to cv2's BGR -> gray. 8-bit planar YUV: a view
    of the Y plane, stretched from limited range (PyAV's to_ndarray("gray") is slower than
    converting to BGR); other formats: to_ndarray("gray")."""
    if frame.format.name not in ("yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "nv12"):
        return frame.to_ndarray(format="gray")
    p = frame.planes[0]
    y = np.frombuffer(p, np.uint8).reshape(-1, p.line_size)[:frame.height, :frame.width]
    if frame.format.name.startswith("yuvj"):
        return y
    return cv2.convertScaleAbs(y, alpha=255 / 219, beta=-16 * 255 / 219)


def read_frames(path, convert=True):
    """Yield (index, pts, bgr uint8). pts is in the stream time base. convert=False yields the
    decoded av.VideoFrame instead, for callers that convert only some frames (the YUV -> BGR
    conversion costs about as much as decoding)."""
    with av.open(path) as c:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        warned = False
        for i, frame in enumerate(c.decode(v)):
            if not warned and getattr(frame, "rotation", 0):
                print("warning: video carries rotation metadata; frames are processed as stored. "
                      "Normalise first, e.g. ffmpeg -i in.mp4 -c:a copy out.mp4", file=sys.stderr)
                warned = True
            yield i, (frame.pts if frame.pts is not None else i), frame.to_ndarray(format="bgr24") if convert else frame


def prefetch(gen, depth=8):
    """Iterate `gen` in a background thread, up to `depth` items ahead. PyAV, OpenCV and torch
    release the GIL, so CPU prep of the next frames overlaps the consumer's GPU work."""
    import queue
    import threading
    q = queue.Queue(depth)

    def work():
        try:
            for x in gen:
                q.put(("item", x))
            q.put(("end", None))
        except BaseException as e:   # re-raised in the consumer
            q.put(("err", e))

    threading.Thread(target=work, daemon=True).start()
    while True:
        kind, x = q.get()
        if kind == "end":
            return
        if kind == "err":
            raise x
        yield x


def nvenc_available():
    from . import backend
    return "h264_nvenc" in av.codecs_available and bool(backend.gpu_ids())


class VideoWriter:
    """encoder: "auto" (NVENC when an NVIDIA GPU is visible, else libx264), "nvenc" or "x264".
    crf maps to NVENC constant quality as cq = crf + 2 (NVENC needs ~2 more for a similar size)."""

    def __init__(self, path, width, height, fps, time_base, encoder="auto", crf=21,
                 preset="medium", max_bitrate=None):
        self.tb = Fraction(time_base)
        rate = Fraction(fps).limit_denominator(100000)
        nv = encoder == "nvenc" or (encoder == "auto" and nvenc_available())
        for codec in (["h264_nvenc"] if nv else []) + ["libx264"]:
            self.c = av.open(path, "w")
            self.s = self.c.add_stream(codec, rate=rate)
            self.s.width, self.s.height, self.s.pix_fmt = width, height, "yuv420p"
            if codec == "h264_nvenc":
                # p7 + B-frames + lookahead + two-pass: at the bitrate cap, 1080p, 0.99x source size,
                # PSNR-Y 41.9 dB vs 40.4 for p5, 134 fps (x264 medium: 1.06x, 46.2 dB, 26 fps).
                # Probed on T4 2026-11-11; the user chose speed over x264's extra quality.
                opts = {"preset": "p7", "tune": "hq", "rc": "vbr", "cq": str(crf + 2), "b": "0",
                        "spatial-aq": "1", "temporal-aq": "1", "bf": "3", "b_ref_mode": "middle",
                        "rc-lookahead": "32", "multipass": "fullres"}
            else:
                opts = {"crf": str(crf), "preset": preset}
            if max_bitrate:
                opts["maxrate"] = str(int(max_bitrate))
                opts["bufsize"] = str(int(max_bitrate * 2))
            self.s.options = opts
            self.s.codec_context.time_base = self.tb
            try:
                self.s.codec_context.open()   # fail here, not mid-video (e.g. NVENC sessions exhausted)
                break
            except Exception as e:
                self.c.close()
                if codec == "libx264":
                    raise
                print(f"note: NVENC unavailable ({e}); encoding with libx264", file=sys.stderr)
        self.codec = codec
        self.last = None
        # Encoding runs in its own thread (PyAV releases the GIL in the BGR -> YUV conversion and
        # the encoder), overlapping the caller's work: 225 s of a 1028 s remove on Kaggle's CPU.
        import queue
        import threading
        self.q, self.err = queue.Queue(8), None
        self.th = threading.Thread(target=self._encode, daemon=True, name="encode")
        self.th.start()

    def _encode(self):
        while True:
            x = self.q.get()
            if x is None:
                return
            if self.err is not None:
                continue   # keep draining so write() never blocks; close() raises
            try:
                bgr, pts = x
                f = av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr), format="bgr24")
                f.pts, f.time_base = pts, self.tb
                for p in self.s.encode(f):
                    self.c.mux(p)
            except Exception as e:
                self.err = e

    def write(self, bgr, pts):
        """bgr must not be modified afterwards (it is encoded later, in the writer thread)."""
        if self.err is not None:
            raise self.err
        if self.last is not None and pts <= self.last:
            pts = self.last + 1
        self.last = pts
        self.q.put((bgr, pts))

    def close(self):
        self.q.put(None)
        self.th.join()
        if self.err is not None:
            raise self.err
        for p in self.s.encode(None):
            self.c.mux(p)
        self.c.close()


def _add_copy(oc, s):
    if hasattr(oc, "add_stream_from_template"):
        return oc.add_stream_from_template(s)
    return oc.add_stream(template=s)


def mux_audio(video_only, source, out):
    """Copy the video stream of `video_only` and the audio streams of `source` into `out`."""
    with av.open(video_only) as vi, av.open(source) as si, av.open(out, "w") as oc:
        vs = vi.streams.video[0]
        ovs = _add_copy(oc, vs)
        amap = {a.index: _add_copy(oc, a) for a in si.streams.audio}

        def gen(container, streams, mapping):
            for p in container.demux(streams):
                if p.dts is None:
                    continue
                t = float(p.dts * p.time_base)
                yield t, id(p), p, mapping[p.stream.index]

        merged = heapq.merge(gen(vi, [vs], {vs.index: ovs}),
                             gen(si, list(si.streams.audio), amap))
        for _, _, p, os_ in merged:
            p.stream = os_
            oc.mux(p)


def finalize(video_only, source, out):
    """Attach source audio if present; fall back to the silent file if the copy fails."""
    try:
        with av.open(source) as si:
            has_audio = len(si.streams.audio) > 0
        if has_audio:
            mux_audio(video_only, source, out)
            os.remove(video_only)
            return True
    except Exception as e:
        print(f"warning: audio copy failed ({e}); writing video without audio", file=sys.stderr)
        if os.path.exists(out):
            os.remove(out)
    os.replace(video_only, out)
    return False
