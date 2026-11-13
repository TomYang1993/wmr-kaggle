"""Rec through _ShapeBuckets (batch and width zero-padded, one session per width bucket) must read
what RapidOCR reads unpadded, as wmr compares text (norm_text: letters and digits; the extra
padding can move a space). Runs on the CPU provider (the buckets are only used on CUDA, where
a change of input shape is expensive).  python tests/test_rec_buckets.py"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wmr.detect import OCR, _put, _sessions, _ShapeBuckets, _warmup_image, norm_text  # noqa: E402


def frame(k):
    img = np.full((720, 1280, 3), 60 + 20 * k, np.uint8)
    lines = ["@studio_mark", "LIVE 2026", "www.example-site.com", "ID 4417-QX", "hello2026 sample text line"]
    for j, t in enumerate(lines[: 2 + k]):   # 2..5 crops: batch size and widths change per frame
        cv2.putText(img, t, (40 + 30 * j, 90 + 120 * j), cv2.FONT_HERSHEY_SIMPLEX, 1.0 + 0.3 * (j % 3),
                    (255, 255, 255), 2)
    return img


ocr = OCR(use_gpu=False)
imgs = [frame(k) for k in range(4)] + [_warmup_image()]
plain = [sorted(norm_text(d["text"]) for d in ocr(im)) for im in imgs]

import onnxruntime as ort  # noqa: E402
for owner, key, s, is_map in _sessions(ocr.engine):
    if s.get_inputs()[0].shape[2:3] == [48]:
        path = s._model_path
        made = []
        b = _ShapeBuckets(lambda: made.append(1) or ort.InferenceSession(path, providers=["CPUExecutionProvider"]), 6)
        _put(owner, key, b, is_map)
bucketed = [sorted(norm_text(d["text"]) for d in ocr(im)) for im in imgs]
for p, q in zip(plain, bucketed):
    print(p, "|", q)
assert plain == bucketed
print(f"ok: {len(made)} sessions for widths {sorted(b.by_w)}")
