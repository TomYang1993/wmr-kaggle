"""End-to-end smoke test on a small synthetic clip (about a minute on a laptop CPU).
Run:  python tests/test_smoke.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wmr import evalm, remove, synth, track  # noqa: E402


def main():
    d = tempfile.mkdtemp(prefix="wmr_smoke_")
    synth.synth(d, ["@my_channel:bounce:1.0", "SAMPLE MARK:static:0.45", "hello2026:teleport:1.0"],
                frames=150, seed=3)
    texts = open(os.path.join(d, "texts.txt")).read().splitlines()
    marked, tracks, out = (os.path.join(d, n) for n in ("marked.mp4", "tracks.json", "out.mp4"))
    doc = track.analyze(marked, tracks, texts=texts, use_gpu=False)
    assert len(doc["tracks"]) == 3, [t["label"] for t in doc["tracks"]]
    remove.remove(marked, tracks, out)
    ev = evalm.evaluate(d, out, tracks, marked)
    print(ev)
    assert ev["frames_with_mask_recall_below_95pct"] == 0, ev
    assert ev["psnr_in_mask_after"] > ev["psnr_in_mask_before"] + 8, ev
    leaks = evalm.verify(out, texts, use_gpu=False)
    print(leaks)
    assert leaks["leak_frames"] <= 1, leaks
    print("smoke test passed:", d)


if __name__ == "__main__":
    main()
