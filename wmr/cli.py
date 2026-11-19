"""Command line interface: wmr <command> ..."""
import argparse
import json
import os
import sys

from . import __version__


def _texts(args):
    texts = []
    if getattr(args, "texts", None):
        with open(args.texts, encoding="utf-8") as fh:
            texts += [ln.strip() for ln in fh if ln.strip()]
    for t in getattr(args, "text", None) or []:
        texts.append(t)
    return texts


def _add_analyze(p):
    p.add_argument("--texts", help="file with one expected overlay string per line")
    p.add_argument("--text", action="append", help="expected overlay string (repeatable)")
    p.add_argument("--discover", action="store_true", help="also look for overlay text that is not on the list")
    p.add_argument("--sample-fps", type=float, default=3.0, help="OCR sampling rate (frames per second)")
    p.add_argument("--enhance", action="store_true", help="extra contrast-enhanced OCR pass for faint text")
    p.add_argument("--min-seconds", type=float, default=8.0, help="discovery: minimum on-screen time")
    p.add_argument("--cpu", action="store_true", help="do not use a GPU for OCR or fill")


def _add_remove(p):
    p.add_argument("--engine", choices=["builtin", "propainter"], default="builtin",
                   help="fill method for opaque overlays")
    p.add_argument("--propainter-dir", help="path of a ProPainter checkout")
    p.add_argument("--gpus", help="comma-separated GPU indices (default: all visible)")
    p.add_argument("--pp-chunk", type=int, default=240, help="frames per ProPainter job")
    p.add_argument("--window", type=int, default=40, help="frames of look-ahead/behind for the built-in fill")
    p.add_argument("--crf", type=int, default=21)
    p.add_argument("--preset", default="medium", help="libx264 preset (NVENC uses p7)")
    p.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto",
                   help="auto: NVENC on an NVIDIA GPU, else libx264 (x264: smaller loss of fine grain, ~5x slower)")
    p.add_argument("--no-bitrate-cap", action="store_true", help="do not cap bitrate near the source's")
    p.add_argument("--keep-work", action="store_true", help="keep ProPainter working files")
    p.add_argument("--force-fill", action="store_true",
                   help="treat semi-transparent overlays as opaque (use if blend inversion leaves ghost outlines)")
    p.add_argument("--inpaint", choices=["auto", "lama", "telea"], default="auto",
                   help="holes no nearby frame can fill: auto = LaMa on a CUDA GPU (downloads 196 MB once), else Telea")
    p.add_argument("--feather", type=int, default=3,
                   help="px beyond the mask that are filled and blended (0: hard edge, as before 0.6.0)")


def _gpus(args):
    if getattr(args, "gpus", None):
        return [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    return None


def _do_remove(args, video, tracks, out):
    from . import remove as rm
    patches = None
    work = None
    if args.engine == "propainter":
        from . import propainter as pp
        work = out + ".ppwork"
        patches = pp.build(video, tracks, work, pp_dir=args.propainter_dir, gpus=_gpus(args), chunk=args.pp_chunk,
                           force_fill=args.force_fill)
    tot = rm.remove(video, tracks, out, window=args.window, crf=args.crf, preset=args.preset, encoder=args.encoder,
                    device="cpu" if getattr(args, "cpu", False) else "auto",
                    cap_bitrate=not args.no_bitrate_cap, patches=patches, force_fill=args.force_fill,
                    inpaint=args.inpaint, feather=args.feather)
    if work and not args.keep_work:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return tot


def main(argv=None):
    from . import backend
    if not os.environ.get("WMR_THREADS"):
        n = backend.cpu_quota()
        if n < (os.cpu_count() or n):   # a CPU share smaller than the cores in view (RunPod pods)
            # ponytail: child processes (batch workers) inherit the whole share each
            os.environ["WMR_THREADS"] = str(n)
    if os.environ.get("WMR_THREADS"):   # see detect.py; OpenCV's pool, torch and BLAS (read at import, later)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(k, os.environ["WMR_THREADS"])
        import cv2
        cv2.setNumThreads(int(os.environ["WMR_THREADS"]))
    ap = argparse.ArgumentParser(prog="wmr", description="Text-overlay removal for your own videos.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="show detected hardware and library versions")

    p = sub.add_parser("analyze", help="find and track overlays; writes tracks.json")
    p.add_argument("video")
    p.add_argument("-o", "--out", help="tracks.json path (default: <video>.tracks.json)")
    p.add_argument("--preview", help="also write a preview video with the tracks drawn")
    _add_analyze(p)

    p = sub.add_parser("preview", help="draw tracks on the video")
    p.add_argument("video")
    p.add_argument("tracks")
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("remove", help="remove the overlays listed in tracks.json")
    p.add_argument("video")
    p.add_argument("tracks")
    p.add_argument("-o", "--out", required=True)
    _add_remove(p)

    p = sub.add_parser("verify", help="OCR the result and report frames where expected text is still readable")
    p.add_argument("video")
    p.add_argument("--texts")
    p.add_argument("--text", action="append")
    p.add_argument("--tracks", help="take the expected strings from a tracks.json")
    p.add_argument("--cpu", action="store_true")

    p = sub.add_parser("run", help="analyze + remove + verify in one go")
    p.add_argument("video")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--preview", help="also write a preview video with the tracks drawn")
    p.add_argument("--no-verify", action="store_true")
    _add_analyze(p)
    _add_remove(p)

    p = sub.add_parser("batch", help="run every video in a folder, one worker per GPU")
    p.add_argument("in_dir")
    p.add_argument("-o", "--out-dir", required=True)
    p.add_argument("--workers", type=int, help="parallel workers (default: one per GPU, else 1)")
    p.add_argument("--no-verify", action="store_true")
    _add_analyze(p)
    _add_remove(p)

    p = sub.add_parser("synth", help="make a test clip with known overlays and ground truth")
    p.add_argument("-o", "--out-dir", required=True)
    p.add_argument("--overlay", action="append", required=True, metavar="TEXT:MOTION:OPACITY[:SIZE[:shadow]]",
                   help="motion is static|linear|bounce|teleport; shadow adds a soft dark drop shadow; repeatable")
    p.add_argument("--clean", help="clean source video (default: a procedural clip)")
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--size", default="640x360")
    p.add_argument("--static-scene", action="store_true")
    p.add_argument("--dwell", type=int, default=60, help="teleport dwell in frames")
    p.add_argument("--speed", type=float, default=3.0, help="pixels per frame for moving overlays")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blobs", type=int, default=1, help="discs moving on their own (local motion)")
    p.add_argument("--blob-radius", type=int, default=26)
    p.add_argument("--exposure", type=float, default=0.0, help="brightness swing, +- levels")

    p = sub.add_parser("eval", help="score an output against a synth ground-truth folder")
    p.add_argument("gt_dir")
    p.add_argument("output")
    p.add_argument("--tracks")

    p = sub.add_parser("setup-propainter", help="clone ProPainter (optional, non-commercial licence)")
    p.add_argument("--dir")

    args = ap.parse_args(argv)

    if args.cmd == "doctor":
        from . import backend
        print(json.dumps(backend.describe(), indent=2))
    elif args.cmd == "analyze":
        from . import track
        out = args.out or os.path.splitext(args.video)[0] + ".tracks.json"
        track.analyze(args.video, out, texts=_texts(args), discover=args.discover, sample_fps=args.sample_fps,
                      enhance=args.enhance, use_gpu=not args.cpu, min_seconds=args.min_seconds)
        print(f"tracks: {out}")
        if args.preview:
            track.preview(args.video, out, args.preview)
            print(f"preview: {args.preview}")
    elif args.cmd == "preview":
        from . import track
        track.preview(args.video, args.tracks, args.out)
    elif args.cmd == "remove":
        _do_remove(args, args.video, args.tracks, args.out)
        print(f"output: {args.out}")
    elif args.cmd == "verify":
        from . import evalm
        texts = _texts(args)
        if args.tracks:
            with open(args.tracks) as fh:
                texts += [t["label"] for t in json.load(fh)["tracks"] if t["enabled"]]
        if not texts:
            sys.exit("verify needs --texts, --text or --tracks")
        print(json.dumps(evalm.verify(args.video, sorted(set(texts)), use_gpu=not args.cpu, tracks=args.tracks), indent=2))
    elif args.cmd == "run":
        import time
        from . import evalm, track
        from . import prof
        from .detect import OCRPool
        clock, secs = time.perf_counter, {}
        prof.phase("ocr init")
        t0 = clock()
        ocr = OCRPool(use_gpu=not args.cpu)   # one set of engines for analyze and verify
        secs["ocr init"] = clock() - t0
        print(f"[run] OCR {ocr.model} on {ocr.provider} x{len(ocr.engines)}, init seconds (slowest engine): "
              + ", ".join(f"{k} {v}" for k, v in ocr.init_secs.items()))
        tracks = os.path.splitext(args.out)[0] + ".tracks.json"
        t0 = clock()
        doc = track.analyze(args.video, tracks, texts=_texts(args), discover=args.discover,
                            sample_fps=args.sample_fps, enhance=args.enhance, use_gpu=not args.cpu,
                            min_seconds=args.min_seconds, ocr=ocr)
        secs["analyze"] = clock() - t0
        if args.preview:
            t0 = clock()
            track.preview(args.video, tracks, args.preview)
            secs["preview"] = clock() - t0
        t0 = clock()
        prof.phase("remove")
        tot = _do_remove(args, args.video, tracks, args.out)
        secs["remove"] = clock() - t0
        report = {"video": args.video, "output": args.out, "tracks": tracks, "remove": tot}
        labels = sorted({t["label"] for t in doc["tracks"] if t["enabled"]})
        if labels and not args.no_verify:
            t0 = clock()
            prof.phase("verify")
            report["verify"] = evalm.verify(args.out, labels, use_gpu=not args.cpu, ocr=ocr, tracks=tracks)
            secs["verify"] = clock() - t0
            print(f"[verify] {report['verify']['leak_frames']} of {report['verify']['checked_frames']} "
                  "sampled frames still show expected text")
            for k, r in report["verify"].get("residue", {}).items():
                print(f"[verify] track {k} residue (template match on the output; ~0.5-0.9 before removal): "
                      f"median {r['median']:.2f}, p90 {r['p90']:.2f}, {r['share_ge_0.35']:.0%} of frames >= 0.35")
        report["seconds"] = {k: round(v, 2) for k, v in secs.items()}
        print("[run] seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in secs.items()))
        with open(os.path.splitext(args.out)[0] + ".report.json", "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"output: {args.out}")
    elif args.cmd == "batch":
        from . import batch as b
        extra = []
        for flag in ("texts", "sample_fps", "min_seconds", "engine", "propainter_dir", "pp_chunk", "window",
                     "crf", "preset", "encoder"):
            v = getattr(args, flag)
            if v is not None:
                extra += ["--" + flag.replace("_", "-"), str(v)]
        for t in args.text or []:
            extra += ["--text", t]
        for flag in ("discover", "enhance", "cpu", "no_bitrate_cap", "no_verify", "keep_work", "force_fill"):
            if getattr(args, flag):
                extra.append("--" + flag.replace("_", "-"))
        res = b.batch(args.in_dir, args.out_dir, extra, gpus=_gpus(args), workers=args.workers)
        bad = [k for k, v in res.items() if v != 0]
        print(f"[batch] {len(res) - len(bad)} ok, {len(bad)} failed")
        sys.exit(1 if bad else 0)
    elif args.cmd == "synth":
        from . import synth
        w, h = (int(x) for x in args.size.lower().split("x"))
        synth.synth(args.out_dir, args.overlay, clean=args.clean, frames=args.frames, size=(w, h),
                    static_scene=args.static_scene, seed=args.seed, dwell=args.dwell, speed=args.speed,
                    blobs=args.blobs, blob_r=args.blob_radius, exposure=args.exposure)
        print(f"wrote {args.out_dir}/marked.mp4, clean.mp4, gt.json, texts.txt")
    elif args.cmd == "eval":
        from . import evalm
        marked = os.path.join(args.gt_dir, "marked.mp4")
        print(json.dumps(evalm.evaluate(args.gt_dir, args.output, args.tracks,
                                        marked if os.path.exists(marked) else None), indent=2))
    elif args.cmd == "setup-propainter":
        from . import propainter
        propainter.setup(args.dir)


if __name__ == "__main__":
    main()
