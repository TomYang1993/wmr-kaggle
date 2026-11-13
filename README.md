# wmr 0.2.1

Finds, tracks and removes **text overlays** (watermarks) from your own videos: static,
linear, bouncing and teleporting text, opaque or semi-transparent, several at once.
One Python package runs on Kaggle (2x T4), macOS (Apple Silicon) and Windows + NVIDIA; only the
install profile differs per platform.

## Changes in 0.2.1

- GPU text recognition now works with the OCR library version that installs on Python 3.13
  (0.2.0 printed `OCR GPU init failed ('model_path')` and ran on CPU).
- One overlay read by OCR in two box sizes no longer becomes two tracks. In 0.2.0 the duplicates
  also blocked each other's opacity estimate, so the overlay fell back to a box-shaped mask.
- Faint semi-transparent text is tracked more reliably: the overlay template is built from many
  aligned frames instead of one, small position jitter on static text no longer splits its
  opacity estimate, and text touching the frame edge is handled.

## Install

| Platform | Command |
| --- | --- |
| Kaggle | open `notebooks/kaggle_quickstart.ipynb`, or `!bash wmr-0.2.1/scripts/setup_kaggle.sh` |
| macOS | `bash scripts/setup_macos.sh` |
| Windows + NVIDIA | `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1` |
| Anything else | `pip install .` (CPU) |

Add `propainter` (bash) or `-ProPainter` (PowerShell) to also fetch the optional ProPainter engine.
No system FFmpeg is needed: the PyAV wheel carries its own. Check an install with `wmr doctor`
and `python tests/test_smoke.py`.
On Windows the GPU builds of torch and onnxruntime-gpu are CUDA 13: NVIDIA driver 580 or newer.

## Quick start

```bash
# texts.txt: one overlay string per line (each line of a multi-line watermark separately)
wmr run in.mp4 -o out.mp4 --texts texts.txt --preview preview.mp4
```

`run` = `analyze` + `remove` + `verify`. It writes `out.mp4`, `out.tracks.json` (+ `out.tracks.d/`),
`out.report.json`, and the optional preview with every track drawn on it.

Two-step use, so you can review before the slow part:

```bash
wmr analyze in.mp4 --texts texts.txt -o in.tracks.json --preview preview.mp4
#   ... look at preview.mp4, edit in.tracks.json if needed ...
wmr remove in.mp4 in.tracks.json -o out.mp4
wmr verify out.mp4 --tracks in.tracks.json
```

Other commands: `wmr batch DIR -o OUTDIR ...` (one worker per GPU), `wmr synth` (test clips with
ground truth), `wmr eval` (score against that ground truth), `wmr doctor`, `wmr setup-propainter`.

### When you do not know the texts

Leave out `--texts`, or add `--discover` next to it to catch strings you forgot:

```bash
wmr analyze in.mp4 --discover --texts texts.txt --preview preview.mp4
```

Discovery keeps text that stays on screen across a scene cut, or for at least `--min-seconds`
(default 8). Burned-in titles can satisfy the same rule, so check the printed list: tracks marked
`DISABLED - review` are not removed until you set `"enabled": true` in the tracks file, and you
can set `"enabled": false` on anything you want to keep.

## How it works

1. **Analyze, pass A.** OCR (PP-OCRv6 via RapidOCR 3.x on ONNX Runtime) on about 3 frames per second; fuzzy match
   against your strings; scene-cut detection.
2. **Analyze, pass B.** A gradient template per overlay is matched on every frame, so positions
   are pixel-exact and teleport jumps land on the right frame. Gaps are bridged; moving overlays
   are carried through frame-edge entry/exit and unreadable stretches with a constant-velocity,
   wall-bouncing motion model. Aligned crops are accumulated to estimate each overlay's opacity
   map and colour.
3. **Remove.** One decode, one encode, all overlays together:
   - tier `alpha` (semi-transparent): the blend is inverted, `I = (J - aW) / (1 - a)`;
   - tier `fill` (opaque): real pixels are borrowed from nearby frames of the same shot after
     global-motion compensation; what was never visible is inpainted and kept stable over time.
4. **Verify.** OCR on the result reports frames where an expected string is still readable.

Opacity at or above 0.7 is treated as opaque. Output is H.264, CRF 21, bitrate capped near the
source's, original timestamps kept, audio stream-copied.

### tracks.json fields you may want to edit

| Field | Meaning |
| --- | --- |
| `enabled` | remove this track or not |
| `tier` | `alpha` or `fill`; switch to `fill` if inversion leaves ghost outlines (or use `--force-fill`) |
| `alpha_gain` | multiply the estimated opacity, e.g. 1.05 if a faint ghost of the text remains, 0.95 if a negative one does |
| `spans` | `[first_frame, last_frame, x, y]` runs; x, y is the top-left of the padded box |

## Engines for opaque overlays

| Engine | Needs | Best for |
| --- | --- | --- |
| `builtin` (default) | CPU only | moving and teleporting overlays; static overlays when the camera moves |
| `propainter` | NVIDIA GPU, `wmr setup-propainter` | static opaque overlays on locked-off shots, objects moving behind the text |

```bash
wmr run in.mp4 -o out.mp4 --texts texts.txt --engine propainter --gpus 0,1
```

With ProPainter, only cropped regions around the overlays are processed, cut into jobs of
`--pp-chunk` frames that are spread over the listed GPUs. ProPainter is not bundled; it is
cloned from its repository and is under the NTU S-Lab License 1.0 (non-commercial use only).

## Running on your own videos (Kaggle, from this Mac)

Only for videos you own or have the rights to edit.

1. **Upload the videos once** as a private dataset. Put them in a folder with a
   `dataset-metadata.json` such as `{"id": "yxyxtx/my-videos", "title": "my videos", "licenses": [{"name": "unknown"}]}`,
   then `kaggle datasets create -p <folder>` (new versions: `kaggle datasets version -p <folder> -m "msg"`).
2. **Point the notebook at it**: in `kaggle/real/kernel-metadata.json` replace `yxyxtx/YOUR-VIDEO-DATASET`,
   and in the first cell of `kaggle/real/wmr-production-test.ipynb` set `VIDEO_DATASET`. That cell is the only one to edit:
   - `MODE = 'single'` with `VIDEO = 'name.mp4'` (or `''` for the first video): one video, plus a preview video.
   - `MODE = 'batch'`: every video in the dataset, one worker per T4.
   - `TEXTS`: watermark strings you know, one per line. Listed strings are always removed.
     Empty = discovery only (confident finds removed, uncertain ones reported as "needs review").
   - `ENCODER`: `'auto'` (NVENC, fast) or `'x264'` (keeps fine grain, about 5x slower).
3. **Run**: `KERNEL=kaggle/real KERNEL_ONLY=1 kaggle/push.sh` (drop `KERNEL_ONLY=1` after changing the code;
   for long batches add `TIMEOUT=20000`). Watch with `kaggle kernels status yxyxtx/wmr-production-test`.
4. **Download**: `kaggle kernels output yxyxtx/wmr-production-test -p kaggle/real/out`.

**What comes back** (in `out/`), per video:

| File | What it is |
| --- | --- |
| `<name>_clean.mp4` | the result: overlays removed, original audio copied, re-encoded once at about the source's size |
| `<name>_clean_stills.jpg` | input vs output side by side at 20%, 50% and 80% of the video: the quickest look |
| `<name>_preview.mp4` | single mode only: the input with every track drawn (red = removed, yellow = needs review) |
| `<name>_clean.tracks.json` | what was found: text, removed or not, motion type, opacity, frame range, per-frame positions |
| `<name>_clean.report.json` | timings per stage, encoder, and the leftover check |
| `<name>.log` | batch mode only: the full log of that video |

**How to judge a result** (real videos have no ground truth, unlike `wmr synth`):

- Look at `_stills.jpg`, then play `_clean.mp4` where the watermark was, ideally full screen.
- The notebook prints, per video, each track as REMOVED or "needs review", and the leftover check:
  how many of the sampled output frames OCR can still read an expected string in (0 is the goal).
- A real watermark marked "needs review": add its text to `TEXTS` and run again. Something removed that should
  stay (a burned-in title): leave `TEXTS` as is and set `DISCOVER = False`.
- Wrong position or a missed stretch: the `_preview.mp4` shows where tracking lost it.

## Using both Kaggle GPUs

- `wmr batch /kaggle/input/myvideos -o /kaggle/working/out --texts texts.txt --gpus 0,1`
  runs two videos at a time, one per GPU.
- A single video with `--engine propainter --gpus 0,1` splits its inpainting jobs over both.
- With the default engine the GPU runs OCR (PP-OCRv6 medium) and H.264 encoding (NVENC p7, about 5x libx264
  medium on T4; at the default bitrate cap it smooths fine grain slightly more than x264); decode, tracking and fill stay on CPU. `--encoder x264` forces software encoding.

## Tested, and not

Tested on Linux, CPU only, on synthetic clips with exact ground truth (`python tests/test_smoke.py`
and the cases in `wmr synth`): static, linear, bouncing and teleporting overlays at opacity 0.25
to 1.0, three overlays at once, audio copy, discovery mode, batch mode.

Run on Kaggle T4: CUDA OCR and NVENC (synthetic clips). Not yet run by the author: macOS, Windows, and the real ProPainter model
(its adapter was exercised only against a stand-in script with the same interface). `wmr doctor`
shows what each machine actually provides; OCR falls back to CPU if the GPU provider fails.

## Limits

- Text only. Logos and pictures are not detected.
- Rigid overlays: no rotation, scaling or per-letter animation.
- The opacity estimate needs the background under the overlay to change over time; when it does
  not, the track becomes a box-shaped `fill`.
- A static opaque overlay over a static shot can only be inpainted; with the built-in engine
  that looks soft. Use the ProPainter engine there.
- Output is 8-bit H.264. 10-bit/HDR sources are converted; rotation metadata is not applied
  (normalise such files with FFmpeg first).
- Overlay colour is assumed uniform; outlined or gradient text is routed to `fill`.
