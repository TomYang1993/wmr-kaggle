# HANDOFF: wmr (video text-watermark removal)

For the next Claude Code session. Read this first, then `README.md`.

## Goal

Detect, track and remove **text** overlays (watermarks) from the user's own videos. The overlays can be:

- static, linear, bouncing or teleporting;
- held in one position for 3+ minutes;
- opaque or semi-transparent;
- several per video.

Fonts and sizes are unknown, and the strings are sometimes unknown too (discovery mode). Logos and pictures are out of scope.

Targets: Kaggle (2x T4) for testing, plus macOS (Apple Silicon) and Windows + NVIDIA locally.

Design doc (Claude Docs): https://claude.ai/artifact/6zqYnFvd3cvshhJTfsGJZP

## Current state: v0.6.2 (history below)

All synthetic tests pass on Linux CPU. The self-test also ran end to end on Kaggle with v0.2.0; the bugs that run exposed are fixed in 0.2.1.

**Not yet verified:**

- the real ProPainter model (only a stand-in script with the same CLI was tested);
- macOS and Windows installs;
- any real user video.

## Next step: closed loop with the Kaggle CLI on the user's machine

The goal is that the user no longer relays code and errors by hand.

**Status of the setup:**

- The user is setting up `pip install kaggle` and `~/.kaggle/kaggle.json`.
- Never print, copy or commit `kaggle.json`.
- Kaggle username: `yxyxtx` (from the dataset path).

**Inputs:**

- Video dataset: mounted at `/kaggle/input/datasets/yxyxtx/video-test-set-1`. It holds about 20 files named `BJ  1.mp4` … `BJ  20.mp4`; note the double spaces.
- Code dataset: currently uploaded by hand under the title "watermark removal core code". Its slug is unknown; find it with `kaggle datasets list --mine`.

**Loop:**

1. Check access with `kaggle datasets list --mine` and `kaggle kernels list --mine`.
2. Put the package folder (without `HANDOFF.md`, if preferred) in a dataset dir with a `dataset-metadata.json` holding the id `yxyxtx/<slug>`. Publish a new version with `kaggle datasets version -p <dir> -m "msg" --dir-mode zip`. Alternatively, create a new dataset `yxyxtx/wmr-code` with `kaggle datasets create`.
3. Put `notebooks/kaggle_quickstart.ipynb` in a kernel dir with a `kernel-metadata.json`:
   - `"id"`: `"yxyxtx/wmr-test"`
   - `"code_file"`: the notebook
   - `"language"`: `"python"`
   - `"kernel_type"`: `"notebook"`
   - `"enable_gpu"`: `true`
   - `"enable_internet"`: `true`
   - `"dataset_sources"`: `["yxyxtx/<code-slug>", "yxyxtx/video-test-set-1"]`

   Two things to verify:
   - how to request **T4 x2** rather than P100 (a `machine_shape` field, or a setting in the UI). Check `kaggle kernels push --help` and the current Kaggle API docs.
   - the exact video dataset slug.
4. Start the run with `kaggle kernels push -p <kernel dir>`, poll `kaggle kernels status yxyxtx/wmr-test`, then fetch logs and outputs with `kaggle kernels output yxyxtx/wmr-test -p ./out`.
5. Pushed notebooks run as non-interactive "Save & Run All" versions. Batch-mode notebooks cannot pause for review, so:
   - for the review step (discovered tracks need `enabled: true`), run analyze, then download `*.tracks.json` and the preview for the user;
   - or rely on `--discover` defaults and report the "needs review" list.
6. GPU quota is about 30 h/week shared with the user's other work. Keep test runs short: one video first, a 300-frame synth.

Optional, as the user may want it later: a GitHub repo, with the notebook running `pip install git+https://github.com/<user>/wmr@main`.

## Session 2026-11-10 (Kaggle loop + GPU OCR fix)

- `kaggle/push.sh` uploads the package as a new version of `yxyxtx/watermark-removal-core-code` and runs kernel `yxyxtx/fable-5-2-watermark-detection-test` on T4 x2 (`--accelerator NvidiaTeslaT4`). `KERNEL_ONLY=1` skips the upload. Fetch results with `kaggle kernels output <kernel> -p kaggle/out`.
- The test kernel is synthetic-only. `video-test-set-1` is explicit third-party footage (19tv.info watermark), not the user's own; don't run removal on it.
- GPU OCR fix in `detect.py`: cuDNN 9.19 (pinned by torch 2.11+cu130) has no engine for the det model's NCHW convs on T4 at inputs >= ~736 px, and ORT silently rebuilt the session on CPU. Now `setup_kaggle.sh` upgrades to `nvidia-cudnn-cu13>=9.28` (torch still runs; pip warns about its pin), and `OCR` tries CUDA option sets in order (plain NCHW, then `prefer_nhwc=1`), each checked by a warm-up run with `disable_fallback()`, then CPU with a logged note. Probed det call times at 4K: 143 ms NCHW on 9.28 vs 252 ms NHWC on 9.19; every size up to 2176x3840 works with either. Verified on Kaggle: OCR on CUDA with plain NCHW, reads identical to CPU on 30/30 frames, 0 cuDNN warnings, eval unchanged (11.9 -> 29.8 dB).
- 0.3.0 (2026-11-11): OCR is PP-OCRv6 via `rapidocr>=3.10` (medium models on CUDA, small elsewhere); RapidOCR builds sessions lazily, so `OCR` warms up once on CPU, then finds the sessions with `_sessions()` and rebuilds them on CUDA. Encode uses `h264_nvenc` (preset p5, tune hq, vbr, cq = crf + 2, spatial AQ) when an NVIDIA GPU is visible, with libx264 fallback; `--encoder auto|nvenc|x264`. Decode stays software (231 fps vs 133 fps with PyAV CUDA hwaccel at 1080p). Stage timers in `[analyze]`/`[remove]` log lines.
- Kaggle T4, 300-frame synth, GPU mode vs `--cpu --encoder x264`: 720p clip 103 s vs 126 s total, 1080p faint-text clip 85 s vs 181 s. OCR 7.1 vs 29.2 s (720p) and 13.1 vs 60.8 s (1080p); encode 1.2 vs 7.7 s and 3.1 vs 16.2 s. Quality in mask about equal (29.3 vs 29.5 dB; 30.6 vs 31.1 dB).
- Fill borrowing on GPU (2026-11-11): `remove.py` keeps the window's frames on CUDA (`e["g"]`, `e["gm"]`) and `_borrow_torch` mirrors `_borrow_cv` (same sources, order, photometric check; grid_sample warps). Telea leftovers and the carried blend stay on CPU. `--cpu` keeps OCR and fill on CPU. `tests/test_fill_torch.py` checks torch vs OpenCV (torch CPU on a Mac, CUDA on Kaggle; both: same borrowed counts, mean diff 0.04). Kaggle 720p synth: fill 34-39 s -> 9.5 s, PSNR in mask unchanged (29.33 dB), total run 103 s -> 68.5 s (CPU mode 108 s). MPS not enabled until measured.
- Encoder choice (2026-11-11): on normal-bitrate sources (x264 crf 23) with the 1.15x bitrate cap, 1080p: x264 medium 1.06x size / PSNR-Y 46.2 dB / 26 fps; NVENC p5 0.92x / 40.4 dB / 153 fps; NVENC p7 + bf 3 + b_ref_mode middle + lookahead 32 + two-pass 0.99x / 41.9 dB / 134 fps; hevc_nvenc p7 1.07x / 42.8 dB / 86 fps. At 3x zoom NVENC smooths film grain slightly. User chose NVENC p7 as default (speed); `--encoder x264` for quality.
- Open: about 40 s of each run is outside the timed stages (process start, OCR model load, verify); verify leak counts differ between modes because the verifier's OCR model differs (medium on GPU reads faint remnants better).
- Synth verify still shows 1 of 20 sampled frames with leftover text (unchanged by the fix).

## Session 2026-11-11 (0.4.0: both GPUs, fewer syncs, phase timers)

- `OCRPool` (`detect.py`): one OCR engine per visible NVIDIA GPU, each fed by its own thread, input order kept, at most 2 frames per engine in flight. Pass A and verify use it; `wmr run` builds the engines once and shares them (`analyze(..., ocr=)`, `verify(..., ocr=)`). `OCR(device_id=)` sets the CUDA EP device.
- `_borrow_torch` judges every source at once and does one device->host sync per frame (the accept test reads only the ring, which borrowing never writes, so "first accepted source per pixel" equals the old greedy loop). `tests/test_fill_torch.py` now has 4 cases (moving overlay, brightened frames that must be rejected).
- `remove` runs decode + alpha + motion + upload in a background thread (`vio.prefetch`). `wmr run` prints `[run] seconds: ocr init, analyze, preview, remove, verify` and stores them in the report.
- libx264 is not bit-repeatable at scene cuts with the bitrate cap (0.3.0 too: 2 of 5 identical runs differ at frames 77-78 of a synth clip); NVENC/encoder input is identical run to run.
- A/B on Kaggle T4 x2, synthetic, same inputs (0.3.0 -> 0.4.0 wall): 720p 300 frames 69.8 -> 61.7 s; 1080p 900 frames 194.8 -> 147.4 s. Pass A 39.9 -> 21.3 s, verify on 2 GPUs. PSNR, recall, flicker, tracks and leak counts identical. Fill barely moved (38.0 -> 36.4 s), so it is not sync-bound.
- OCR probe, 90 1080p frames, one engine: cuDNN algo search EXHAUSTIVE 32 s, HEURISTIC 33 s, DEFAULT 109 s (keep the default EXHAUSTIVE); `OCRPool` x2 18.6 s. About 360 ms per 1080p OCR call with low GPU use, so it is probably CPU-bound inside RapidOCR pre/post-processing.
- Where a 1080p run spends time now: OCR init ~35 s (two engines, each a CPU warm-up then a GPU warm-up), fill ~40 ms/frame, verify (`enhance=True` doubles OCR), pass B ~19 ms/frame on CPU. Next levers, biggest first: time inside `fill_frame` (borrow vs Telea vs copies); verify OCR on track crops only; build the engines in parallel with a small CPU warm-up image; RapidOCR call breakdown.
- `test-dataset-2` was described as the user's own footage, but its first video showed third-party site and username marks, so it was dropped from the kernel. Like `video-test-set-1`, don't run removal on it.

## Session 2026-11-11, later (0.5.1: OCR start-up, batched fill warp)

- OCR start-up was a medium-model CPU warm-up inference (~16 s per engine on Kaggle's CPU). `OCR` now calls RapidOCR's `_load_det_model()` / `_load_rec_model()` (download + CPU session build, no inference), then moves the sessions to CUDA and warms up there. `OCRPool` builds one engine per GPU in parallel; downloads are serialized by a lock. `[run]` prints the split (`load`, `gpu build+warm-up`).
- `_borrow_torch` warps all source frames in one batched `grid_sample` over a shared source region, with a fixed number of host<->device copies per frame (each small `torch.as_tensor(..., device=)` upload was a sync). `[remove]` prints `fill: borrow` and `fill: inpaint`.
- Tried and reverted in 0.5.0: verify on track crops. More, smaller OCR calls made it slower (720p synth 3.5 -> 9.4 s), and CLAHE on small crops read differently (1 vs 4 leak frames of 20). Verify stays full-frame on all GPUs.
- A/B on Kaggle T4 x2, synthetic (0.4.0 -> 0.5.1 wall): 720p 300 frames 61.2 -> 36.8 s; 1080p faint text 300 frames 62.5 -> 32.7 s; 1080p 900 frames 149.2 -> 113.7 s (0.3.0: 194.8 s). OCR init 35 -> 2-10 s. PSNR, recall, flicker unchanged; verify leak counts within one frame (borderline reads).
- 0.5.2: `fill_frame` fills each separate part of the mask in its own region (`_regions`: padded boxes from a dilated mask's connected parts, overlapping boxes merged). The photometric check and the carried blend are now judged per region. Kaggle A/B vs 0.5.1, 1080p 900 frames: fill 34.3 -> 22.7 s (borrow 22.4 -> 9.6, inpaint 6.0 -> 1.4), wall 112.8 -> 101.5 s; PSNR in mask 31.27 -> 31.36 dB, 720p 29.30 -> 29.61 dB; recall and flicker unchanged; verify leaks within 2 frames of 60 (borderline reads). `tests/test_fill_torch.py` has a two-overlay case.
- Was: fill worked on one bounding box around all masked pixels, so overlays at opposite ends of the frame make it warp most of the frame per source (1080p 900-frame clip: borrow 25 ms/frame vs ~3 ms in the single-box CUDA test); fixed in 0.5.2. Next: OCR (pass A + verify, ~44 s of 101 s on the 1080p clip), pass B (CPU template matching, ~18 ms/frame at 1080p) and the per-call cost of RapidOCR (~360 ms per 1080p frame with the GPU mostly idle).

## Session 2026-11-11, night (0.5.3: OCR per call 403 -> 137 ms)

Kaggle's CPU is 4 vCPU (2 cores + HT) Xeon @ 2.0 GHz, much weaker than the user's M1 Max, so OCR was profiled stage by stage on Kaggle (`scripts/ocr_profile.py VIDEO [frames]`: one engine, then the pool with 1 and 2 threads per engine, per-stage ms per call). One 1080p call was 403 ms: det pre 104 (CPU numpy), det infer 176, rec infer 110, the rest ~12.

- det preprocessing: RapidOCR's normalise builds numpy float64 temporaries; `_DetPre` does it with one `cv2.dnn.blobFromImage` (same values to 1e-7). 104 -> 15-35 ms on Kaggle.
- det infer is real T4 compute: fp32 at 1088x1920 = 160 ms of GPU (copies only 7 ms; the old "143 ms per 4K call" note was wrong). `_fp16()` converts det and rec to fp16 with `onnxconverter-common` (now in `requirements/kaggle.txt`) and caches `<model>.fp16.onnx` next to the original. ReduceMean must stay fp32: the det model's squeeze-excite global pools sum a whole 1080p feature map and overflow fp16 to NaN (all-fp16 gave 2M NaN pixels on CUDA; CPU EP does not reproduce it). det 163 -> 89 ms, rec at a fixed shape 19.5 -> 11 ms. `WMR_OCR_FP16=0` keeps fp32.
- rec infer was shape churn, not compute: ORT's cuDNN Conv rebuilds its state whenever the input shape differs from the session's previous run (~75 ms for the rec model; host time all in Conv; EXHAUSTIVE / HEURISTIC / width buckets on one session / workspace / arena options make no difference; DEFAULT is 3x slower). RapidOCR's crop batches change width and count every call. `_ShapeBuckets` pads the batch to 6 and the width up to a bucket (160 px steps to 1280, then 1600 / 1920 / 2560) with one session per bucket. Rec 100 -> 20 ms. Extra padding can move a space in a read; `tests/test_rec_buckets.py` checks `norm_text` reads are unchanged.
- `OCRPool.imap` runs each GPU engine from 2 threads (overlaps one frame's CPU work with another's GPU run): pool 82 -> 65 ms per 1080p frame in the profile. CLAHE is created per call (its object is not thread-safe).
- Not useful: ORT 1 thread / no spin-wait (no change); TensorRT EP is listed but its libraries are not installed on Kaggle.
- A/B on Kaggle T4 x2 (0.5.2 -> 0.5.3 wall, fp32-OCR arm in brackets): 1080p 900 frames 102.1 -> 79.6 s [83.1]; pass A 21.0 -> 9.7 [11.1]; verify 23.1 -> 10.1 [13.5]; OCR init 4.9 -> 6.5 (fp16 sessions; the first run on a fresh machine also converts, ~4 s). 1080p faint 300 frames 32.3 -> 26.8 s. The fp32 arm reproduces 0.5.2 quality exactly, so only fp16 moves it: PSNR in mask long 31.36 -> 31.42, synth1 29.61 -> 29.26, hard 30.64 -> 30.65; recall within 0.001; verify leaks 2 -> 0 (long), 2 -> 1 (hard). fp16 shifts OCR boxes by a pixel, which moves pass B's tracks slightly.
- Split question (user): OCR on the Mac and removal on Kaggle does not pay. `wmr analyze` on the same 1080p clip on an M1 Max took 66 s (pass A 57 s with the small model on CoreML, plus a spurious extra track) vs 28.8 s on Kaggle now (pass A 9.7 s, medium fp16).
- Where the 1080p 900-frame run (79.6 s) goes now: OCR init 6.5, pass A 9.7, pass B 16.6 (CPU template matching), remove 32.6 (fill 23.5 of which borrow 9.9; alpha+motion 15.4 on CPU in the background thread; encode 7.4), verify 10.1. Next levers: pass B on GPU, the CPU-side fill overhead (~12 s), alpha+motion on GPU.
- `kaggle/baseline/wmr-0.5.3.zip` is the next baseline and the notebook says `VERSION, BASE = '0.5.4', '0.5.3'`: bump `wmr/__init__.py` and `pyproject.toml` to 0.5.4 before the next A/B push.

## Session 2026-11-12 (0.5.4-0.5.8: real 1080p video, profiler, 2199 -> 1555 s)

First real user video: `yxyxtx/huohua-studio-test-data` 072q-1 (1080p30, 942 s, 28258 frames, four "Huohua Studio" marks; run with `--texts`, no discovery, by the check kernel `kaggle/check` = `yxyxtx/wmr-overlay-check`, which keeps the output video on Kaggle for the user and returns text only; fetch with `kaggle/fetch.sh DIR yxyxtx/wmr-overlay-check '\.txt$'`). v12 / v27 of that set carry third-party marks: not processed.

- `WMR_PROF=file` (wmr/prof.py): sampler thread charges every thread's wall time to its innermost wmr line every 10 ms, per phase (`prof.phase()` in cli/track), plus per-native-thread CPU from /proc. The check kernel adds GPU util per phase.
- 0.5.4 profile (2199 s): every phase main-thread / CPU bound on the 4 vCPU; GPUs mostly idle (pass B 0%, remove GPU0 13% GPU1 0%). Pass A 286 (decode + BGR conversion of every frame for cut detection), pass B 539 (matchTemplate), remove 1028 (full-frame mask labelling 317, encode 225 on the main thread, inpaint 118), verify 306 (decoding and converting all frames for a 1-in-15 sample).
- 0.5.6 fixes: `_regions` on an 8x8-block mask (uint64 view), 11 -> 0.2 ms/frame; `VideoWriter` encodes in its own thread; `read_frames(convert=False)` + `vio.luma()` (Y plane view; PyAV's `to_ndarray("gray")` is slower than BGR); verify converts only sampled frames; pass B searches tracks in a thread pool (same results); `WMR_THREADS` caps ORT/OpenCV threads per process; OCR models ship in the private dataset `yxyxtx/wmr-ocr-models` (setup_kaggle.sh copies them; RapidOCR checks sha256).
- 0.5.5 regression, reverted: stopping pass B's candidate search at the first match >= T_HI picks the wrong instance when several identical marks are on screen (072q-1 verify 12 -> 906 of 1884 leaking; track 0 teleport -> bounce). The A/B now includes `same` (four identical marks); 0.5.4 and 0.5.6 both handle it badly (recall 0.55, all sampled frames leak): a real weakness for identical crossing marks.
- 072q-1 on 0.5.6: 1734 s (-21%), identical tracks and verify (12 of 1884). Pass A 240, pass B 455, remove 709 (encode hidden), verify 304. Synthetic A/B: quality identical on all four clips; long 76.3 -> 63.3 s.
- Now CPU-throughput bound almost everywhere (pass A 3.7 cores, pass B 3.1, remove 3.25, verify 3.8 of 4). Further gains need work moved off the CPU: pass B NCC on the GPU, NVDEC decode (the video is decoded four times), inpaint, alpha+motion; or another machine (TPU VM host CPU probe `kaggle/tpu`, `ACC=TpuV5E8 kaggle/push.sh`, queued for hours on 2026-11-12).
- 0.5.7-0.5.8: pass B template search on the GPU (`_TorchMatch` in track.py: grad + TM_CCOEFF_NORMED per candidate window, FFT correlation, float64 integral-image window sums, one host sync per frame; `WMR_MATCH_DEVICE=none` forces OpenCV, `cpu` runs torch on the CPU for tests; `tests/test_match_torch.py` checks scores vs cv2, 2.4e-7). A direct conv2d with the ~60x300 template (0.5.7) kept the T4 86% busy for 390 s; FFT: 247.6 s, GPU 35%. Track positions identical to the OpenCV path on the synthetic clips; 072q-1 tracks and verify unchanged.
- 072q-1 on 0.5.8: 1555 s (2199 on 0.5.4, -29%): pass A 243, pass B 248, remove 722, verify 316. Pass B is now host-bound (frame upload 53 s, torch launch overhead ~150 s; process at 3.67 of 4 cores).
- Baseline is `kaggle/baseline/wmr-0.5.6.zip`; the notebook says `VERSION, BASE = '0.5.8', '0.5.6'`. Bump the code version before the next A/B push.
- Windows (user's own box, RTX 3070-class 8 GB, never run yet): `setup_windows.ps1` now installs torch from the cu130 index (onnxruntime-gpu 1.31 on PyPI is a CUDA 13 build and borrows torch's CUDA/cuDNN DLLs; cu124 torch would leave OCR on the CPU; needs driver 580+), and `requirements/windows.txt` adds onnx + onnxconverter-common for fp16 OCR. prof.py no longer crashes on Windows (no /proc: wall samples only). Watch GPU memory there: remove peaked at 7.6 GB on a T4 (OCR sessions kept for verify + ~81 frames on the GPU + torch cache).

## Session 2026-11-17 (0.6.0-0.6.1: fill quality — LaMa, consensus borrowing, feather, two-tone marks)

User saw hard-edged, unblended removal regions in 072q-1's output (detection was fine). Causes found in the fill
code, not detection: borrowed patches pasted with a hard edge after a whole-ring mean check (local motion got
through), Telea smear for the rest, and shadowed marks either un-blended into an embossed ghost (alpha tier)
or leaving a shadow outline (fill mask ~3 px past the glyphs). Ideas taken from the remove-watermark-pro skill
(in-process big-lama TorchScript, 50 px context crops, feathered composite), minus its Mac parts (MPS,
VideoToolbox, RAM disk, frames on disk).

- `wmr/lama.py`: big-lama.pt (IOPaint's build, md5 e3aa4aaa..., Apache-2.0, 196 MB) via `torch.jit.load`,
  downloaded once to the torch hub cache (or `WMR_LAMA`); padded to x64 (symmetric). fp32 only (FFT layers).
  `--inpaint auto|lama|telea` (auto = LaMa when the fill runs on CUDA); holes < 64 px stay Telea; LaMa sees the
  region + 48 px with other marks masked.
- Borrowing (both paths now the same batched algorithm; CPU path vectorised, slower on CPU but identical):
  per-source brightness offset (<= 24 levels) measured on the ring; reject a source if the ring mean OR any
  16 px ring tile is off (> 2x photo_thr); per pixel take the nearest accepted source that is within 2x
  photo_thr of the lower median of all accepted sources (>= 3) — rejects objects passing under the mark.
  `tests/test_fill_torch.py` has a brightness-drift case and a local-object case; both fail with the checks off.
- `--feather` px (default 3): region filled to 3 px past the mask, weight 1 to 1.5 px then linear to 0.
  8 px caused borrow artifacts and scored worse; 3 px removed shadow dashes that 0 px left (visual check).
- Two-tone detection (`alpha.estimate` "two_tone": share of glyph pixels with the opposite brightness change):
  0.42-0.51 on shadowed synthetic marks, 0.08-0.21 on plain ones; > 0.3 -> tier fill. Shadowed alpha-tier marks
  had an embossed ghost (edge_err 30 -> 20 on the local clip once filled).
- Synthetic: `synth --blobs N --blob-radius R --exposure A` and overlay suffix `:shadow`; A/B clip `local`
  (shadowed marks, 8 moving discs, +-10 brightness). Eval adds `psnr_ring_10px` and `edge_err_near_mask`
  (gradient-magnitude error near the mark). Ground-truth scores penalise any change to clean pixels, so feather
  always scores a little worse there; judge seams by eye (side-by-side crops of synthetic clips only).
- Local CPU results on a 640x360 `local` clip (in-mask PSNR / ring PSNR / edge err): 0.5.8 20.59 / 31.37 / 20.28;
  0.6.0 LaMa feather 0: 21.92 / 32.47 / 18.89; feather 3: 20.63 / 30.65 / 19.48 (cleanest by eye).
- Known: in that clip a teleporting mark's first position (frames 0-49) was dropped as "follows the same text"
  as a static mark next to it (track.py dedupe). Not fixed. LaMa then copies the leaked glyphs into nearby holes.
- Kaggle A/B 0.6.0 (test kernel v31; in-mask PSNR / edge err; arms 0.5.8 | 0.6.0 Telea | 0.6.0 LaMa):
  local 25.27/12.54 | 23.67/12.89 | 26.27/11.04; synth1 29.26/11.75 | 29.37/10.55 | 31.17/9.44;
  long 31.42/10.60 | 31.32/10.57 | 31.50/10.50; hard (alpha only) identical. Ring PSNR ~0.4 dB lower (feather).
  Telea alone regresses on `local` (stricter borrowing leaves more to Telea); LaMa wins everywhere.
  LaMa costs ~60 ms per crop on a T4 (local fill 9.6 -> 44.6 s; long 20 -> 60 s): ~2 h projected for 072q-1.
- Bug found (since before 0.5.8): the "carry the previous output" check used `ok.all()` over the whole
  region box, never true, so holes were re-inpainted from scratch every frame. 0.6.1: fixed, and a carry must
  pass the borrow tests (offset, mean, tiles); carried holes are re-inpainted (1:3 blend) every 8th frame
  (REFRESH). Local clip: carried 33%, LaMa crops 261 -> 156, quality equal, flicker 10.3 -> 9.7.
  Unchecked carry (96% carried) was frozen under moving objects: ring PSNR -3 dB.
- Kaggle A/B 0.6.1 (v32; in-mask PSNR / edge err / flicker; 0.5.8 in brackets): local 25.82/11.19/8.61
  (25.27/12.54/8.27); synth1 31.18/9.41/8.82 (29.26/11.75/8.76); long 31.54/10.50/8.46 (31.42/10.60/8.38);
  hard identical; same 16.15/31.17 (15.56/38.03). Carried share of holes: local 66%, synth1 77%, long 69%.
- 072q-1 on 0.6.1 (check kernel v9, output 072q-1.wmr.mp4 kept on Kaggle for the user): 2926 s (0.5.8: 1555).
  Analyze 534, remove 2065 (fill 2011: LaMa 21,032 crops 1329 s = 63 ms each, borrow 487), verify 313; leaks
  13 of 1884 (12). Borrowed 57% of filled area (81% on 0.5.8: tile/consensus checks reject more sources in real
  footage), carried 32% of the rest. GPU0 peak 7.8 GB in remove (tight for an 8 GB card); GPU1 idle in remove.
  Track 2 stays alpha (opacity 0.59, conf 0.54); its two_tone score is in tracks.json on Kaggle, not fetched.
- Next levers for the LaMa cost: fp16 at power-of-two padded sizes (cuFFT half needs them), LaMa on GPU1
  (idle in remove), batching a frame's crops, a higher LAMA_MIN_PX or REFRESH. Needs the user's visual review
  of 072q-1.wmr.mp4 first (quality vs 0.5.8 on real footage is not measurable here).

## Package layout (`wmr/`, about 2,000 lines of Python)

| File | Role |
| --- | --- |
| `cli.py` | `wmr doctor / analyze / preview / remove / verify / run / batch / synth / eval / setup-propainter` |
| `backend.py` | torch device, ORT providers (`ort.preload_dlls()` after importing torch), visible GPUs |
| `vio.py` | PyAV decode (keeps pts), libx264 encode (CRF 21, maxrate about 1.15x source), audio stream copy |
| `detect.py` | RapidOCR (PP-OCRv6 ONNX); `match_text` fuzzy matching; GPU by rebuilding ORT sessions (`_move_sessions`: fp16 models, rec shape buckets); `OCRPool` = one engine per GPU, 2 threads each |
| `track.py` | Analyze; details in the Analyze section below |
| `alpha.py` | Opacity and colour from the temporal std dip plus mean; opacity a0 bounded by channel range |
| `remove.py` | One decode/encode pass. Tier `alpha`: inverts the blend (a>0.85 is sent to fill). Tier `fill`: borrows pixels from frames up to ±128 within the same shot, using global similarity motion (LK + RANSAC) and a photometric check, then Telea inpaint blended with the previous output for stability |
| `propainter.py` | Optional: clones ProPainter (NTU S-Lab licence, non-commercial); crops jobs to multiples of 8, at least 128 px; spreads jobs over GPUs; blends overlaps |
| `batch.py` | One subprocess per GPU (`CUDA_VISIBLE_DEVICES`) |
| `synth.py` / `evalm.py` | Synthetic ground truth; metrics (PSNR in/out of mask, flicker, mask recall); OCR leak check |

### Analyze (`track.py`)

1. **Pass A:** OCR every `fps/sample_fps` frames, list match or discovery, scene cuts.
2. **Grouping:** detections are grouped by box size, giving a gradient template per track.
3. **Pass B:** template matching on every frame.
4. **Clean-up:**
   - remove duplicates (`_dedupe`);
   - bridge gaps;
   - hold occluded tracks;
   - classify the motion: static, teleport, linear or bounce;
   - extrapolate at frame edges and coast through unmatched frames with a bounce model;
   - estimate opacity.
5. **Output:** `tracks.json` plus a `tracks.d/` folder holding the mask, alpha and template PNGs.

Key thresholds:

- `T_HI=0.5`, `T_LO=0.35`, `T_ACC=0.4`;
- opaque if a0 ≥ 0.7;
- static if at least 90% of positions are within 4 px of the mode;
- discovery accepts text seen across a scene cut, or for `--min-seconds` (8).

## Test results (v0.2.1)

Synthetic cases, 640x360 and 150 frames each. PSNR inside the mask, before → after; the codec ceiling is about 37 dB.

| Case | Before → after (dB) |
| --- | --- |
| Static, opacity 0.45 | 17.9 → 32.6 |
| Static, opacity 0.25 | 22.9 → 32.5 |
| Bounce, opacity 0.5 | 17.4 → 32.2 |
| Teleport, opaque | 11.4 → 35.1 |
| Linear, opaque | 11.0 → 35.0 |
| Static opaque, panning shot | 11.4 → 35.0 |
| Static opaque, locked-off shot | 11.2 → 20.9 (known weak case; ProPainter is meant for it) |

Kaggle self-test clip (1280x720, 300 frames, 3 overlays): 11.9 → 29.7 dB, 0 frames with mask recall below 95%, 0 OCR leaks out of 20 checked frames.

How to run the checks:

- Smoke test: `python tests/test_smoke.py`. It takes about a minute on CPU.
- Ad-hoc cases: use `wmr synth` with `--overlay TEXT:MOTION:OPACITY[:SIZE]`, then `wmr run` and `wmr eval`.

## Kaggle environment, as observed on 2026-11-10

- Python 3.13, torch 2.11.0+cu130, onnxruntime 1.31.1 (the GPU build was installed by `setup_kaggle.sh`), PyAV 19, OpenCV 4.14.
- Inputs are mounted at `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>`. The notebook therefore searches with `os.walk`.
- Without the GPU accelerator, the session gets torch `+cpu` and ORT fails to load CUDA 13 libraries. That is harmless: OCR falls back to CPU.

## Session 2026-11-17, later (0.6.2: residue from inverted moving marks; LaMa 2-4x faster)

User screenshots of 072q-1 (0.6.1) showed letter-shaped residue: outlines on skin, blocky bright partial letters
on a white shirt. Track 2 was the only alpha-tier track (bounce, opacity 0.59, conf 0.54); the other three were
filled. On the synthetic `same` clip (4x 'Huohua Studio', moving, 0.5-0.65) alpha inversion left dark/inverted
ghost text; `--force-fill` removed it (local CPU, Telea: edge err 33.5 -> 23.1, in-mask PSNR 16.2 -> 18.9).

- Tier rule (track.py): alpha only for **static** marks that are not opaque/two-tone. A moving mark uncovers its
  background in nearby frames, so borrowing gets real pixels; inverting it leaves ghosts wherever the opacity
  map is a pixel or a few percent off (sub-pixel motion, codec, a blurrier estimate on a low-confidence track).
- Verify residue score (evalm._residue): per track, the gradient template matched (TM_CCOEFF_NORMED, +-2 px)
  on the output at the track position, on the OCR-sampled frames. Marks score ~0.85-0.91 before removal; ghosts
  0.48-0.62 (same clip, 0.6.1); clean ~0.0. Logged per track as median / p90 / share >= 0.35 (track.T_LO), and
  in report.json. It reads text logs only, so real-video residue is measurable without pulling video.
- LaMa speed (kaggle/bench, T4): `.half()` gives garbage; autocast fp16 works only at power-of-two padded sizes
  (cuFFT half) and overflows above 256x512 area (256x1024, 512x512: 16/40 crops non-finite; 512x1024: 32/40),
  always as non-finite output. lama.py: autocast fp16 when the pow2 pad is <= 256x512, an isfinite check redoes
  the crop in fp32 (logged); fp32 pads to x8 instead of x64 (1.2-1.4x). fill_frame collects a frame's LaMa crops
  and `Lama.many` runs them one per GPU in threads (all visible GPUs; 2x on T4 x2). No gain: cudnn.benchmark,
  channels_last, jit.freeze; batch4 +15% (skipped). Stream of 150 varying shapes: 73 ms/crop (0.6.1) -> 40 fp16;
  TorchScript profiling compiles cost a few seconds per model at start (not per shape).
- Kaggle A/B 0.6.2 vs 0.6.1 (test kernel v33; in-mask PSNR / edge err / flicker): same 20.02/20.51/18.27
  (16.15/31.17/24.23); hard 32.80/9.35/8.50 (30.65/11.16/8.81); long 33.00/9.08/7.28 (31.54/10.50/8.46);
  local and synth1 equal. Residue ~0 on every track of every clip. Remove wall time was 5-23 s longer on these
  30-110 s clips: second GPU model load + JIT warm-up, and more fill work where tracks left the alpha tier.
- 072q-1 on 0.6.2 (check kernel v10; output 072q-1.wmr.mp4 on Kaggle for the user): 2378 s (0.6.1: 2926,
  0.5.8: 1555). Analyze 515, remove 1520 (fill 1474: borrow 503, inpaint 777 for 15,338 LaMa crops ~50 ms each
  incl. CPU side; 14,652 fp16, 56 non-finite redone in fp32), verify 326. All 4 tracks filled (track 2 left
  alpha). Leaks 5 of 1884 (13). Residue median 0.00-0.01, p90 <= 0.12, 0% >= 0.35 on every track. Borrowed 65%
  of filled area, carried 45% of the rest. GPU1 util only 10% in remove: most frames need one LaMa crop, so the
  second GPU rarely has work; GPU0 peak 8.0 GB (OCR engines stay loaded for verify).
  Next speed levers: overlap a frame's LaMa with the next frame's borrowing (carry depends on the previous
  output, so per-track ordering must hold), or free the OCR engines' GPU memory during remove.
- `same` clip still has mask recall ~0.62 (identical fast marks: tracks miss instances; 28 of 30 frames leak).
  Tracking problem on that synthetic clip, not seen on 072q-1 (13 of 1884).

## Session 2026-11-18 (0.6.3 tried: verify during remove, LaMa overlapped; reverted to 0.6.2)

Reverted on Tom's call: -75 s on 072q-1 did not pay for the extra code. The diff is kept locally at
`kaggle/out/wmr-0.6.3.patch` (gitignored). Only the test notebook's `{VERSION}` path fix stayed.

- Verify runs on the output frames as remove() produces them (evalm.LiveVerify: a thread feeds sampled frames
  to OCRPool.imap; `remove(on_frame=...)`): no second decode. Frames are checked before encoding.
- Fill split into fill_start (regions, feather weights, borrowing: needs no earlier output), fill_end (carry
  check against the previous output, Telea, LaMa crops submitted with Lama.submit) and fill_result (wait,
  blend). remove() holds a frame whose LaMa runs and completes it after the next frame's fill_start. LaMa
  devices are ordered with the fill GPU last, so single crops go to GPU1. Output on CPU is bit-identical to
  0.6.2 on the same tracks (small clip, LaMa on CPU, x264). Note: 0.6.2 itself is not run-to-run identical
  (its `run` vs `remove` on the same tracks differed in frames 73-85: a carry decision flipping on float noise).
- 072q-1 (check kernel v11): 2378 -> 2303 s only. Verify 326 -> 1 s, but remove 1520 -> 1747: borrow 503 ->
  788 s and alpha+motion 462 -> 619 s (the OCR threads, the LaMa thread and GPU0 crops now compete with the
  main loop on Kaggle's 4 vCPU and GPU0), and the main loop still waits 558 s for LaMa (overlap is one frame's
  borrowing, ~28 ms, against ~45-60 ms of LaMa per crop; carry needs the previous output complete).
  Leaks 5 of 1884, residue median <= 0.02 on all tracks (same as 0.6.2). GPU util in remove 35% / 38%.
- The remove loop is CPU/GIL-bound, not GPU-bound. Bigger levers: split the video at a scene cut near the
  middle (borrowing and carry never cross a cut) and run two remove processes, one per GPU, then concatenate
  (needs a GOP start at the split; 4 vCPU limit the gain); fewer LaMa crops (REFRESH, LAMA_MIN_PX); verify OCR
  on one engine with one thread to cut CPU contention.
- Test kernel v34 failed before testing: its setup cell had wmr-0.6.1 hard-coded (fixed: uses {VERSION}).
  Synthetic A/B for 0.6.3 not rerun.

## Session 2026-11-18, later (RunPod prep, still 0.6.2)

- Tom signed up for RunPod. Kaggle's 4 vCPU for 2 T4s is the bottleneck; RunPod cards list vCPU per GPU
  (RTX A4000: 18 vCPU, $0.25/hr; RTX 3090: 32 vCPU, $0.50/hr). "N max" on a card = most GPUs one pod
  (one machine) can have, not a limit on how many pods run at once.
- A pod shows every host core in `os.cpu_count()` but rents a share (RunPod docs: libraries that size
  pools from the visible count get throttled). `backend.cpu_quota()` reads cgroup v2 `cpu.max` / v1
  cfs quota; `cli.main` sets `WMR_THREADS` from it when it is below the visible count, and `WMR_THREADS`
  now also sets `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` (torch reads them at
  import). `wmr doctor` prints `cpu_count`, `cpus_usable`, `WMR_THREADS`. `tests/test_cpu_quota.py`.
  No change on the Mac or wherever the quota equals the visible count. macOS OpenCV (GCD) ignores
  `setNumThreads` anyway.
- `scripts/runpod.sh` (+ `requirements/runpod.txt`): install on an official PyTorch template, then one
  profiled `wmr run` with `gpu.txt` (epoch, per GPU util / MiB every 2 s, joins prof.txt epochs). README
  "Running on a RunPod GPU pod" has the steps. Not yet run on a pod: onnxruntime-gpu vs the image's CUDA
  is the likely snag (doctor's `ort_providers` shows it; OCR falls back to CPU otherwise).
- First test: 072q-1 only (`"Huohua Studio"`), same boundary as Kaggle; fetch only the .txt files.

## Known issues and ideas, in rough priority

2. Do not run removal on `yxyxtx/video-test-set-1` (the `BJ` videos), `yxyxtx/test-dataset-2`, or v12 / v27 of `huohua-studio-test-data`: they carry third-party marks. Synthetic clips and 072q-1 only.
3. Static opaque text on a locked-off shot: test the real ProPainter engine on T4.
4. Speed: tracking, fill and encode are CPU-bound. A 720p, 15 s clip with 3 overlays took about 68 s on 2 CPU cores. Ideas:
   - NVENC for previews;
   - fewer fill offsets;
   - running pass B on a downscaled grey frame.
5. Outlined, shadowed or gradient text goes to `fill` (opacity is estimated as one colour; 0.6.0 detects two-tone marks). Since 0.6.2 moving marks are always filled; static semi-transparent marks are still inverted (no residue check routes a bad inversion to fill yet; the verify residue score shows it).
6. Output is always 8-bit H.264; rotation metadata is not applied.
7. An earlier, unrelated `wmr 0.1.0` build exists in the old cloud workspace (`/home/claude/wmr/src`). Ignore it.

## User preferences seen in this project

- Encode for "no visible change" at about the source size, not maximum fidelity.
- Avoid physically clipping videos. Handle all overlays in one pass with one re-encode.
- Minimise manual relaying; prefer direct, automated loops.
