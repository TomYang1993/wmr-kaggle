"""Where a run spends its time, per phase and per thread. Enabled with WMR_PROF=report.txt.
A sampler thread reads every thread's Python stack each 10 ms and charges the wall time to the
innermost wmr line (C calls in cv2 / numpy / ORT / PyAV show as the wmr line that made them).
On Linux it also reads each native thread's CPU time (/proc/self/task) at every phase change, so
threads Python never sees (ORT, OpenCV, FFmpeg pools) are counted too.
    from wmr import prof; prof.phase("remove")"""
import atexit
import collections
import linecache
import os
import sys
import threading
import time

_on = bool(os.environ.get("WMR_PROF"))
_cur = "start"
_wall = collections.defaultdict(float)            # (phase, thread, where) -> seconds
_spans = []                                         # (phase, t0 epoch, t1 epoch, process cpu, {thread: cpu})
_t0, _c0, _task0 = time.time(), None, None
_PKG = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _tasks():
    """native thread name -> CPU seconds (user + system), summed over threads with that name."""
    out = collections.Counter()
    try:
        tick = os.sysconf("SC_CLK_TCK")
        for tid in os.listdir("/proc/self/task"):
            with open(f"/proc/self/task/{tid}/stat") as fh:
                s = fh.read()
            name = s[s.index("(") + 1:s.rindex(")")]
            f = s[s.rindex(")") + 2:].split()
            out[name] += (int(f[11]) + int(f[12])) / tick
    except (OSError, AttributeError):   # no /proc or os.sysconf (Windows, macOS): wall samples only
        pass
    return out


def phase(name):
    """Close the current phase and start `name`."""
    global _cur, _t0, _c0, _task0
    if not _on:
        return
    now, cpu, task = time.time(), sum(os.times()[:2]), _tasks()
    _spans.append((_cur, _t0, now, cpu - _c0, task - _task0))
    _cur, _t0, _c0, _task0 = name, now, cpu, task


def _where(frame):
    leaf, ours = frame, None
    while frame is not None:
        if frame.f_code.co_filename.startswith(_PKG) and not frame.f_code.co_filename.endswith("prof.py"):
            ours = frame
            break
        frame = frame.f_back
    f = ours or leaf
    src = linecache.getline(f.f_code.co_filename, f.f_lineno).strip()[:70]
    w = f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno} {f.f_code.co_name}  | {src}"
    if ours is not None and leaf is not ours:
        w += f"  -> {leaf.f_code.co_name} ({os.path.basename(leaf.f_code.co_filename)})"
    return w


def _sample():
    me, last = threading.get_ident(), time.perf_counter()
    while True:
        time.sleep(0.01)
        now = time.perf_counter()
        dt, last = now - last, now
        names = {t.ident: t.name for t in threading.enumerate()}
        for ident, fr in sys._current_frames().items():
            if ident != me:
                _wall[(_cur, names.get(ident, str(ident)), _where(fr))] += dt


def _report():
    phase("end")
    with open(os.environ["WMR_PROF"], "w") as fh:
        start = _spans[0][1]
        for ph, a, b, cpu, task in _spans:
            if ph == "end" or b - a < 0.1:
                continue
            fh.write(f"\n##### {ph}: {b - a:.1f} s wall (t={a - start:.0f}..{b - start:.0f} s, epoch {a:.0f}..{b:.0f}),"
                     f" process CPU {cpu:.1f} s = {cpu / max(b - a, 1e-9):.2f} cores\n")
            fh.write("  native threads by CPU s: " + ", ".join(
                f"{k} {v:.1f}" for k, v in task.most_common() if v >= 0.5) + "\n")
            by_thread = collections.defaultdict(list)
            for (p, th, w), s in _wall.items():
                if p == ph:
                    by_thread[th].append((s, w))
            for th, rows in sorted(by_thread.items(), key=lambda kv: -sum(s for s, _ in kv[1])):
                rows.sort(reverse=True)
                fh.write(f"  -- thread {th}: {sum(s for s, _ in rows):.1f} s sampled\n")
                for s, w in rows[:14]:
                    if s >= 0.5:
                        fh.write(f"     {s:8.1f}  {w}\n")


if _on:
    _c0, _task0 = sum(os.times()[:2]), _tasks()
    threading.Thread(target=_sample, daemon=True, name="prof").start()
    atexit.register(_report)
