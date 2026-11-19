#!/usr/bin/env bash
# RunPod GPU pod from an official PyTorch template (torch + CUDA preinstalled), wmr unpacked on the pod.
#   bash wmr/scripts/runpod.sh                       install (heavy part once per pod), print `wmr doctor`
#   bash wmr/scripts/runpod.sh VIDEO "TEXT" [OUT]    then one profiled `wmr run` on VIDEO, whose mark reads TEXT
# OUT (default /workspace/out) gets the cleaned video plus the text evidence: results.txt (doctor, log,
# wall time), prof.txt (WMR_PROF stage profile) and gpu.txt (epoch seconds, then per GPU
# index, util %, MiB used, every 2 s; matches the epochs in prof.txt).
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# big-lama (196 MB) goes to the torch hub cache: keep it on /workspace (the pod's volume or a network
# volume), so a restarted pod skips the download
[ -d /workspace ] && export TORCH_HOME="${TORCH_HOME:-/workspace/cache/torch}"
if ! python -c "import rapidocr, onnxruntime, av, scipy" 2>/dev/null; then
  pip install -q -r "$HERE/requirements/runpod.txt"
  pip uninstall -y -q onnxruntime 2>/dev/null || true   # a CPU build would shadow onnxruntime-gpu
  pip install -q onnxruntime-gpu
fi
pip install -q --no-deps "$HERE"   # every time: picks up a newly uploaded wmr
if [ -z "$1" ]; then python -m wmr doctor; exit; fi

VIDEO=$1; OUT=${3:-/workspace/out}; B=$(basename "$VIDEO")
mkdir -p "$OUT"
printf '%s\n' "$2" > "$OUT/texts.txt"
while sleep 2; do
  echo "$(date +%s) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ';')"
done > "$OUT/gpu.txt" &
trap "kill $!" EXIT
{ python -m wmr doctor
  T0=$SECONDS
  WMR_PROF="$OUT/prof.txt" python -m wmr run "$VIDEO" -o "$OUT/${B%%.*}.wmr.mp4" --texts "$OUT/texts.txt"
  echo "=== $B: $((SECONDS - T0)) s wall"; } 2>&1 | tee "$OUT/results.txt"
