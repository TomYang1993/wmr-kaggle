#!/usr/bin/env bash
# Download the test kernel's output and log:  kaggle/fetch.sh [out dir] [kernel] [filename regex]
# The kaggle client advertises brotli but fails to decode Kaggle's br responses, so brotli is hidden.
set -euo pipefail
OUT=${1:-kaggle/out/latest}
PAT=${3:-}   # optional filename regex, e.g. "\\.txt$" to skip videos
K=${2:-yxyxtx/fable-5-2-watermark-detection-test}
PY=$(head -1 "$(command -v kaggle)" | sed 's/^#!//')
mkdir -p "$OUT"
"$PY" -c "
import sys
sys.modules['brotli'] = None; sys.modules['brotlicffi'] = None
from kaggle.cli import main
sys.argv = ['kaggle', 'kernels', 'output', sys.argv[1], '-p', sys.argv[2]] + (['--file-pattern', sys.argv[3]] if sys.argv[3] else [])
main()" "$K" "$OUT" "$PAT"
