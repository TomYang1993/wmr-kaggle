#!/usr/bin/env bash
# Publish the package as a new version of the code dataset, then run the test kernel on T4 x2.
#   kaggle/push.sh "message"        (KERNEL_ONLY=1 skips the dataset upload)
#   KERNEL=kaggle/real kaggle/push.sh   runs the real-video notebook instead of the synthetic test
#   ACC=TpuV5E8 picks another accelerator
#   TIMEOUT=seconds                      run-time limit (default 3000; raise for long batches)
set -euo pipefail
cd "$(dirname "$0")/.."
V=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' wmr/__init__.py)
[ -n "$V" ] || { echo "no version in wmr/__init__.py"; exit 1; }
DS=yxyxtx/watermark-removal-core-code
if [ -z "${KERNEL_ONLY:-}" ]; then
  rm -rf kaggle/build && mkdir -p kaggle/build/dataset kaggle/build/wmr-$V
  cp -r wmr scripts requirements tests pyproject.toml README.md kaggle/build/wmr-$V/
  (cd kaggle/build && zip -qr dataset/wmr-$V.zip wmr-$V -x '*__pycache__*')
  cp kaggle/dataset-metadata.json kaggle/build/dataset/
  cp kaggle/baseline/*.zip kaggle/build/dataset/ 2>/dev/null || true   # older release(s) for A/B timing runs
  kaggle datasets version -p kaggle/build/dataset -m "${1:-wmr $V}"
  # ponytail: fixed poll, kernel would otherwise mount the previous version
  until kaggle datasets status $DS | grep -qi ready; do sleep 15; done
fi
kaggle kernels push -p "${KERNEL:-kaggle/kernel}" --accelerator "${ACC:-NvidiaTeslaT4}" -t "${TIMEOUT:-3000}"
