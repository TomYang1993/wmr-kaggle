"""backend.cpu_quota reads the container's CPU share from cgroup v2 or v1 files."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wmr.backend import cpu_quota  # noqa: E402

n = cpu_quota("/nonexistent")   # no cgroup: the cores this process may run on
assert n >= 1


def quota(files):
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(d, name)), exist_ok=True)
            with open(os.path.join(d, name), "w") as fh:
                fh.write(text)
        return cpu_quota(d)


assert quota({"cpu.max": "max 100000\n"}) == n                       # v2, no limit
assert quota({"cpu.max": "100000 100000\n"}) == 1                    # v2, one CPU
assert quota({"cpu.max": "150000 100000\n"}) == min(n, 2)            # v2, 1.5 CPUs rounds up
assert quota({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}) == n      # v1, no limit
assert quota({"cpu/cpu.cfs_quota_us": "100000\n", "cpu/cpu.cfs_period_us": "100000\n"}) == 1  # v1, one CPU
assert quota({"cpu.max": f"{(n + 8) * 100000} 100000\n"}) == n       # a quota above the cores in view
print(f"cpu_quota ok ({n} cores in view)")
