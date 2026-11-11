"""Capture and extract matched CANN/native and optimized mHC kernel samples."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xpu_kernels.mhc_workload import SHAPES


def extract(profile_root, checks_path, output):
    checks = {}
    for line in checks_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (tuple(row["shape"]), row["implementation"])
        if key in checks or row["correctness"]["status"] != "PASS":
            raise ValueError("duplicate or failed correctness record")
        checks[key] = row
    records = []
    data = ROOT / "docs/performance/data"
    data.mkdir(parents=True, exist_ok=True)
    for impl in ("native", "optimized"):
        paths = list((profile_root / impl).rglob("op_summary*.csv"))
        # Also accept the exact CSVs transferred from a remote project.
        if not paths and (profile_root / f"{impl}.csv").is_file():
            paths = [profile_root / f"{impl}.csv"]
        if len(paths) != 1:
            raise ValueError(f"expected one fresh {impl} op_summary CSV, found {len(paths)}")
        with paths[0].open(encoding="utf-8-sig", newline="") as source:
            rows = [(line, row) for line, row in enumerate(csv.DictReader(source), 2)
                    if row["Task Type"] == "MIX_AIC" and
                    ((impl == "native" and row["Op Name"].startswith("aclnnMhcPostBackward_")) or
                     (impl == "optimized" and row["Op Name"] == "xpu_mhc_post_backward_bf16"))]
        rows.sort(key=lambda pair: float(pair[1]["Task Start Time(us)"]))
        if len(rows) != len(SHAPES) * 25:
            raise ValueError(f"{impl}: expected {len(SHAPES)*25} target launches, got {len(rows)}")
        exported = []
        for index, shape in enumerate(SHAPES):
            evidence = checks[(shape, impl)]
            if evidence["dtype"] != "bfloat16":
                raise ValueError("profile matrix requires BF16 correctness evidence")
            chunk = rows[index*25:(index+1)*25]
            durations = [float(row["Task Duration(us)"]) for _, row in chunk]
            if any(not math.isfinite(t) or t <= 0 for t in durations):
                raise ValueError("invalid profiler duration")
            if impl == "native":
                expected = ",".join(map(str, shape))
                if any(row["Input Shapes"].strip('"').split(";")[0] != expected for _, row in chunk):
                    raise ValueError("native launch order disagrees with recorded matrix")
            for iteration, (line, row) in enumerate(chunk):
                exported.append({"shape": "x".join(map(str, shape)), "iteration": iteration,
                    "sample": "warmup" if iteration < 20 else "measured", "source_csv_line": line,
                    **{key: row[key] for key in ("Op Name", "Task Type", "Task ID", "Stream ID",
                        "Task Start Time(us)", "Task Duration(us)", "Input Shapes")}})
            records.append({"schema_version": 1, "operator": "mhc_post_backward",
                "implementation": impl, "hardware": evidence["hardware"], "device": "npu",
                "shape": list(shape), "dtype": "bfloat16", "h_res": True,
                "measurement_method": "msprof Task Duration(us)", "task_type": "MIX_AIC",
                "cache_policy": "same-buffer steady state", "clock_control": "unmodified",
                "warmup_iterations": 20, "samples_us": durations[20:],
                "latency_us_median": statistics.median(durations[20:]),
                "source_csv": f"docs/performance/data/mhc-mainline-{impl}.csv",
                "native_api": "torch_npu.npu_mhc_post_backward -> aclnnMhcPostBackward",
                "correctness": evidence["correctness"], "torch": evidence["torch"],
                "runtime": evidence["runtime"]})
        with (data / f"mhc-mainline-{impl}.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(exported[0]))
            writer.writeheader()
            writer.writerows(exported)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in records), encoding="utf-8")
    print(f"Saved {len(records)} verified profiler rows and {len(rows)*2} raw kernel launches")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-profile", type=Path, help="extract an existing fresh capture without using a device")
    p.add_argument("--output", type=Path, default=ROOT / "results/profiles/mhc-matrix")
    p.add_argument("--checks", type=Path, default=ROOT / "results/benchmarks/mhc-post-backward-npu-mainline.jsonl")
    p.add_argument("--records", type=Path, default=ROOT / "results/benchmarks/mhc-post-backward-msprof.jsonl")
    args = p.parse_args()
    if args.from_profile is None:
        if os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
            p.error("disable private custom-op overlays for the CANN mainline comparison")
        from xpu_kernels.cli import _profile_command, _profile_subprocess_env
        from xpu_kernels.environment import require_device
        require_device("npu")
        binary = shutil.which("msprof")
        if not binary:
            p.error("msprof was not found; manually source the selected CANN environment")
        args.output.mkdir(parents=True, exist_ok=False)
        help_text = subprocess.run([binary, "--help"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        for impl in ("native", "optimized"):
            out = args.output / impl
            out.mkdir()
            workload = [sys.executable, "-B", "-u", "-m", "xpu_kernels.mhc_workload",
                "--matrix", "--implementation", impl, "--profile-warmup", "20", "--profile-repeats", "5"]
            command = _profile_command("msprof", binary, out.resolve(), workload, help_text)
            (out / "command.json").write_text(json.dumps(command), encoding="utf-8")
            with (out / "application.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, env=_profile_subprocess_env(),
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=420)
        args.from_profile = args.output
    extract(args.from_profile, args.checks, args.records)


if __name__ == "__main__":
    main()
