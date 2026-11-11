from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .bench import attach_existing_baseline, run_benchmark, validate_benchmark_record
from .environment import inspect_environment, require_device


def _csv_shapes(value: str) -> list[tuple[int, ...]]:
    result = []
    for item in value.split(","):
        try:
            shape = tuple(int(part) for part in item.lower().split("x"))
        except (ValueError, TypeError) as exc:
            raise argparse.ArgumentTypeError(f"invalid shape {item!r}; expected dimensions separated by x") from exc
        if len(shape) not in (2, 3, 4) or any(n < 1 for n in shape):
            raise argparse.ArgumentTypeError("shape dimensions must be positive")
        result.append(shape)
    if not result:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xpu-kernels")
    commands = parser.add_subparsers(dest="command", required=True)
    env = commands.add_parser("check-env", help="inspect local toolchains and devices")
    env.add_argument("--backend", choices=("all", "gpu", "npu"), default="all")
    build = commands.add_parser("build", help="build the selected native backend")
    build.add_argument("--backend", choices=("gpu", "npu"), required=True)
    build.add_argument("--force", action="store_true")
    build.add_argument("--implementation", choices=("cuda", "triton", "auto", "native", "optimized"))
    example = commands.add_parser("example", help="run one RMSNorm correctness example")
    example.add_argument("--device", choices=("cuda", "npu"), required=True)
    example.add_argument("--implementation", choices=("cuda", "triton", "ascendc", "auto", "native", "optimized"))
    example.add_argument("--shape", type=_csv_shapes)
    test = commands.add_parser("test", help="run golden and backend correctness cases")
    test.add_argument("--device", choices=("cuda", "npu"), required=True)
    bench = commands.add_parser("bench", help="measure RMSNorm implementations over aligned and tail shapes")
    bench.add_argument("--device", choices=("cuda", "npu"), required=True)
    bench.add_argument("--shapes", type=_csv_shapes, default=None)
    bench.add_argument("--batches", "--repeats", dest="batches", type=int, default=20)
    bench.add_argument("--iterations-per-batch", type=int, default=10)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--output", type=Path)
    bench.add_argument("--implementations", help="GPU: cuda,triton,torch; NPU: ascendc")
    bench.add_argument("--dtypes", default="float32,bfloat16,float16")
    bench.add_argument("--native-reference", action="store_true")
    profile = commands.add_parser("profile", help="run an independent profiler workload")
    profile.add_argument("--device", choices=("cuda", "npu"), required=True)
    profile.add_argument("--tool", choices=("msprof", "nsys", "ncu"), required=True)
    profile.add_argument("--output", type=Path, default=Path("results/profiles"))
    profile.add_argument("--shape", type=_csv_shapes)
    profile.add_argument("--implementation", choices=("cuda", "triton", "ascendc", "auto", "native", "optimized"))
    profile.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    for command in (build, example, test, bench, profile):
        command.add_argument("--operator", choices=("rms_norm", "mhc_post_backward"), default="rms_norm")
    return parser


def _root() -> Path:
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "pyproject.toml").is_file():
        return package_root
    configured = os.environ.get("XPU_KERNELS_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file():
        return cwd
    raise RuntimeError("run the command from the xpu-kernels project root or set XPU_KERNELS_PROJECT_ROOT")


def _build(backend: str, force: bool, implementation: str | None = None) -> int:
    if backend == "gpu":
        require_device("cuda")
        if implementation == "triton":
            report = inspect_environment("gpu")
            if not report["triton"]:
                raise RuntimeError("Triton is missing")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        from .backends.cuda import _load_extension
        if force:
            _load_extension.cache_clear()
        module = _load_extension()
        print(f"built and loaded CUDA extension: {module.__file__}")
        return 0
    require_device("npu")
    if force:
        import shutil

        from .backends.npu import _project_root

        native_root = _project_root()
        checkout = (native_root / "pyproject.toml").is_file()
        cache_value = str(native_root / "cache") if checkout else os.environ.get("XPU_KERNELS_CACHE_DIR")
        if not cache_value:
            raise RuntimeError("set XPU_KERNELS_CACHE_DIR before forcing a wheel-installed extension rebuild")
        cache_root = Path(cache_value)
        build_dir = cache_root / "build" / "npu-extension"
        if build_dir.exists():
            shutil.rmtree(build_dir)
    from .backends.npu import _load_extension

    if force:
        _load_extension.cache_clear()
    module = _load_extension()
    print(f"built and loaded native extension: {module.__file__}")
    return 0


def _example(device: str, shape: tuple[int, int], implementation: str | None = None) -> int:
    require_device(device)
    import torch

    from . import rms_norm
    from .golden import rms_norm_golden
    from .bench import _tolerances

    dev = torch.device(device)
    rows, width = shape
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.randn(rows, width, device=dev, dtype=dtype)
        weight = torch.randn(width, device=dev, dtype=dtype)
        actual = rms_norm(x, weight, implementation=implementation)
        expected = rms_norm_golden(x, weight)
        _synchronize(device)
        rtol, atol = _tolerances(torch, dtype)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        print(f"PASS device={device} dtype={dtype} shape={tuple(x.shape)} max_abs={(actual.float()-expected.float()).abs().max().item():.6g}")
    return 0


def _synchronize(device: str) -> None:
    import torch

    (torch.cuda if device == "cuda" else torch.npu).synchronize()


def _test(device: str) -> int:
    require_device(device)
    root = _root()
    test_file = root / "tests" / "test_rms_norm.py"
    if not test_file.is_file():
        raise RuntimeError(f"device correctness test file is missing: {test_file}")
    env = os.environ.copy()
    env["XPU_KERNELS_TEST_DEVICE"] = device
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    if completed.returncode:
        return completed.returncode
    _validate_test_report(completed.stdout, device)
    print(f"Correctness suite passed with {device} available.")
    return 0


def _validate_test_report(output: str, device: str) -> None:
    match = re.search(r"Ran (\d+) tests? in", output)
    target_class = "GpuKernelTests" if device == "cuda" else "NpuKernelTests"
    if not match or int(match.group(1)) < 1:
        raise RuntimeError("unittest discovery ran zero tests")
    lines = output.splitlines()
    start = next((i for i, line in enumerate(lines) if target_class in line and "test_multishape_multidtype_against_golden" in line), None)
    if start is None:
        raise RuntimeError(f"requested {device} correctness case was not discovered")
    block = [lines[start].strip()]
    if block[0].endswith("... ok"):
        return
    for line in lines[start + 1:]:
        stripped = line.strip()
        if re.match(r"^test_[A-Za-z0-9_]+ \(test_", stripped):
            break
        if stripped.startswith("Ran ") or stripped.startswith("===") or stripped.startswith("---"):
            break
        block.append(stripped)
        if stripped == "ok" or stripped.endswith("... ok"):
            return
        if stripped == "FAIL" or stripped == "ERROR" or "skipped" in stripped.lower():
            break
    raise RuntimeError(f"requested {device} correctness case did not finish successfully: {' | '.join(block[-6:])}")


def _profile_command(tool: str, binary: str, output: Path, workload: list[str], msprof_help: str = "", implementation: str = "triton") -> list[str]:
    if tool == "msprof":
        if "--output" not in msprof_help or "--application" not in msprof_help:
            raise RuntimeError("msprof --help did not advertise both --output and --application")
        return [binary, f"--output={output}", f"--application={shlex.join(workload)}"]
    if tool == "nsys":
        return [binary, "profile", "--output", str(output / "nsys"), *workload]
    if tool == "ncu":
        return [
            binary,
            "--target-processes", "all",
            "--launch-count", "1",
            "--kernel-name", "regex:_rms_norm_kernel" if implementation == "triton" else "regex:.*rms_norm.*",
            "--replay-mode", "kernel",
            "--cache-control", "none",
            "--clock-control", "none",
            "--section", "SpeedOfLight",
            "--section", "LaunchStats",
            "--force-overwrite", "--export", str(output / "rmsnorm"),
            *workload,
        ]
    raise ValueError(f"unsupported profiler {tool!r}")


def _profile_subprocess_env() -> dict[str, str]:
    """Expose this interpreter's import paths to profiler-launched Python apps."""
    env = os.environ.copy()
    entries = [entry for entry in sys.path if entry]
    entries.extend(entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry and entry not in entries)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _bench(args) -> int:
    require_device(args.device)
    from .suites import RMSNORM_SHAPES
    shapes = args.shapes or list(RMSNORM_SHAPES)
    implementations = tuple(args.implementations.split(",")) if args.implementations else None
    rows = run_benchmark(args.device, shapes, args.warmup, args.batches, args.iterations_per_batch,
                         dtypes=tuple(args.dtypes.split(",")), implementations=implementations,
                         include_native_baseline=args.native_reference)
    rows = attach_existing_baseline(rows, _root() / "results" / "benchmarks")
    for row in rows:
        validate_benchmark_record(row)
    rendered = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    output = args.output
    if output is None:
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("results") / "benchmarks" / f"{args.device}-{stamp}.jsonl"
    if not output.is_absolute():
        output = _root() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"saved={output}")
    print(rendered)
    return 0


def _profile(args) -> int:
    if (args.device, args.tool) not in (("npu", "msprof"), ("cuda", "nsys"), ("cuda", "ncu")):
        raise RuntimeError(f"{args.tool} does not profile the {args.device} backend")
    require_device(args.device)
    binary = shutil.which(args.tool)
    if binary is None and args.tool == "msprof":
        ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
        if ascend_home:
            candidate = Path(ascend_home) / "bin" / "msprof"
            if candidate.is_file():
                binary = str(candidate)
    if binary is None:
        raise RuntimeError(f"profiler executable {args.tool!r} was not found")
    from .profile_workload import check_profile_case

    implementation = args.implementation or ("cuda" if args.device == "cuda" else "ascendc")
    check_profile_case(args.device, args.shape[0], args.dtype, implementation)
    output = args.output if args.output.is_absolute() else _root() / args.output
    output.mkdir(parents=True, exist_ok=True)
    workload = [
        sys.executable, "-m", "xpu_kernels.profile_workload",
        "--device", args.device,
        "--shape", f"{args.shape[0][0]}x{args.shape[0][1]}",
        "--dtype", args.dtype, "--implementation", implementation,
    ]
    if args.tool == "msprof":
        help_result = subprocess.run([binary, "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        command = _profile_command(args.tool, binary, output, workload, help_result.stdout, implementation)
    else:
        command = _profile_command(args.tool, binary, output, workload, implementation=implementation)
    print("PROFILE_COMMAND=" + json.dumps(command))
    return subprocess.run(command, cwd=_root(), env=_profile_subprocess_env()).returncode


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if getattr(args, "operator", None) == "mhc_post_backward":
            from .mhc_workload import run_cli
            return run_cli(args)
        if getattr(args,"implementation",None) in ("auto","native","optimized"):
            raise ValueError("auto/native/optimized implementations apply to mhc_post_backward")
        if args.command in ("example", "profile") and args.shape is None:
            args.shape = [(4, 4096)] if args.command == "example" else [(16, 4096)]
        if args.command == "check-env":
            report = inspect_environment(args.backend)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            missing = report.get("missing", [])
            return 1 if missing else 0
        if args.command == "build":
            return _build(args.backend, args.force, args.implementation)
        if args.command == "example":
            return _example(args.device, args.shape[0], args.implementation)
        if args.command == "test":
            return _test(args.device)
        if args.command == "bench":
            if args.batches < 1 or args.iterations_per_batch < 1 or args.warmup < 0:
                raise RuntimeError("--batches and --iterations-per-batch must be positive; --warmup must be non-negative")
            return _bench(args)
        if args.command == "profile":
            return _profile(args)
    except Exception as exc:
        print(f"xpu-kernels: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
