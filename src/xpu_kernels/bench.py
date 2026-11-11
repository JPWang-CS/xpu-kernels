from __future__ import annotations

import json
import os
import platform
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


def _cann_version() -> str | None:
    configured = os.environ.get("CANN_VERSION")
    if configured:
        return configured
    root = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not root:
        return None
    for filename in (
        "version.info", "version.cfg",
        "aarch64-linux/ascend_toolkit_install.info",
        "x86_64-linux/ascend_toolkit_install.info",
        "opp/version.info",
    ):
        path = Path(root) / filename
        try:
            contents = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"(?im)^\s*(?:Version|CANN_VERSION)\s*[:=]\s*([^\s;]+)", contents)
        if match:
            build = re.search(r"(?im)^\s*innerversion\s*=\s*([^\s;]+)", contents)
            return match.group(1) + (f" ({build.group(1)})" if build else "")
    return None


def _event_samples(call, device_module, stream, warmup: int, batches: int, iterations_per_batch: int) -> list[float]:
    """Measure elapsed device-stream intervals; exclude host wall-clock timing."""
    for _ in range(warmup):
        call()
    stream.synchronize()
    samples = []
    for _ in range(batches):
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iterations_per_batch):
            call()
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations_per_batch)
    return samples


def _native_baseline(device: str, torch, x, weight, eps: float):
    width = x.shape[-1]
    if device == "cuda":
        native = getattr(torch.nn.functional, "rms_norm", None)
        if native is None:
            return None, {"status": "unavailable", "reason": "torch.nn.functional.rms_norm is absent"}

        def call():
            return native(x, (width,), weight, eps)

        return call, {
            "status": "available",
            "name": "torch.nn.functional.rms_norm",
            "semantics": "PyTorch native RMSNorm; accumulation and cast order may differ from the Pangu golden.",
            "allocates_output": True,
        }

    import torch_npu

    native = getattr(torch_npu, "npu_rms_norm", None)
    if native is None:
        native = getattr(getattr(torch.ops, "npu", None), "npu_rms_norm", None)
    if native is None:
        return None, {"status": "unavailable", "reason": "torch_npu.npu_rms_norm is absent"}

    def call():
        result = native(x, weight, epsilon=eps)
        return result[0] if isinstance(result, (tuple, list)) else result

    return call, {
        "status": "available",
        "name": "torch_npu.npu_rms_norm",
        "semantics": "torch_npu native RMSNorm; accumulation and cast order may differ from the Pangu golden.",
        "allocates_output": True,
    }


def run_benchmark(
    device: str,
    shapes: list[tuple[int, int]],
    warmup: int = 5,
    batches: int = 20,
    iterations_per_batch: int = 10,
    dtypes: tuple[str, ...] | None = None,
    check_correctness: bool = True,
    include_native_baseline: bool = True,
) -> list[dict]:
    import torch

    from . import rms_norm
    from .golden import rms_norm_golden

    if batches < 1 or iterations_per_batch < 1 or warmup < 0:
        raise ValueError("batches and iterations_per_batch must be positive; warmup must be non-negative")
    device_module = torch.cuda if device == "cuda" else torch.npu
    dev = torch.device(device)
    device_index = device_module.current_device()
    if device == "cuda":
        hardware = torch.cuda.get_device_name(device_index)
        runtime = {
            "cuda": torch.version.cuda,
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "triton": __import__("triton").__version__,
        }
        measurement_method = "torch.cuda.Event.elapsed_time"
    else:
        import torch_npu

        hardware = torch.npu.get_device_name(device_index)
        runtime = {
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "cann_version": _cann_version(),
        }
        measurement_method = "torch.npu.Event.elapsed_time"

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    selected_dtypes = tuple(dtype_map) if dtypes is None else dtypes
    if any(name not in dtype_map for name in selected_dtypes):
        raise ValueError(f"unsupported benchmark dtype selection: {selected_dtypes}")
    rows = []
    context = device_module.device(device_index)
    with context:
        stream = device_module.current_stream(device_index)
        for row_count, width in shapes:
            for dtype_name in selected_dtypes:
                dtype = dtype_map[dtype_name]
                x = torch.randn(row_count, width, device=dev, dtype=dtype)
                weight = torch.randn(width, device=dev, dtype=dtype)
                out = torch.empty_like(x)
                golden_out = torch.empty_like(x)
                op = lambda: rms_norm(x, weight, out=out)
                reference = lambda: rms_norm_golden(x, weight, out=golden_out)

                if check_correctness:
                    actual = op()
                    expected = reference()
                    stream.synchronize()
                    rtol, atol = _tolerances(torch, dtype)
                    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

                kernel_samples = _event_samples(op, device_module, stream, warmup, batches, iterations_per_batch)
                if include_native_baseline:
                    native_call, native_info = _native_baseline(device, torch, x, weight, 1e-5)
                else:
                    native_call, native_info = None, {"status": "not_requested"}
                if native_call is not None:
                    try:
                        native_result = native_call()
                        if not isinstance(native_result, torch.Tensor):
                            raise TypeError(f"native RMSNorm returned {type(native_result).__name__}, expected Tensor")
                        stream.synchronize()
                        native_samples = _event_samples(
                            native_call, device_module, stream, warmup, batches, iterations_per_batch
                        )
                        native_info["latency_ms_median"] = statistics.median(native_samples)
                        native_info["latency_ms_samples"] = native_samples
                        native_info["measurement_method"] = measurement_method
                    except Exception as exc:
                        stream.synchronize()
                        native_info = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

                rows.append({
                    "schema_version": 2,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "device": device,
                    "device_index": device_index,
                    "hardware": hardware,
                    "shape": [row_count, width],
                    "dtype": str(dtype).removeprefix("torch."),
                    "measurement_method": measurement_method,
                    "measurement_scope": "device_stream_elapsed_per_call",
                    "execution_mode": "eager",
                    "graph_capture": False,
                    "warmup_iterations": warmup,
                    "batches": batches,
                    "iterations_per_batch": iterations_per_batch,
                    "latency_ms_median": statistics.median(kernel_samples),
                    "latency_ms_samples": kernel_samples,
                    "correctness_reference": "xpu_kernels.golden.rms_norm_golden (not a performance baseline)",
                    "native_baseline": native_info,
                    "baseline_best_ms": None,
                    "speedup_vs_baseline": None,
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "runtime": runtime,
                })
    return rows


def validate_benchmark_record(row: dict) -> None:
    required = {
        "schema_version", "timestamp_utc", "device", "device_index", "hardware", "shape", "dtype",
        "measurement_method", "measurement_scope", "execution_mode", "graph_capture",
        "warmup_iterations", "batches", "iterations_per_batch", "latency_ms_median",
        "latency_ms_samples", "correctness_reference", "native_baseline", "baseline_best_ms",
        "speedup_vs_baseline", "python", "torch", "runtime",
    }
    if set(row) != required:
        raise ValueError(f"benchmark record fields differ from schema: {sorted(set(row) ^ required)}")
    if row["schema_version"] != 2 or len(row["shape"]) != 2:
        raise ValueError("invalid benchmark record version or shape")
    if row["batches"] < 1 or len(row["latency_ms_samples"]) != row["batches"]:
        raise ValueError("batch count does not match device-timing sample count")
    if row["iterations_per_batch"] < 1 or row["latency_ms_median"] < 0:
        raise ValueError("invalid benchmark timing parameters")
    if row["measurement_method"] not in ("torch.cuda.Event.elapsed_time", "torch.npu.Event.elapsed_time"):
        raise ValueError("benchmark latency must come from framework device events")
    if row["measurement_scope"] != "device_stream_elapsed_per_call" or row["graph_capture"]:
        raise ValueError("benchmark scope must be eager device-stream timing")


def _tolerances(torch, dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-6
    if dtype == torch.float16:
        return 2e-3, 9.765625e-4
    return 1e-2, 7.8125e-3


def attach_existing_baseline(rows: list[dict], results_dir: Path) -> list[dict]:
    prior = []
    if results_dir.is_dir():
        for path in results_dir.glob("*.jsonl"):
            try:
                prior.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            except (OSError, ValueError):
                continue
    for row in rows:
        runtime = row.get("runtime", {})
        if row.get("device") == "npu":
            versions_known = bool(runtime.get("cann_version") and runtime.get("torch_npu"))
        else:
            versions_known = bool(runtime.get("cuda") and runtime.get("triton"))
        if not versions_known:
            row["baseline_best_ms"] = None
            row["speedup_vs_baseline"] = None
            continue
        matches = [
            item for item in prior
            if item.get("schema_version") == 2
            and item.get("device") == row["device"]
            and item.get("hardware") == row["hardware"]
            and item.get("shape") == row["shape"]
            and item.get("dtype") == row["dtype"]
            and item.get("measurement_method") == row["measurement_method"]
            and item.get("execution_mode") == row["execution_mode"]
            and item.get("device_index") == row["device_index"]
            and item.get("runtime") == row["runtime"]
            and item.get("torch") == row["torch"]
            and item.get("python") == row["python"]
        ]
        best = min((item["latency_ms_median"] for item in matches), default=None)
        row["baseline_best_ms"] = best
        row["speedup_vs_baseline"] = (best / row["latency_ms_median"] if best is not None and row["latency_ms_median"] > 0 else None)
    return rows
