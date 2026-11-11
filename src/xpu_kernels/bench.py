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
            "semantics": "Native RMSNorm applies gamma before the final output cast; it does not materialize the normalized input-dtype tensor required by the Pangu contract.",
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
        "semantics": "CANN native RMSNorm is checked against the fused formula with one final output cast; Pangu rounds the normalized value to input dtype before multiplying gamma.",
        "allocates_output": True,
    }


def _native_rms_norm_reference(torch, x, weight, eps: float):
    """Reference the fused native contract, with gamma applied before output cast."""
    width = x.shape[-1]
    rows = x.numel() // width
    x32 = x.reshape(rows, width).to(torch.float32)
    w32 = weight.to(torch.float32)
    result = x32 * torch.rsqrt((x32 * x32).mean(dim=-1, keepdim=True) + float(eps))
    result = (result * w32).to(dtype=x.dtype)
    return result.reshape(x.shape)


def _make_kernel_call(rms_norm, implementation: str, x, weight, out):
    return lambda: rms_norm(x, weight, out=out, implementation=implementation)


def _benchmark_out(torch, x, device: str, include_native_baseline: bool):
    if device == "npu" and include_native_baseline:
        return None
    return torch.empty_like(x)


def run_benchmark(
    device: str,
    shapes: list[tuple[int, int]],
    warmup: int = 5,
    batches: int = 20,
    iterations_per_batch: int = 10,
    dtypes: tuple[str, ...] | None = None,
    check_correctness: bool = True,
    include_native_baseline: bool = False,
    implementations: tuple[str, ...] | None = None,
) -> list[dict]:
    import torch
    from . import rms_norm
    from .golden import rms_norm_golden
    from .baselines import make_cuda_composed

    if batches < 1 or iterations_per_batch < 1 or warmup < 0:
        raise ValueError("batches and iterations_per_batch must be positive; warmup must be non-negative")
    if include_native_baseline and not check_correctness:
        raise ValueError("native comparison requires correctness checks before timing")
    allowed = ("cuda", "triton", "torch") if device == "cuda" else ("ascendc",)
    selected = allowed if implementations is None else implementations
    if not selected or len(set(selected)) != len(selected) or any(x not in allowed for x in selected):
        raise ValueError(f"{device} implementations must be unique selections from {allowed}")
    device_module = torch.cuda if device == "cuda" else torch.npu
    device_index = device_module.current_device()
    hardware = device_module.get_device_name(device_index)
    if device == "cuda":
        import importlib.metadata
        try:
            triton_version = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            triton_version = None
        runtime = {"cuda": torch.version.cuda,
                   "compute_capability": list(torch.cuda.get_device_capability(device_index)),
                   "triton": triton_version}
        measurement_method = "torch.cuda.Event.elapsed_time"
    else:
        import torch_npu
        runtime = {"torch_npu": torch_npu.__version__, "cann_version": _cann_version()}
        measurement_method = "torch.npu.Event.elapsed_time"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    selected_dtypes = tuple(dtype_map) if dtypes is None else dtypes
    if not selected_dtypes or any(name not in dtype_map for name in selected_dtypes):
        raise ValueError(f"unsupported benchmark dtype selection: {selected_dtypes}")
    records = []
    with torch.inference_mode(), device_module.device(device_index):
        stream = device_module.current_stream(device_index)
        for row_count, width in shapes:
            for dtype_name in selected_dtypes:
                dtype = dtype_map[dtype_name]
                # Every implementation sees exactly the same input and weight.
                torch.manual_seed(20261111 + row_count + width)
                x = torch.randn(row_count, width, device=device, dtype=dtype)
                weight = torch.randn(width, device=device, dtype=dtype)
                expected = rms_norm_golden(x, weight, eps=1e-5) if check_correctness else None
                rtol, atol = _tolerances(torch, dtype)
                cases = []
                native_info = {"status": "not_requested"}
                for implementation in selected:
                    allocate_output_per_call = include_native_baseline and device == "npu"
                    out = _benchmark_out(torch, x, device, include_native_baseline)
                    if implementation == "torch":
                        op = make_cuda_composed(x, weight, out)
                    else:
                        op = _make_kernel_call(rms_norm, implementation, x, weight, out)
                    max_abs = None
                    if check_correctness:
                        actual = op()
                        stream.synchronize()
                        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                        max_abs = (actual.float()-expected.float()).abs().max().item()
                    cases.append((implementation, op, max_abs, {
                        "strategy": "out_none_per_call" if allocate_output_per_call else "preallocated_out",
                        "api_call": "rms_norm(out=None)" if allocate_output_per_call else "rms_norm(out=preallocated)",
                        "event_timing_note": (
                            "torch.npu.Event measures device-stream elapsed time; CPU submission gaps may appear in the interval, "
                            "and host output-allocation time is not isolated."
                            if device == "npu" else
                            "torch.cuda.Event measures device-stream elapsed time; host output-allocation time is not isolated."
                        ),
                    }, "xpu_kernels.golden.rms_norm_golden"))
                if include_native_baseline:
                    native_call, native_info = _native_baseline(device, torch, x, weight, 1e-5)
                    if native_call is not None:
                        native_result = native_call()
                        stream.synchronize()
                        native_expected = _native_rms_norm_reference(torch, x, weight, 1e-5)
                        torch.testing.assert_close(native_result, native_expected, rtol=rtol, atol=atol)
                        native_max_abs = (native_result.float()-native_expected.float()).abs().max().item()
                        cases.append(("native", native_call, native_max_abs, {
                            "strategy": "operator_allocates_output_per_call",
                            "api_call": native_info["name"],
                            "event_timing_note": (
                                "torch.npu.Event measures device-stream elapsed time; CPU submission gaps may appear in the interval, "
                                "and host output-allocation time is not isolated."
                                if device == "npu" else
                                "torch.cuda.Event measures device-stream elapsed time; host output-allocation time is not isolated."
                            ),
                        }, "xpu_kernels.bench._native_rms_norm_reference"))
                # Correctness for every implementation is completed before any timed samples are collected.
                for implementation, op, max_abs, output_allocation, correctness_reference in cases:
                    samples = _event_samples(op, device_module, stream, warmup, batches, iterations_per_batch)
                    record = {
                        "schema_version": 4 if include_native_baseline else 3,
                        "operator": "rms_norm", "implementation": implementation,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "device": device, "device_index": device_index, "hardware": hardware,
                        "shape": [row_count, width], "dtype": dtype_name,
                        "measurement_method": measurement_method,
                        "measurement_scope": "device_stream_elapsed_per_call", "execution_mode": "eager",
                        "graph_capture": False, "warmup_iterations": warmup, "batches": batches,
                        "iterations_per_batch": iterations_per_batch,
                        "latency_ms_median": statistics.median(samples), "latency_ms_samples": samples,
                        "correctness_reference": correctness_reference,
                        "correctness": {"status": "PASS" if check_correctness else "not_checked_in_profile",
                                        "max_abs_error": max_abs, "rtol": rtol, "atol": atol},
                        "native_baseline": ({"status": "self", "name": native_info.get("name"),
                                             "semantics": native_info.get("semantics"),
                                             "allocates_output": True}
                                            if implementation == "native" else native_info),
                        "baseline_best_ms": None, "speedup_vs_baseline": None,
                        "python": platform.python_version(), "torch": torch.__version__, "runtime": runtime,
                    }
                    if include_native_baseline:
                        record["output_allocation"] = output_allocation
                    records.append(record)
                    print(f"CASE {device}/{implementation} {row_count}x{width} {dtype_name} "
                          f"{record['correctness']['status']} {record['latency_ms_median']*1000:.3f} us", flush=True)
    return records


def validate_benchmark_record(row: dict) -> None:
    required = {
        "schema_version", "timestamp_utc", "device", "device_index", "hardware", "shape", "dtype",
        "measurement_method", "measurement_scope", "execution_mode", "graph_capture",
        "warmup_iterations", "batches", "iterations_per_batch", "latency_ms_median",
        "latency_ms_samples", "correctness_reference", "native_baseline", "baseline_best_ms",
        "speedup_vs_baseline", "python", "torch", "runtime",
    }
    if row.get("schema_version") in (3, 4):
        required |= {"operator", "implementation", "correctness"}
        if row["operator"] != "rms_norm" or row["implementation"] not in ("cuda", "triton", "torch", "ascendc", "native"):
            raise ValueError("invalid operator or implementation")
    if row.get("schema_version") == 4:
        required.add("output_allocation")
    if set(row) != required:
        raise ValueError(f"benchmark record fields differ from schema: {sorted(set(row) ^ required)}")
    if row["schema_version"] not in (2, 3, 4) or len(row["shape"]) != 2:
        raise ValueError("invalid benchmark record version or shape")
    if row["schema_version"] == 4:
        allocation = row["output_allocation"]
        allocation_fields = {"strategy", "api_call", "event_timing_note"}
        if not isinstance(allocation, dict) or set(allocation) != allocation_fields:
            raise ValueError("schema 4 requires explicit output allocation and Event timing metadata")
        if allocation["strategy"] not in (
            "preallocated_out", "out_none_per_call", "operator_allocates_output_per_call"
        ):
            raise ValueError("invalid output allocation strategy")
        if any(not isinstance(allocation[key], str) or not allocation[key] for key in allocation_fields):
            raise ValueError("output allocation metadata values must be non-empty strings")
        if row["device"] == "npu" and "CPU submission" not in allocation["event_timing_note"]:
            raise ValueError("NPU Event metadata must describe possible CPU submission gaps")
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


def _baseline_allocation_strategy(row: dict) -> str | None:
    if row.get("schema_version") == 4:
        allocation = row.get("output_allocation")
        return allocation.get("strategy") if isinstance(allocation, dict) else None
    # Schemas 2 and 3 benchmarked custom implementations with a preallocated out.
    return "preallocated_out"


def _correctness_status(row: dict) -> str:
    correctness = row.get("correctness")
    return correctness.get("status", "unknown") if isinstance(correctness, dict) else "unknown"


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
            if item.get("schema_version") in (2, 3, 4)
            and item.get("operator", "rms_norm") == row.get("operator", "rms_norm")
            and item.get("implementation", "triton" if item.get("device") == "cuda" else "ascendc") == row.get("implementation", "triton" if row.get("device") == "cuda" else "ascendc")
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
            and all(row.get(key) is not None and item.get(key) == row.get(key) for key in (
                "warmup_iterations", "batches", "iterations_per_batch",
            ))
            and _baseline_allocation_strategy(item) == _baseline_allocation_strategy(row)
            and _correctness_status(item) == _correctness_status(row)
        ]
        best = min((item["latency_ms_median"] for item in matches), default=None)
        row["baseline_best_ms"] = best
        row["speedup_vs_baseline"] = (best / row["latency_ms_median"] if best is not None and row["latency_ms_median"] > 0 else None)
    return rows
