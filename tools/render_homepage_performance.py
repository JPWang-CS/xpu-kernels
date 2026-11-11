#!/usr/bin/env python3
"""Render the README's BF16 RMSNorm comparison from public benchmark JSONL."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile


BEGIN = "<!-- BEGIN homepage-performance -->"
END = "<!-- END homepage-performance -->"
SHAPES = ((128, 4096), (129, 4097), (1024, 4096), (4096, 8192))
DTYPE = "bfloat16"
SOURCES = (
    ("rmsnorm-gpu-v2.jsonl", "cuda", "NVIDIA L20X", ("cuda", "triton", "torch")),
    ("rmsnorm-npu-v2.jsonl", "npu", "Ascend950DT_9582", ("ascendc",)),
)


class ValidationError(ValueError):
    """The public evidence or README markers do not meet the rendering contract."""


def timing_signature(record: dict, location: str) -> tuple:
    values = []
    for key, minimum in (("warmup_iterations", 1), ("batches", 1), ("iterations_per_batch", 1)):
        value = record.get(key)
        if type(value) is not int or value < minimum:
            raise ValidationError(f"{location}: {key} must be an integer >= {minimum}")
        values.append(value)
    mode = record.get("execution_mode")
    if not isinstance(mode, str) or not mode:
        raise ValidationError(f"{location}: execution_mode is missing")
    scope = record.get("measurement_scope", record.get("timing_scope"))
    if scope != "device_stream_elapsed_per_call":
        raise ValidationError(f"{location}: unsupported or missing timing scope: {scope!r}")
    if "timing_scope" in record and record["timing_scope"] != scope:
        raise ValidationError(f"{location}: timing_scope and measurement_scope disagree")
    graph = record.get("graph_capture")
    if type(graph) is not bool:
        raise ValidationError(f"{location}: graph_capture must be a boolean")
    return (*values, mode, scope, graph)


def load_records(path: Path, device: str, hardware: str, implementations: tuple[str, ...]) -> dict:
    """Validate every row and reject duplicate (shape, dtype, implementation) keys."""
    records = {}
    expected_method = f"torch.{device}.Event.elapsed_time"
    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            location = f"{path.name}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{location}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValidationError(f"{location}: a benchmark record must be a JSON object")
            correctness = record.get("correctness")
            if not isinstance(correctness, dict) or correctness.get("status") != "PASS":
                raise ValidationError(f"{location}: correctness.status must be PASS")
            for key, expected in (("operator", "rms_norm"), ("device", device), ("hardware", hardware), ("measurement_method", expected_method)):
                if record.get(key) != expected:
                    raise ValidationError(f"{location}: {key} must be {expected!r}, got {record.get(key)!r}")
            dtype = record.get("dtype")
            implementation = record.get("implementation")
            if dtype not in ("float32", "float16", "bfloat16"):
                raise ValidationError(f"{location}: unsupported dtype: {dtype!r}")
            if implementation not in implementations:
                raise ValidationError(f"{location}: invalid {device} implementation: {implementation!r}")
            shape = record.get("shape")
            if not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
                raise ValidationError(f"{location}: shape must contain two positive integers")
            latency = record.get("latency_ms_median")
            if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
                raise ValidationError(f"{location}: latency_ms_median must be finite and positive")
            timing_signature(record, location)
            key = (tuple(shape), dtype, implementation)
            if key in records:
                raise ValidationError(f"{location}: duplicate shape/dtype/implementation: {key!r}")
            records[key] = record
    if not records:
        raise ValidationError(f"{path.name}: no benchmark records")
    return records


def render_performance(gpu: dict, npu: dict) -> str:
    # Compare protocol settings within every common shape/dtype case. The event
    # implementation differs across devices and is deliberately not in this key.
    protocols = {}
    for records in (gpu, npu):
        for (shape, dtype, implementation), record in records.items():
            case = (shape, dtype)
            signature = timing_signature(record, f"{record['device']}/{implementation}/{shape}/{dtype}")
            if case in protocols and protocols[case] != signature:
                raise ValidationError(f"{shape}/{dtype}: warmup/batches/iterations/mode/timing scope differ between implementations or devices")
            protocols[case] = signature

    selected = {}
    for shape in SHAPES:
        rows = []
        for records, implementation in ((npu, "ascendc"), (gpu, "cuda"), (gpu, "triton"), (gpu, "torch")):
            key = (shape, DTYPE, implementation)
            if key not in records:
                raise ValidationError(f"missing BF16 record: shape={shape}, implementation={implementation}")
            rows.append(records[key]["latency_ms_median"])
        selected[shape] = rows

    lines = [
        "| Shape | NPU 950DT | GPU L20X | GPU L20X（Triton） | GPU L20X（单算子拼接） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for shape in SHAPES:
        times = selected[shape]
        fastest_gpu = min(times[1:])
        cells = []
        for column, latency_ms in enumerate(times):
            text = f"{latency_ms * 1000:.3f}"
            cells.append(f"**{text}**" if column > 0 and latency_ms == fastest_gpu else text)
        lines.append(f"| {shape[0]} × {shape[1]} | " + " | ".join(cells) + " |")
    small = selected[(128, 4096)]
    large = selected[(4096, 8192)]
    lines.extend((
        "",
        f"相对 GPU 单算子拼接，128×4096 的 CUDA 加速比为 {small[3] / small[1]:.2f}×，"
        f"4096×8192 的 Triton 加速比为 {large[3] / large[2]:.2f}×。",
    ))
    return "\n".join(lines)


def replace_marked_block(original: str, content: str) -> str:
    if original.count(BEGIN) != 1 or original.count(END) != 1:
        raise ValidationError("README must contain exactly one BEGIN and one END homepage-performance marker")
    start = original.index(BEGIN) + len(BEGIN)
    stop = original.index(END)
    if stop < start:
        raise ValidationError("README homepage-performance END marker precedes BEGIN")
    following = original[start:]
    newline = "\r\n" if following.startswith("\r\n") else "\n"
    return original[:start] + newline + content.replace("\n", newline) + newline + original[stop:]


def write_if_unchanged(path: Path, original: bytes, updated: bytes) -> None:
    """Publish in the same directory; refuse an observed concurrent README edit."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".README.homepage-performance-", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            target.write(updated)
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        if path.read_bytes() != original:
            raise ValidationError("README changed during rendering; retry against the current file")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the generated block without writing README")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        loaded = []
        for filename, device, hardware, implementations in SOURCES:
            path = (root / "results" / "benchmarks" / filename).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValidationError(f"public benchmark path resolves outside the project: {filename}")
            loaded.append(load_records(path, device, hardware, implementations))
        content = render_performance(*loaded)
        readme = (root / "README.md").resolve(strict=True)
        if not readme.is_relative_to(root):
            raise ValidationError("README resolves outside the project")
        original = readme.read_bytes()
        updated = replace_marked_block(original.decode("utf-8"), content).encode("utf-8")
        if updated == original:
            print("README homepage performance is current.")
            return 0
        if args.check:
            print("README homepage performance is out of date.", file=sys.stderr)
            return 1
        write_if_unchanged(readme, original, updated)
        print("Updated README homepage performance.")
        return 0
    except (OSError, UnicodeError, ValidationError) as exc:
        print(f"render_homepage_performance: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
