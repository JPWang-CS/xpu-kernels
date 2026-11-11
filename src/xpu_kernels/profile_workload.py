from __future__ import annotations

import argparse
import json

from .bench import run_benchmark
from .environment import require_device


_DTYPES = {"float32", "bfloat16", "float16"}


def check_profile_case(device: str, shape: tuple[int, int], dtype_name: str, implementation: str | None = None) -> None:
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported dtype {dtype_name!r}")
    require_device(device)
    import torch

    from . import rms_norm
    from .bench import _tolerances
    from .golden import rms_norm_golden

    dtype = getattr(torch, dtype_name)
    rows, width = shape
    x = torch.randn(rows, width, device=device, dtype=dtype)
    weight = torch.randn(width, device=device, dtype=dtype)
    out = torch.empty_like(x)
    expected = torch.empty_like(x)
    actual = rms_norm(x, weight, out=out, implementation=implementation)
    reference = rms_norm_golden(x, weight, out=expected)
    (torch.cuda if device == "cuda" else torch.npu).synchronize()
    rtol, atol = _tolerances(torch, dtype)
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--shape", default="16x4096")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--implementation", choices=("cuda", "triton", "ascendc"))
    args = parser.parse_args()
    rows, width = (int(part) for part in args.shape.split("x", 1))
    records = run_benchmark(
        args.device, [(rows, width)], warmup=5, batches=10, iterations_per_batch=10,
        dtypes=(args.dtype,), check_correctness=False, include_native_baseline=False,
        implementations=(args.implementation or ("cuda" if args.device == "cuda" else "ascendc"),),
    )
    for record in records:
        print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
