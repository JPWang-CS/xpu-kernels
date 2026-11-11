from __future__ import annotations

import os
from pathlib import Path

_checkout = Path(__file__).resolve().parents[3]
if (_checkout / "pyproject.toml").is_file():
    _cache_root = _checkout / "cache"
else:
    _cache_value = os.environ.get("XPU_KERNELS_CACHE_DIR")
    if not _cache_value:
        raise RuntimeError("wheel installs require XPU_KERNELS_CACHE_DIR to point to a writable project-owned cache directory")
    _cache_root = Path(_cache_value).expanduser().resolve()
_triton_cache = _cache_root / "triton"
_triton_cache.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(_triton_cache)

import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(X, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    w = tl.load(W + col, col < WIDTH, other=0)
    mean_square = tl.sum(x * x, axis=0) / WIDTH
    normalized = x * tl.rsqrt(mean_square + EPS)
    # Preserve the golden's cast-before-gamma rounding for low precision.
    normalized = normalized.to(Y.dtype.element_ty).to(tl.float32)
    result = normalized * w.to(tl.float32)
    tl.store(Y + row * WIDTH + col, result, col < WIDTH)


def rms_norm_gpu(x, weight, eps: float, out):
    if not x.is_cuda:
        raise RuntimeError("GPU backend requires a CUDA tensor")
    width = x.shape[-1]
    rows = x.numel() // width
    block = triton.next_power_of_2(width)
    _rms_norm_kernel[(rows,)](x, weight, out, width, eps, block)
    return out
