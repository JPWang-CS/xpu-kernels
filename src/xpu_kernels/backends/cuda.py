from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _project_root() -> Path:
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "csrc" / "cuda" / "rms_norm.cu").is_file():
        return checkout
    packaged = Path(__file__).resolve().parents[1] / "_native"
    if (packaged / "csrc" / "cuda" / "rms_norm.cu").is_file():
        return packaged
    raise RuntimeError("CUDA RMSNorm sources are missing from the checkout and installed package")


@lru_cache(maxsize=1)
def _load_extension():
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError("CUDA RMSNorm requires a CUDA-enabled PyTorch build and its extension tools") from exc
    if torch.version.cuda is None:
        raise RuntimeError("CUDA RMSNorm requires a CUDA-enabled PyTorch build")

    root = _project_root()
    sources = [root / "csrc" / "cuda" / "rms_norm.cpp", root / "csrc" / "cuda" / "rms_norm.cu"]
    if not all(source.is_file() for source in sources):
        raise RuntimeError("CUDA RMSNorm host/device source files are missing")
    cache_value = str(root / "cache") if (root / "pyproject.toml").is_file() else os.environ.get("XPU_KERNELS_CACHE_DIR")
    if not cache_value:
        raise RuntimeError("wheel installs require XPU_KERNELS_CACHE_DIR pointing to a writable project-owned cache")
    cache_root = Path(cache_value).expanduser().resolve()
    build_dir = cache_root / "build" / "cuda-extension"
    temporary_dir = cache_root / "tmp" / "cuda-extension"
    build_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    # Restrict compiler temporary files to the project for this build invocation.
    temporary_keys = ("TMPDIR", "TMP", "TEMP")
    previous = {key: os.environ.get(key) for key in temporary_keys}
    try:
        for key in temporary_keys:
            os.environ[key] = str(temporary_dir)
        return load(
            name="xpu_kernels_cuda",
            sources=[str(source) for source in sources],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=["-O3", "-std=c++20", "--fmad=false"],
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=True,
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def rms_norm_cuda(x, weight, eps: float, out):
    if not x.is_cuda:
        raise RuntimeError("CUDA backend requires a CUDA tensor")
    return _load_extension().rms_norm(x, weight, out, eps)
