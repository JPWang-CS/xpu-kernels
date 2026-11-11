from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _project_root() -> Path:
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "csrc" / "ascend" / "rms_norm.cpp").is_file():
        return checkout
    packaged = Path(__file__).resolve().parents[1] / "_native"
    if (packaged / "csrc" / "ascend" / "rms_norm.cpp").is_file():
        return packaged
    raise RuntimeError("native RMSNorm sources are missing from both the source checkout and installed package")


@lru_cache(maxsize=1)
def _load_extension():
    try:
        import torch
        import torch_npu  # noqa: F401  # registers the NPU device and stream API
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError("NPU execution requires torch, torch_npu, and CANN") from exc

    root = _project_root()
    source = root / "csrc" / "ascend" / "rms_norm.cpp"
    deep_jit = root / "third-party" / "deep-jit" / "include"
    if not source.is_file() or not (deep_jit / "deep_jit" / "backend" / "ascend" / "backend.hpp").is_file():
        raise RuntimeError("NPU sources or DeepJIT headers are missing; initialize the repository submodule")

    checkout = (root / "pyproject.toml").is_file()
    cache_value = str(root / "cache") if checkout else os.environ.get("XPU_KERNELS_CACHE_DIR")
    if not cache_value:
        raise RuntimeError("wheel installs require XPU_KERNELS_CACHE_DIR to point to a writable project-owned cache directory")
    cache_root = Path(cache_value)
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    build_dir = cache_root / "build" / "npu-extension"
    build_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / "torch-extensions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache_dir)
    os.environ["XPU_KERNELS_ROOT"] = str(root)
    os.environ["XPU_KERNELS_JIT_CACHE_DIR"] = str(cache_root / "deep-jit")
    os.environ.setdefault("XPU_KERNELS_JIT_DEBUG", "0")

    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not ascend_home:
        raise RuntimeError("source the project-selected CANN environment before building the NPU extension")
    torch_npu_dir = Path(torch_npu.__file__).resolve().parent
    includes = [
        deep_jit,
        root / "kernels" / "include",
        *map(Path, torch.utils.cpp_extension.include_paths()),
        Path(ascend_home) / "include",
        Path(ascend_home) / "aarch64-linux" / "include",
        torch_npu_dir / "include",
        torch_npu_dir / "include" / "third_party" / "acl" / "inc",
    ]
    libraries = [Path(ascend_home) / "lib64", torch_npu_dir / "lib"]
    ldflags = ["-ldl", "-lascendcl", "-ltorch_npu"]
    ldflags.extend(f"-L{path}" for path in libraries)
    ldflags.extend(f"-Wl,-rpath,{path}" for path in libraries)
    return load(
        name="xpu_kernels_ascend",
        sources=[str(source), str(root / "csrc" / "ascend" / "mhc_post_backward.cpp")],
        extra_cflags=["-std=c++20", "-O3", "-fPIC", "-DDJ_DISABLE_GIL=1"],
        extra_include_paths=[str(path) for path in includes],
        extra_ldflags=ldflags,
        build_directory=str(build_dir),
        with_cuda=False,
        verbose=True,
    )


def rms_norm_npu(x, weight, eps: float, out):
    if x.device.type != "npu":
        raise RuntimeError("NPU backend requires an NPU tensor")
    module = _load_extension()
    return module.rms_norm(x, weight, out, eps)
