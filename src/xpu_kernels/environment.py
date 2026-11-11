from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import sys
from pathlib import Path


def _spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _selected_cxx() -> tuple[str | None, str | None]:
    configured = os.environ.get("CXX")
    if configured:
        parts = shlex.split(configured)
        command = parts[0] if parts else None
    else:
        command = shutil.which("c++") or shutil.which("g++")
        configured = command
    resolved = shutil.which(command) if command else None
    if resolved is None and command and Path(command).is_file():
        resolved = str(Path(command).resolve())
    return configured, resolved


def inspect_environment(backend: str = "all") -> dict:
    configured_cxx, resolved_cxx = _selected_cxx()
    result = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "torch_installed": _spec("torch"),
        "triton": _spec("triton"),
        "torch_npu": _spec("torch_npu"),
        "cxx": configured_cxx or shutil.which("c++") or shutil.which("g++"),
        "cxx_resolved": resolved_cxx,
        "nvcc": shutil.which("nvcc"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "bisheng": shutil.which("bisheng"),
        "ld_lld": shutil.which("ld.lld"),
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        "ascend_toolkit_home": os.environ.get("ASCEND_TOOLKIT_HOME"),
        "project_root": str(Path(__file__).resolve().parents[2]),
    }
    missing = []
    if backend in ("all", "gpu"):
        if not result["torch_installed"]:
            missing.append("torch")
        if not result["triton"]:
            missing.append("triton")
        if backend == "gpu" and not result["nvcc"]:
            missing.append("NVCC (CUDA C++ backend)")
        if backend == "gpu" and result["cxx_resolved"] is None:
            missing.append("C++ compiler")
        if backend == "gpu" and result["torch_installed"]:
            try:
                import torch

                result["cuda_available"] = bool(torch.cuda.is_available())
                if result["cuda_available"]:
                    result["gpu_name"] = torch.cuda.get_device_name()
                    result["gpu_compute_capability"] = list(torch.cuda.get_device_capability())
                result["torch_version"] = torch.__version__
                result["cuda_runtime"] = torch.version.cuda
                if backend == "gpu" and not result["cuda_available"]:
                    missing.append("CUDA device")
            except Exception as exc:
                result["cuda_probe_error"] = str(exc)
    if backend in ("all", "npu"):
        if not result["torch_installed"]:
            missing.append("torch")
        if not result["torch_npu"]:
            missing.append("torch_npu")
        if result["cxx_resolved"] is None:
            missing.append("C++ compiler")
        if backend == "npu" and result["torch_npu"]:
            try:
                import torch
                import torch_npu  # noqa: F401

                result["npu_available"] = bool(torch.npu.is_available())
                if result["npu_available"]:
                    result["npu_name"] = torch.npu.get_device_name()
                result["torch_version"] = torch.__version__
                result["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")
                if backend == "npu" and not result["npu_available"]:
                    missing.append("NPU device")
            except Exception as exc:
                result["npu_probe_error"] = str(exc)
        if backend == "npu" and not (result["ascend_home_path"] or result["ascend_toolkit_home"]):
            missing.append("ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME")
        if backend == "npu" and not result["bisheng"]:
            missing.append("bisheng")
        if backend == "npu" and not result["ld_lld"]:
            missing.append("ld.lld")
    result["missing"] = sorted(set(missing))
    return result


def require_device(device: str) -> None:
    if not _spec("torch"):
        raise RuntimeError("PyTorch is not installed in the active Python environment")
    import torch

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable; no device test or benchmark was run")
    elif device == "npu":
        if not _spec("torch_npu"):
            raise RuntimeError("torch_npu is not installed; no device test or benchmark was run")
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("NPU device is unavailable; no device test or benchmark was run")
