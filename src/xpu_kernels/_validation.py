from __future__ import annotations

from typing import Any


SUPPORTED_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def validate_rms_norm(x: Any, weight: Any, eps: float, out: Any | None) -> Any:
    import torch

    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("x and weight must be torch.Tensor objects")
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("x must have at least one dimension and a non-empty last dimension")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not weight.is_contiguous() or weight.ndim != 1:
        raise ValueError("weight must be a contiguous 1-D tensor")
    if str(x.dtype) not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported input dtype: {x.dtype}")
    if weight.dtype != x.dtype:
        raise TypeError("weight dtype must match x dtype")
    if weight.device != x.device:
        raise ValueError("weight and x must be on the same device")
    if weight.numel() != x.shape[-1]:
        raise ValueError("weight length must equal the last dimension of x")
    if not isinstance(eps, (float, int)) or eps <= 0:
        raise ValueError("eps must be a positive number")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor or None")
        if out.shape != x.shape or out.device != x.device or out.dtype != x.dtype:
            raise ValueError("out must match x shape, device, and dtype")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    return torch
