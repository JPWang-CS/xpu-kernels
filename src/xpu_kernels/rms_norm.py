from __future__ import annotations

from ._validation import validate_rms_norm


def rms_norm(x, weight, eps: float = 1e-5, out=None):
    """Normalize the last dimension using the implementation for ``x.device``.

    There is no implicit reference fallback. Use ``golden.rms_norm_golden``
    explicitly when a reference result is wanted.
    """
    torch = validate_rms_norm(x, weight, eps, out)
    result = out if out is not None else torch.empty_like(x)
    if x.device.type == "cuda":
        from .backends.gpu import rms_norm_gpu

        if x.numel() == 0:
            return result
        return rms_norm_gpu(x, weight, float(eps), result)
    if x.device.type == "npu":
        from .backends.npu import rms_norm_npu

        if x.numel() == 0:
            return result
        return rms_norm_npu(x, weight, float(eps), result)
    raise RuntimeError(
        f"rms_norm has no implementation for device type {x.device.type!r}; "
        "supported devices are CUDA and NPU"
    )
