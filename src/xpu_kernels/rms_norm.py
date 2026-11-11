from __future__ import annotations

from ._validation import validate_rms_norm


def rms_norm(x, weight, eps: float = 1e-5, out=None, *, implementation=None):
    """Normalize the last dimension using the implementation for ``x.device``.

    CUDA defaults to the native CUDA extension; ``implementation="triton"``
    selects Triton. NPU defaults to AscendC and accepts ``"ascendc"`` explicitly.
    There is no implicit reference fallback. Use ``golden.rms_norm_golden``
    explicitly when a reference result is wanted.
    """
    torch = validate_rms_norm(x, weight, eps, out)
    result = out if out is not None else torch.empty_like(x)
    if x.device.type == "cuda":
        if implementation not in (None, "cuda", "triton"):
            raise ValueError("CUDA input requires implementation=None, 'cuda', or 'triton'")
        if x.numel() == 0:
            return result
        if implementation in (None, "cuda"):
            from .backends.cuda import rms_norm_cuda

            return rms_norm_cuda(x, weight, float(eps), result)
        from .backends.gpu import rms_norm_gpu

        return rms_norm_gpu(x, weight, float(eps), result)
    if x.device.type == "npu":
        if implementation not in (None, "ascendc"):
            raise ValueError("NPU input requires implementation=None or 'ascendc'")
        if x.numel() == 0:
            return result
        from .backends.npu import rms_norm_npu

        return rms_norm_npu(x, weight, float(eps), result)
    raise RuntimeError(
        f"rms_norm has no implementation for device type {x.device.type!r}; "
        "supported devices are CUDA and NPU"
    )
