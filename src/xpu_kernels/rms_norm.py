from __future__ import annotations

from ._validation import validate_rms_norm


def rms_norm(x, weight, eps: float = 1e-5, out=None, *, implementation=None):
    """Normalize the last dimension using the implementation for ``x.device``.

    CUDA defaults to the native CUDA extension; ``implementation="triton"``
    selects Triton. NPU uses the selected optimization inside its exact scope
    and CANN mainline otherwise. ``implementation="native"`` selects CANN.
    There is no implicit reference fallback. Use ``golden.rms_norm_golden``
    explicitly when a reference result is wanted.
    """
    npu_input = getattr(getattr(x, "device", None), "type", None) == "npu"
    torch = validate_rms_norm(x, weight, eps, out, allow_noncontiguous=npu_input)
    result = out if out is not None else (None if npu_input else torch.empty_like(x))
    if x.device.type == "cuda":
        if implementation not in (None, "cuda", "triton"):
            raise ValueError("CUDA input requires implementation=None, 'cuda', or 'triton'")
        if x.numel() == 0:
            return result if result is not None else torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
        if implementation in (None, "cuda"):
            from .backends.cuda import rms_norm_cuda

            return rms_norm_cuda(x, weight, float(eps), result)
        from .backends.gpu import rms_norm_gpu

        return rms_norm_gpu(x, weight, float(eps), result)
    if x.device.type == "npu":
        if implementation not in (None, "auto", "ascendc", "native", "optimized"):
            raise ValueError("NPU implementation must be auto, native, or optimized ('ascendc' aliases auto)")
        if x.numel() == 0:
            return result if result is not None else torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
        from .backends.npu import rms_norm_npu

        return rms_norm_npu(x, weight, float(eps), result, implementation=implementation or "auto")
    raise RuntimeError(
        f"rms_norm has no implementation for device type {x.device.type!r}; "
        "supported devices are CUDA and NPU"
    )
