"""CANN-native dispatch and the exact measured Ascend 950 optimization scope."""
from __future__ import annotations


NUMERICAL_CONTRACT = "rmsnorm-fp32-normalization-gamma-final-cast"


def optimized_rms_norm_scope(x, weight, eps=1e-5, *, device_name=None):
    """True only for a measured layout, shape, dtype and Ascend architecture."""
    if x.device.type != "npu" or str(x.dtype) != "torch.bfloat16":
        return False
    if not x.is_contiguous() or not weight.is_contiguous() or float(eps) != 1e-5:
        return False
    if x.shape[-1] != 20480 or x.numel() // 20480 not in (2048, 4096):
        return False
    if device_name is None:
        import torch
        device_name = torch.npu.get_device_name(x.device)
    return str(device_name).startswith("Ascend950")


def rms_norm_npu_with_rstd(x, weight, eps=1e-5, *, out=None, rstd_out=None, implementation="auto"):
    """Internal two-output adapter used by device validation and the public Y API."""
    import torch
    import torch_npu
    from ._validation import validate_rms_norm

    validate_rms_norm(x, weight, eps, out, allow_noncontiguous=True)
    if x.device.type != "npu":
        raise RuntimeError("CANN RMSNorm requires NPU input")
    if implementation == "ascendc":
        implementation = "auto"
    if implementation not in ("auto", "native", "optimized"):
        raise ValueError("implementation must be auto, native, or optimized")
    rshape = tuple(x.shape[:-1]) + (1,)
    if rstd_out is not None:
        if (not isinstance(rstd_out, torch.Tensor) or tuple(rstd_out.shape) != rshape
                or rstd_out.dtype != torch.float32 or rstd_out.device != x.device
                or not rstd_out.is_contiguous()):
            raise ValueError("rstd_out must be contiguous FP32 with the input leading dimensions and last dimension 1")
    eligible = implementation != "native" and optimized_rms_norm_scope(x, weight, eps)
    if implementation == "optimized" and not eligible:
        raise ValueError("optimized RMSNorm requires Ascend950, contiguous BF16, D=20480, M in {2048,4096}, eps=1e-5")
    if eligible:
        from .backends.npu import rms_norm_cann_npu
        y = out if out is not None else torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
        rstd = rstd_out if rstd_out is not None else torch.empty(rshape, dtype=torch.float32, device=x.device)
        return rms_norm_cann_npu(x, weight, float(eps), y, rstd)
    native = getattr(torch_npu, "npu_rms_norm", None)
    if native is None:
        native = getattr(getattr(torch.ops, "npu", None), "npu_rms_norm", None)
    if native is None:
        raise RuntimeError("CANN mainline npu_rms_norm is unavailable; numerical-reference fallback is disabled")
    y, rstd = native(x.contiguous(), weight.contiguous(), epsilon=float(eps))
    if out is not None:
        out.copy_(y)
        y = out
    if rstd_out is not None:
        rstd_out.copy_(rstd.reshape(rshape))
        rstd = rstd_out
    return y, rstd
