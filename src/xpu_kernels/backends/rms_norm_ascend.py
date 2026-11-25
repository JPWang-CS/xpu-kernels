"""Lazy Ascend RMSNorm dispatch and optimized launch."""
from ._ascend import _load_extension

def rms_norm_npu(x, weight, eps: float, out, *, implementation="auto"):
    if x.device.type != "npu":
        raise RuntimeError("NPU backend requires an NPU tensor")
    from ..rms_norm_dispatch import rms_norm_npu_with_rstd
    return rms_norm_npu_with_rstd(x, weight, eps, out=out, implementation=implementation)[0]


def rms_norm_optimized_npu(x, weight, eps: float, y, rstd):
    """Launch the selected CANN-compatible kernel on the current NPU stream."""
    module = _load_extension()
    module.rms_norm(x, weight, y, rstd, eps)
    return y, rstd
