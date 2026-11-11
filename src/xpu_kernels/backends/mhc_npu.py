"""Lazy Ascend backend for mHC post backward."""

from __future__ import annotations


def mhc_post_backward_npu(grad_y, x, h_res, h_out, h_post, outputs, adaptive=True):
    if x.device.type != "npu":
        raise RuntimeError("mHC NPU backend requires an NPU tensor")
    from .npu import _load_extension

    module = _load_extension()
    module.mhc_post_backward(grad_y, x, h_res, h_out, h_post, *outputs, adaptive)
    return outputs
