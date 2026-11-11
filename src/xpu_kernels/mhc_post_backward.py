"""mHC post-backward Python API, golden, and supported scope.

The NPU API supports Ascend 950, FP16/BF16, N=4, contiguous 3D [T,4,D]
or 4D [B,S,4,D] tensors, 512 <= token count (T or B*S) <= 49152, and
1 <= D <= 8192. Wider legacy/reference shapes are outside this first API
scope. The golden remains available independently for small CPU/CUDA/NPU cases.
"""

from __future__ import annotations


def mhc_post_backward_golden(grad_y, x, h_res, h_out, h_post):
    """Evaluate the mHC post-backward equations in explicit FP64."""
    import torch

    tensors = (grad_y, x, h_res, h_out, h_post)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all inputs must be torch tensors")
    if not (grad_y.shape == x.shape and x.ndim in (3, 4) and x.shape[-2] == 4):
        raise ValueError("grad_y and x must share shape [..., 4, D]")
    lead, d = x.shape[:-2], x.shape[-1]
    if h_res.shape != (*lead, 4, 4) or h_out.shape != (*lead, d) or h_post.shape != (*lead, 4):
        raise ValueError("h_res, h_out, and h_post have incompatible shapes")
    gy, xx = grad_y.detach().cpu().double(), x.detach().cpu().double()
    hr = h_res.detach().cpu().double()
    ho, hp = h_out.detach().cpu().double(), h_post.detach().cpu().double()
    gx = torch.einsum("...ij,...jd->...id", hr, gy).to(device=x.device, dtype=x.dtype)
    ghr = torch.einsum("...id,...jd->...ij", xx, gy).to(device=x.device, dtype=torch.float32)
    gho = torch.einsum("...j,...jd->...d", hp, gy).to(device=x.device, dtype=x.dtype)
    ghp = torch.einsum("...d,...jd->...j", ho, gy).to(device=x.device, dtype=torch.float32)
    return gx, ghr, gho, ghp


def _validate(grad_y, x, h_res, h_out, h_post, out):
    import torch

    vals = (grad_y, x, h_res, h_out, h_post)
    if any(not isinstance(t, torch.Tensor) for t in vals):
        raise TypeError("all inputs must be torch tensors")
    if x.ndim not in (3, 4) or x.shape[-2] != 4 or x.shape[-1] < 1:
        raise ValueError("x must have shape [T,4,D] or [B,S,4,D] with D positive")
    if not x.is_contiguous() or any(not t.is_contiguous() for t in vals):
        raise ValueError("all inputs must be contiguous")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("x and grad_y must use float16 or bfloat16")
    lead, d = x.shape[:-2], x.shape[-1]
    if grad_y.shape != x.shape or grad_y.dtype != x.dtype or grad_y.device != x.device:
        raise ValueError("grad_y must match x shape, dtype, and device")
    if h_res.shape != (*lead, 4, 4) or h_res.dtype != torch.float32:
        raise ValueError("h_res must be FP32 with shape [...,4,4]")
    if h_out.shape != (*lead, d) or h_out.dtype != x.dtype:
        raise ValueError("h_out must match x dtype with shape [...,D]")
    if h_post.shape != (*lead, 4) or h_post.dtype != torch.float32:
        raise ValueError("h_post must be FP32 with shape [...,4]")
    if any(t.device != x.device for t in (h_res, h_out, h_post)):
        raise ValueError("all inputs must be on the same device")
    specs = ((x.shape, x.dtype), (h_res.shape, torch.float32), (h_out.shape, x.dtype), (h_post.shape, torch.float32))
    if out is None:
        outputs = None
    else:
        if not isinstance(out, tuple) or len(out) != 4:
            raise TypeError("out must be a tuple of four tensors")
        for i, (t, (shape, dtype)) in enumerate(zip(out, specs)):
            if not isinstance(t, torch.Tensor) or t.shape != shape or t.dtype != dtype or t.device != x.device or not t.is_contiguous():
                raise ValueError(f"out[{i}] must be contiguous with shape {tuple(shape)}, dtype {dtype}, and device {x.device}")
            if any(torch._C._overlaps(t, source) for source in (*vals, *out[:i])):
                raise ValueError("output buffers must not overlap inputs or other outputs")
        outputs = out
    return torch, outputs


def optimized_scope(x):
    """Return whether the final adaptive kernel applies to this input."""
    import torch
    tokens = x.shape[0] if x.ndim == 3 else x.shape[0] * x.shape[1]
    return (x.dtype == torch.bfloat16 and 2048 <= tokens <= 49152
            and x.shape[-2] == 4 and x.shape[-1] in (2560, 5120, 6144, 7168))


def mhc_post_backward_native(grad_y, x, h_res, h_out, h_post):
    """Call the installed CANN mainline ACLNN operator, without a private overlay."""
    import torch_npu
    native = getattr(torch_npu, "npu_mhc_post_backward", None)
    if native is None:
        raise RuntimeError("torch_npu.npu_mhc_post_backward is unavailable; use a compatible CANN/torch_npu package")
    return tuple(native(grad_y, x, h_res, h_out, h_post))


def mhc_post_backward(grad_y, x, h_res, h_out, h_post, out=None, *, implementation="auto"):
    """Run mHC post backward on Ascend 950 within the supported scope."""
    torch, outputs = _validate(grad_y, x, h_res, h_out, h_post, out)
    if x.device.type != "npu":
        raise RuntimeError("mHC post backward requires an Ascend NPU; golden fallback is disabled")
    token_count = x.shape[0] if x.ndim == 3 else x.shape[0] * x.shape[1]
    if not 512 <= token_count <= 49152:
        raise ValueError("NPU implementation requires token count T (or B*S) in [512, 49152]")
    if x.shape[-1] > 8192:
        raise ValueError("NPU implementation requires 1 <= D <= 8192")
    if implementation not in ("auto", "native", "optimized"):
        raise ValueError("implementation must be 'auto', 'native', or 'optimized'")
    eligible = optimized_scope(x)
    if implementation == "optimized" and not eligible:
        raise ValueError("optimized mHC requires BF16, N=4, 2048..49152 tokens and D in {2560,5120,6144,7168}")
    if implementation == "native" or (implementation == "auto" and not eligible):
        result = mhc_post_backward_native(grad_y, x, h_res, h_out, h_post)
        if out is None:
            return result
        # Preserve the public out contract; paired timing calls both without out.
        import torch
        with torch.no_grad():
            for target, source in zip(outputs, result):
                target.copy_(source)
        return outputs
    from .backends.mhc_npu import mhc_post_backward_npu
    if outputs is None:
        outputs = tuple(torch.empty_like(t) for t in (x, h_res, h_out, h_post))
    return mhc_post_backward_npu(grad_y, x, h_res, h_out, h_post, outputs, True)
