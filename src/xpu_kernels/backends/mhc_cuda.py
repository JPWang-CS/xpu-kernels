"""Adapt the upstream DeepSeek TileKernels post-backward CUDA kernel."""
from __future__ import annotations

import math


def mhc_post_backward_cuda(grad_y, x, h_res, h_out, h_post, outputs=None):
    import torch

    if x.dtype != torch.bfloat16 or x.shape[-1] % 128:
        raise ValueError("GPU mHC supports BF16 and feature widths divisible by 128")
    try:
        from ._tilekernels.post_cuda import mhc_post_bwd_cuda
    except ImportError as exc:
        raise RuntimeError("GPU mHC requires tilelang>=0.1.15; install the mhc-gpu extra") from exc
    tokens = x.numel() // (4*x.shape[-1])
    width = x.shape[-1]
    with torch.cuda.device(x.device):
        # Upstream a,b,c,d correspond to h_res,x,h_post,h_out respectively.
        da, db, dc, dd = mhc_post_bwd_cuda(
            grad_y.reshape(tokens,4,width), h_res.reshape(tokens,4,4),
            x.reshape(tokens,4,width), h_post.reshape(tokens,4), h_out.reshape(tokens,width),
            n_thr=64, h_blk=math.gcd(width,512), use_pdl=False,
        )
        result = (db.view_as(x), da.view_as(h_res), dd.view_as(h_out), dc.view_as(h_post))
        if outputs is not None:
            with torch.no_grad():
                for target, source in zip(outputs,result):
                    target.copy_(source)
            return outputs
        return result
