"""Reference RMSNorm preserving the agreed Pangu rounding order."""

from __future__ import annotations

from ._validation import validate_rms_norm


def rms_norm_golden(x, weight, eps: float = 1e-5, out=None):
    torch = validate_rms_norm(x, weight, eps, out)
    width = x.shape[-1]
    rows = x.numel() // width
    x32 = x.reshape(rows, width).to(torch.float32)
    w32 = weight.to(torch.float32)
    normalized = x32 * torch.rsqrt((x32 * x32).mean(dim=-1, keepdim=True) + float(eps))
    # Pangu materializes the normalized tensor in input dtype before gamma.
    normalized = normalized.to(dtype=x.dtype)
    result = (normalized * w32.to(dtype=x.dtype)).reshape(x.shape)
    if out is None:
        return result
    out.copy_(result)
    return out
