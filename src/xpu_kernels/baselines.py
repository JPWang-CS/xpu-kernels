"""Device-side composition of PyTorch primitives, measured independently of Golden."""


def make_cuda_composed(x, weight, out, eps=1e-5):
    import torch

    if not x.is_cuda:
        raise ValueError("the composed benchmark is GPU-only")
    width = x.shape[-1]
    rows = x.numel() // width
    original = x.reshape(rows, width)
    x32 = original if x.dtype == torch.float32 else torch.empty_like(original, dtype=torch.float32)
    squared = torch.empty_like(x32)
    variance = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    normalized32 = torch.empty_like(x32)
    normalized = normalized32 if x.dtype == torch.float32 else torch.empty_like(original)
    destination = out.reshape(rows, width)

    def call():
        if x.dtype != torch.float32:
            x32.copy_(original)
        torch.mul(x32, x32, out=squared)
        torch.mean(squared, dim=-1, keepdim=True, out=variance)
        torch.add(variance, eps, out=variance)
        torch.rsqrt(variance, out=variance)
        torch.mul(x32, variance, out=normalized32)
        if x.dtype != torch.float32:
            normalized.copy_(normalized32)
        torch.mul(normalized, weight, out=destination)
        return out

    return call
