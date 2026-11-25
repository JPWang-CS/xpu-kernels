"""Independent FP64 reference for mHC post backward."""

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
