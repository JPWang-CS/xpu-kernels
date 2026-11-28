"""Independent CPU/device reference for the fixed INT8 payload layout.

Float64 division rounded to FP32 is the mathematical reference, not a claim
that a newly compiled scalar refinement is bitwise equivalent on all inputs.
"""
def fused_quant_payload_golden(x, topk_ids, active):
    import torch
    from .fused_quant_payload import _validate
    _validate(x, topk_ids, active)
    xf = torch.where(active[:,None], x, 0).float()
    if not torch.isfinite(xf).all():
        raise ValueError('active input rows must be finite')
    maximum = xf.abs().amax(dim=1)
    safe = torch.where(maximum == 0, torch.ones_like(maximum), maximum)
    multiplier = (127.0 / safe.double()).float()
    if not torch.isfinite(multiplier).all():
        raise ValueError('quantization multiplier must remain finite')
    quantized = torch.round(xf * multiplier[:,None]).clamp(-128,127).to(torch.int8)
    scale = ((maximum != 0).double() / multiplier.double()).float()
    ids = torch.where(active[:,None], topk_ids, 0).contiguous()
    payload = torch.zeros((16,2624), dtype=torch.int8, device=x.device)
    payload[:,:2560] = quantized
    payload[:,2560:2564] = scale.contiguous().view(torch.int8).reshape(16,4)
    payload[:,2564:2596] = ids.view(torch.int8).reshape(16,32)
    payload[:,2596] = active.to(torch.int8)
    return payload, scale
