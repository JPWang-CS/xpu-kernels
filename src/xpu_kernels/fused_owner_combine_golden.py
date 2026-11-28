"""Independent sequential FP32 reference and separate FP64 mathematical golden."""
def fused_owner_combine_golden(received_slots, topk_ids, topk_weights, active, *, fp64=False):
    import torch
    from .fused_owner_combine import _validate
    _validate(received_slots, topk_ids, topk_weights, active)
    if ((topk_ids[active] < 0) | (topk_ids[active] >= 256)).any():
        raise ValueError('active expert IDs must be in [0,256)')
    ids = torch.where(active[:,None], topk_ids, 0)
    dtype = torch.float64 if fp64 else torch.float32
    accumulator = torch.zeros((16,2560), dtype=dtype, device=received_slots.device)
    rows = torch.arange(16, device=received_slots.device)
    for k in range(8):
        selected = received_slots[ids[:,k].long() // 128, rows, k].to(dtype)
        product = selected * topk_weights[:,k,None].to(dtype)
        accumulator = accumulator + product
    result = torch.where(active[:,None], accumulator, 0)
    return result if fp64 else result.bfloat16()
