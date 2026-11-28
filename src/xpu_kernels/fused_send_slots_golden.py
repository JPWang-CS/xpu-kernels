"""Independent fixed-slot gather reference."""
def fused_send_slots_golden(down_proj, inverse_index, route_ids, active, expert_start, expert_end):
    import torch
    from .fused_send_slots import _validate
    _validate(down_proj, inverse_index, route_ids, active, expert_start, expert_end)
    local = active[:,None] & (route_ids >= expert_start) & (route_ids < expert_end)
    valid = inverse_index[local.reshape(-1)]
    if ((valid < 0) | (valid >= down_proj.shape[0])).any():
        raise ValueError('active local inverse_index is out of range')
    flat = torch.zeros((256,2560), dtype=down_proj.dtype, device=down_proj.device)
    flat[local.reshape(-1)] = down_proj[valid.long()]
    return flat.reshape(2,16,8,2560)
