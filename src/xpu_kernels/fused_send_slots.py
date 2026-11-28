"""Fixed-capacity MoE fused_send_slots; explicit NPU execution, no reference fallback."""
from ._moe import tensors, outputs, expert_range

def _validate(down_proj, inverse_index, route_ids, active, expert_start, expert_end, *, npu=False):
    import torch
    expert_range(expert_start, expert_end)
    if not isinstance(down_proj, torch.Tensor) or down_proj.ndim != 2 or not 1 <= down_proj.shape[0] <= 256:
        raise ValueError('down_proj requires 1..256 rows')
    values = {'down_proj': down_proj, 'inverse_index': inverse_index, 'route_ids': route_ids, 'active': active}
    tensors(values, {'down_proj': ((down_proj.shape[0],2560), torch.bfloat16), 'inverse_index': ((256,), torch.int32), 'route_ids': ((32,8), torch.int32), 'active': ((32,), torch.bool)}, npu=npu)
    return values

def fused_send_slots(down_proj, inverse_index, route_ids, active, expert_start, expert_end, *, out=None):
    import torch
    values = _validate(down_proj, inverse_index, route_ids, active, expert_start, expert_end, npu=True)
    result = outputs(out, (((2,16,8,2560), torch.bfloat16),), tuple(values.values()))
    from .backends.fused_send_slots_ascend import fused_send_slots_npu
    fused_send_slots_npu(down_proj, inverse_index, route_ids, active, result[0], expert_start, expert_end)
    return result[0]
