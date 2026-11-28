"""Fixed-capacity MoE fused_owner_combine; explicit NPU execution, no reference fallback."""
from ._moe import tensors, outputs, expert_range

def _validate(received_slots, topk_ids, topk_weights, active, *, npu=False):
    import torch
    values = {'received_slots': received_slots, 'topk_ids': topk_ids, 'topk_weights': topk_weights, 'active': active}
    tensors(values, {'received_slots': ((2,16,8,2560), torch.bfloat16), 'topk_ids': ((16,8), torch.int32), 'topk_weights': ((16,8), torch.float32), 'active': ((16,), torch.bool)}, npu=npu)
    return values

def fused_owner_combine(received_slots, topk_ids, topk_weights, active, *, out=None):
    import torch
    values = _validate(received_slots, topk_ids, topk_weights, active, npu=True)
    result = outputs(out, (((16,2560), torch.bfloat16),), tuple(values.values()))
    from .backends.fused_owner_combine_ascend import fused_owner_combine_npu
    fused_owner_combine_npu(received_slots, topk_ids, topk_weights, active, result[0])
    return result[0]
