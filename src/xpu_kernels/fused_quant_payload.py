"""Fixed-capacity MoE fused_quant_payload; fixed-capacity NPU implementation."""
from ._moe import tensors, outputs, expert_range

def _validate(x, topk_ids, active, *, npu=False):
    import torch
    values = {'x': x, 'topk_ids': topk_ids, 'active': active}
    tensors(values, {'x': ((16,2560), torch.bfloat16), 'topk_ids': ((16,8), torch.int32), 'active': ((16,), torch.bool)}, npu=npu)
    return values

def fused_quant_payload(x, topk_ids, active, *, out=None):
    import torch
    values = _validate(x, topk_ids, active, npu=True)
    result = outputs(out, (((16,2624), torch.int8), ((16,), torch.float32)), tuple(values.values()))
    from .backends.fused_quant_payload_ascend import fused_quant_payload_npu
    fused_quant_payload_npu(x, topk_ids, active, *result)
    return result
