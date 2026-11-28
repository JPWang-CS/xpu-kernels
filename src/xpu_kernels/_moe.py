"""Metadata validation shared by the fixed-capacity MoE operators."""
from __future__ import annotations

def tensors(values, specs, *, npu=False):
    import torch
    first = next(iter(values.values()))
    if not isinstance(first, torch.Tensor):
        raise TypeError('inputs must be torch tensors')
    for name, value in values.items():
        shape, dtype = specs[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor')
        if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
            raise ValueError(f'{name} requires shape {shape}, dtype {dtype}')
        if not value.is_contiguous() or value.device != first.device:
            raise ValueError('all inputs must be contiguous on the same device')
    if npu and first.device.type != 'npu':
        raise ValueError('the fused implementation requires NPU tensors')

def expert_range(start, end):
    if type(start) is not int or type(end) is not int or start not in (0, 128) or end != start + 128:
        raise ValueError('expert range must be [0,128) or [128,256)')

def outputs(out, specs, inputs):
    import torch
    first = inputs[0]
    if out is None:
        return tuple(torch.empty(shape, dtype=dtype, device=first.device) for shape, dtype in specs)
    if not isinstance(out, tuple) or len(out) != len(specs):
        raise ValueError(f'out must be a tuple of {len(specs)} tensors')
    for i, (value, (shape, dtype)) in enumerate(zip(out, specs)):
        tensors({'out': value}, {'out': (shape, dtype)})
        if value.device != first.device:
            raise ValueError('outputs must be on the input device')
        if any(torch._C._overlaps(value, other) for other in (*inputs, *out[:i])):
            raise ValueError('outputs must not overlap inputs or each other')
    return out
