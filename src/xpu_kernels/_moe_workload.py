"""Shared reproducible inputs and correctness-first MoE CLI workload."""
import importlib
import json

def inputs(operator, device):
    import torch
    generator = torch.Generator().manual_seed(41)
    rand = lambda shape: torch.randn(shape, generator=generator).bfloat16().to(device)
    ids = (torch.arange(128, dtype=torch.int32).reshape(16,8)*3 % 256).to(device)
    active = (torch.arange(16)%3 != 0).to(device)
    if operator == 'fused_quant_payload':
        return rand((16,2560)), ids, active
    if operator == 'fused_send_slots':
        return (rand((256,2560)), torch.arange(256,dtype=torch.int32).to(device),
                torch.cat((ids,ids),dim=0), torch.cat((active,active)), 0, 128)
    return (rand((2,16,8,2560)), ids,
            torch.full((16,8),1/8,dtype=torch.float32,device=device), active)

def run_cli(operator, args):
    import torch
    from .environment import require_device
    from .cli import _build
    device = getattr(args,'device',None) or ('npu' if args.backend=='npu' else 'cuda')
    if device != 'npu':
        raise ValueError('the migrated MoE kernels provide an NPU backend only')
    if getattr(args,'implementation',None) not in (None,'ascendc','optimized'):
        raise ValueError('MoE implementation must be ascendc or optimized')
    shapes = getattr(args,'shapes',None) or getattr(args,'shape',None)
    if shapes and shapes != [(16,2560)]:
        raise ValueError('MoE CLI uses the fixed logical token shape 16x2560')
    if args.command in ('bench','profile'):
        raise RuntimeError('migrated JIT kernels require device precision 2.1 and GPU dual-benchmark acceptance before performance collection')
    require_device(device)
    _build('npu', getattr(args,'force',False))
    values = inputs(operator,device)
    actual = getattr(importlib.import_module('xpu_kernels'),operator)(*values)
    golden = getattr(importlib.import_module('xpu_kernels.'+operator+'_golden'),operator+'_golden')(*values)
    actual = actual if isinstance(actual,tuple) else (actual,)
    golden = golden if isinstance(golden,tuple) else (golden,)
    torch.npu.synchronize()
    for a,b in zip(actual,golden):
        torch.testing.assert_close(a.cpu(),b.cpu(),rtol=0,atol=0)
    print(json.dumps({'operator':operator,'device':device,'smoke_correctness':'PASS',
                      'formal_precision_21':'NOT_RUN','graph_validation':'NOT_RUN'}))
    return 0
