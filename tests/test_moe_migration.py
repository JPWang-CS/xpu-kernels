import importlib
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'PyTorch unavailable')
class MoEReferenceTests(unittest.TestCase):
    def test_payload_zero_row_and_layout(self):
        from xpu_kernels.fused_quant_payload_golden import fused_quant_payload_golden
        x=torch.ones(16,2560,dtype=torch.bfloat16)
        ids=torch.arange(128,dtype=torch.int32).reshape(16,8)
        active=torch.ones(16,dtype=torch.bool);active[1]=False;x[2]=0
        payload,scale=fused_quant_payload_golden(x,ids,active)
        self.assertTrue((payload[0,:2560]==127).all())
        self.assertTrue((payload[1]==0).all())
        self.assertEqual(scale[2].item(),0)
        self.assertTrue((payload[:,2597:]==0).all())
        self.assertTrue(torch.equal(payload[:,2560:2564].contiguous().view(torch.float32).flatten(),scale))
        self.assertTrue(torch.equal(payload[0,2564:2596].view(torch.int32),ids[0]))

    def test_send_slots_rank_filter_and_inverse(self):
        from xpu_kernels.fused_send_slots_golden import fused_send_slots_golden
        down=torch.arange(256,dtype=torch.bfloat16)[:,None].expand(256,2560).contiguous()
        inverse=torch.arange(255,-1,-1,dtype=torch.int32)
        ids=torch.arange(256,dtype=torch.int32).reshape(32,8)
        active=torch.ones(32,dtype=torch.bool);active[2]=False
        result=fused_send_slots_golden(down,inverse,ids,active,0,128).reshape(256,2560)
        self.assertEqual(result[0,0].item(),255)
        self.assertTrue((result[16:24]==0).all())
        self.assertTrue((result[128:]==0).all())
        inverse[0]=256
        with self.assertRaises(ValueError):
            fused_send_slots_golden(down,inverse,ids,active,0,128)

    def test_owner_selection_and_inactive_mask(self):
        from xpu_kernels.fused_owner_combine_golden import fused_owner_combine_golden
        recv=torch.ones(2,16,8,2560,dtype=torch.bfloat16);recv[1]=3
        ids=torch.tensor([0,128]*4,dtype=torch.int32).expand(16,8).contiguous()
        weights=torch.full((16,8),.125);active=torch.ones(16,dtype=torch.bool);active[0]=False
        result=fused_owner_combine_golden(recv,ids,weights,active)
        self.assertTrue((result[0]==0).all());self.assertTrue((result[1:]==2).all())
        precise=fused_owner_combine_golden(recv,ids,weights,active,fp64=True)
        self.assertEqual(precise.dtype,torch.float64)
        ids[1,0]=256
        with self.assertRaises(ValueError):
            fused_owner_combine_golden(recv,ids,weights,active)

    def test_cli_registers_kernels(self):
        from xpu_kernels.cli import _parser
        for op in ('fused_quant_payload','fused_send_slots','fused_owner_combine'):
            self.assertEqual(_parser().parse_args(['test','--device','npu','--operator',op]).operator,op)

    def test_sequential_rounding_is_distinct_from_fp64_golden(self):
        from xpu_kernels.fused_owner_combine_golden import fused_owner_combine_golden
        recv=torch.zeros(2,16,8,2560,dtype=torch.bfloat16)
        recv[0,:,0]=2**24;recv[0,:,1]=1;recv[0,:,2]=-(2**24)
        ids=torch.zeros(16,8,dtype=torch.int32)
        weights=torch.ones(16,8);active=torch.ones(16,dtype=torch.bool)
        rounded=fused_owner_combine_golden(recv,ids,weights,active)
        precise=fused_owner_combine_golden(recv,ids,weights,active,fp64=True)
        self.assertTrue((rounded==0).all())
        self.assertTrue((precise==1).all())

    def test_output_alias_and_expert_range_rejected(self):
        from xpu_kernels._moe import outputs, expert_range
        x=torch.ones(16,2560,dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            outputs((x,),(((16,2560),torch.bfloat16),),(x,))
        for bounds in ((True,129),(1,129),(128,257),(256,384)):
            with self.assertRaises(ValueError):
                expert_range(*bounds)

    def test_example_import_has_no_device_side_effect(self):
        path=Path(__file__).resolve().parents[1]/'examples/moe_fixed_capacity.py'
        spec=importlib.util.spec_from_file_location('moe_example',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(callable(module.fixed_capacity_moe))

    def test_no_performance_before_formal_acceptance(self):
        from types import SimpleNamespace
        from xpu_kernels._moe_workload import run_cli
        for command in ('bench','profile'):
            with self.assertRaisesRegex(RuntimeError,'precision 2.1'):
                run_cli('fused_owner_combine',SimpleNamespace(command=command,device='npu'))
