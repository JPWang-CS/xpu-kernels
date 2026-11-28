"""Contract, reference and optional device checks for fused_quant_payload."""
import importlib
import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None, 'PyTorch unavailable')
class ContractTests(unittest.TestCase):
    def test_reference_and_device_contract(self):
        from xpu_kernels._moe_workload import inputs
        values=inputs('fused_quant_payload','cpu')
        golden=getattr(importlib.import_module('xpu_kernels.fused_quant_payload_golden'),'fused_quant_payload_golden')
        result=golden(*values)
        outputs=result if isinstance(result,tuple) else (result,)
        self.assertTrue(all(torch.isfinite(v).all() for v in outputs))
        public=getattr(importlib.import_module('xpu_kernels'),'fused_quant_payload')
        with self.assertRaisesRegex(ValueError,'NPU'):
            public(*values)

    def test_shape_rejected(self):
        from xpu_kernels._moe_workload import inputs
        values=list(inputs('fused_quant_payload','cpu'));values[0]=values[0][:-1]
        validate=importlib.import_module('xpu_kernels.fused_quant_payload')._validate
        if 'fused_quant_payload'=='fused_send_slots':
            values[0]=values[0][:,:2559]
        with self.assertRaises(ValueError):
            validate(*values)

    def test_noncontiguous_rejected(self):
        from xpu_kernels._moe_workload import inputs
        values=list(inputs('fused_quant_payload','cpu'))
        values[0]=values[0].transpose(-1,-2).contiguous().transpose(-1,-2)
        with self.assertRaises(ValueError):
            importlib.import_module('xpu_kernels.fused_quant_payload')._validate(*values)

    def test_npu_smoke(self):
        try:
            import torch_npu
        except ImportError:
            self.skipTest('NPU runtime unavailable')
        if not torch.npu.is_available():
            self.skipTest('NPU unavailable')
        from xpu_kernels._moe_workload import inputs
        values=inputs('fused_quant_payload','npu')
        actual=getattr(importlib.import_module('xpu_kernels'),'fused_quant_payload')(*values)
        expected=getattr(importlib.import_module('xpu_kernels.fused_quant_payload_golden'),'fused_quant_payload_golden')(*values)
        actual=actual if isinstance(actual,tuple) else (actual,)
        expected=expected if isinstance(expected,tuple) else (expected,)
        torch.npu.synchronize()
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a.cpu(),b.cpu(),rtol=0,atol=0)
