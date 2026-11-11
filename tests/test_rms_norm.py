from __future__ import annotations

import importlib.util
import os
import unittest

from xpu_kernels.bench import _tolerances


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRITON = importlib.util.find_spec("triton") is not None
HAS_TORCH_NPU = importlib.util.find_spec("torch_npu") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class GoldenTests(unittest.TestCase):
    def test_cast_before_gamma_contract_and_output_dtype(self):
        import torch

        from xpu_kernels.golden import rms_norm_golden

        x = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]], dtype=torch.bfloat16)
        w = torch.tensor([0.3, -1.1, 2.0], dtype=torch.bfloat16)
        got = rms_norm_golden(x, w, 1e-5)
        x32 = x.float()
        expected_norm = (x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)
        expected = (expected_norm * w).to(x.dtype)
        self.assertEqual(got.dtype, x.dtype)
        torch.testing.assert_close(got, expected)

    def test_invalid_weight_contract_fails_early(self):
        import torch

        from xpu_kernels import rms_norm

        x = torch.ones((2, 8), dtype=torch.float32)
        with self.assertRaises(ValueError):
            rms_norm(x, torch.ones(7))

    def test_cpu_dispatch_refuses_silent_reference_fallback(self):
        import torch

        from xpu_kernels import rms_norm

        x = torch.ones((2, 8), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "no implementation"):
            rms_norm(x, torch.ones(8))


def _available(device: str) -> bool:
    if os.environ.get("XPU_KERNELS_TEST_DEVICE") not in (None, device):
        return False
    if not HAS_TORCH:
        return False
    import torch

    if device == "cuda":
        return bool(torch.cuda.is_available()) and HAS_TRITON
    if not HAS_TORCH_NPU:
        return False
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


@unittest.skipUnless(_available("cuda"), "CUDA device or Triton is unavailable")
class GpuKernelTests(unittest.TestCase):
    def test_multishape_multidtype_against_golden(self):
        self._run("cuda")

    def _run(self, device):
        import torch

        from xpu_kernels import rms_norm
        from xpu_kernels.golden import rms_norm_golden

        for rows, width in ((1, 257), (4, 4096), (16, 8193)):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                x = torch.randn(rows, width, device=device, dtype=dtype)
                weight = torch.randn(width, device=device, dtype=dtype)
                result = rms_norm(x, weight)
                reference = rms_norm_golden(x, weight)
                torch.cuda.synchronize()
                rtol, atol = _tolerances(torch, dtype)
                torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)

    def test_width_one_empty_rows_zero_values_3d_out_and_non_default_stream(self):
        self._boundary_cases("cuda")

    def _boundary_cases(self, device):
        import torch

        from xpu_kernels import rms_norm
        from xpu_kernels.golden import rms_norm_golden

        module = torch.cuda if device == "cuda" else torch.npu
        stream = module.Stream()
        with module.stream(stream):
            x = torch.zeros((2, 3, 1), device=device, dtype=torch.float32)
            weight = torch.tensor([2.0], device=device, dtype=torch.float32)
            out = torch.empty_like(x)
            actual = rms_norm(x, weight, out=out)
            reference = rms_norm_golden(x, weight)
            self.assertIs(actual, out)
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

            empty = torch.empty((0, 3, 1), device=device, dtype=torch.float32)
            empty_out = torch.empty_like(empty)
            self.assertIs(rms_norm(empty, weight, out=empty_out), empty_out)
        stream.synchronize()


@unittest.skipUnless(_available("npu"), "NPU device or torch_npu is unavailable")
class NpuKernelTests(unittest.TestCase):
    def test_multishape_multidtype_against_golden(self):
        import torch
        import torch_npu  # noqa: F401

        from xpu_kernels import rms_norm
        from xpu_kernels.golden import rms_norm_golden

        for rows, width in ((1, 257), (4, 4096), (16, 8193)):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                x = torch.randn(rows, width, device="npu", dtype=dtype)
                weight = torch.randn(width, device="npu", dtype=dtype)
                result = rms_norm(x, weight)
                reference = rms_norm_golden(x, weight)
                torch.npu.synchronize()
                rtol, atol = _tolerances(torch, dtype)
                torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)

    def test_width_one_empty_rows_zero_values_3d_out_and_non_default_stream(self):
        self._boundary_cases("npu")

    def _boundary_cases(self, device):
        import torch
        import torch_npu  # noqa: F401

        from xpu_kernels import rms_norm
        from xpu_kernels.golden import rms_norm_golden

        module = torch.npu
        stream = module.Stream()
        with module.stream(stream):
            x = torch.zeros((2, 3, 1), device=device, dtype=torch.float32)
            weight = torch.tensor([2.0], device=device, dtype=torch.float32)
            out = torch.empty_like(x)
            actual = rms_norm(x, weight, out=out)
            reference = rms_norm_golden(x, weight)
            self.assertIs(actual, out)
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

            empty = torch.empty((0, 3, 1), device=device, dtype=torch.float32)
            empty_out = torch.empty_like(empty)
            self.assertIs(rms_norm(empty, weight, out=empty_out), empty_out)
        stream.synchronize()


if __name__ == "__main__":
    unittest.main()
