from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


dispatch_module = importlib.import_module("xpu_kernels.rms_norm")


class ImplementationSelectionTests(unittest.TestCase):
    def _run_dispatch(self, device: str, implementation, *, empty: bool = False):
        output = object()
        x = SimpleNamespace(device=SimpleNamespace(type=device), numel=lambda: 0 if empty else 1)
        weight = object()
        cuda = Mock(return_value=output)
        triton = Mock(return_value=output)
        ascendc = Mock(return_value=output)
        modules = {
            "xpu_kernels.backends.cuda": SimpleNamespace(rms_norm_cuda=cuda),
            "xpu_kernels.backends.gpu": SimpleNamespace(rms_norm_gpu=triton),
            "xpu_kernels.backends.npu": SimpleNamespace(rms_norm_npu=ascendc),
        }
        with patch.object(dispatch_module, "validate_rms_norm", return_value=SimpleNamespace()), patch.dict(sys.modules, modules):
            result = dispatch_module.rms_norm(x, weight, out=output, implementation=implementation)
        self.assertIs(result, output)
        return cuda, triton, ascendc

    def test_cuda_default_and_explicit_choice_use_native_extension(self):
        for implementation in (None, "cuda"):
            with self.subTest(implementation=implementation):
                cuda, triton, ascendc = self._run_dispatch("cuda", implementation)
                cuda.assert_called_once()
                triton.assert_not_called()
                ascendc.assert_not_called()

    def test_triton_remains_explicitly_selectable(self):
        cuda, triton, ascendc = self._run_dispatch("cuda", "triton")
        cuda.assert_not_called()
        triton.assert_called_once()
        ascendc.assert_not_called()

    def test_npu_default_and_ascendc_keep_existing_backend(self):
        for implementation in (None, "ascendc"):
            with self.subTest(implementation=implementation):
                cuda, triton, ascendc = self._run_dispatch("npu", implementation)
                cuda.assert_not_called()
                triton.assert_not_called()
                ascendc.assert_called_once()

    def test_incompatible_selection_is_an_error_including_empty_rows(self):
        for device, implementation in (("cuda", "ascendc"), ("cuda", "bogus"), ("npu", "cuda"), ("npu", "triton")):
            for empty in (False, True):
                with self.subTest(device=device, implementation=implementation, empty=empty), self.assertRaises(ValueError):
                    self._run_dispatch(device, implementation, empty=empty)

    def test_empty_rows_return_out_without_loading_a_kernel(self):
        for device, implementation in (("cuda", None), ("cuda", "cuda"), ("cuda", "triton"), ("npu", "ascendc")):
            with self.subTest(device=device, implementation=implementation):
                kernels = self._run_dispatch(device, implementation, empty=True)
                for kernel in kernels:
                    kernel.assert_not_called()

    def test_implementation_is_keyword_only(self):
        with self.assertRaises(TypeError):
            dispatch_module.rms_norm(None, None, 1e-5, None, "cuda")

    def test_cuda_wheel_sources_resolve_without_importing_torch(self):
        from xpu_kernels.backends import cuda

        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / "xpu_kernels"
            native = package / "_native"
            cuda_sources = native / "csrc" / "cuda"
            cuda_sources.mkdir(parents=True)
            (cuda_sources / "rms_norm.cu").touch()
            (cuda_sources / "rms_norm.cpp").touch()
            module_path = package / "backends" / "cuda.py"
            with patch.object(cuda, "__file__", str(module_path)):
                self.assertEqual(cuda._project_root(), native)
        self.assertTrue(hasattr(cuda._load_extension, "cache_info"))


def _cuda_available() -> bool:
    if os.environ.get("XPU_KERNELS_TEST_DEVICE") not in (None, "cuda"):
        return False
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


HAS_CUDA = _cuda_available()
HAS_TRITON = importlib.util.find_spec("triton") is not None


@unittest.skipUnless(HAS_CUDA, "CUDA device and CUDA-enabled PyTorch are required")
class CudaBackendTests(unittest.TestCase):
    def test_aligned_and_tail_widths_all_dtypes(self):
        import torch

        from xpu_kernels import rms_norm
        from xpu_kernels.bench import _tolerances
        from xpu_kernels.golden import rms_norm_golden

        cases = ((1, 1), (3, 127), (17, 257), (3, 4095), (17, 4097), (3, 8193), (17, 16384))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for rows, width in cases:
                with self.subTest(dtype=dtype, rows=rows, width=width):
                    x = torch.randn(rows, width, device="cuda", dtype=dtype)
                    weight = torch.randn(width, device=x.device, dtype=dtype)
                    out = torch.empty_like(x)
                    result = rms_norm(x, weight, out=out, implementation="cuda")
                    expected = rms_norm_golden(x, weight)
                    self.assertIs(result, out)
                    self.assertEqual(result.device, x.device)
                    self.assertEqual(result.dtype, x.dtype)
                    self.assertEqual(result.shape, x.shape)
                    rtol, atol = _tolerances(torch, dtype)
                    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_cast_before_weight_is_visible_for_low_precision(self):
        import torch

        from xpu_kernels import rms_norm
        from xpu_kernels.bench import _tolerances
        from xpu_kernels.golden import rms_norm_golden

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.linspace(-3.0, 3.0, 257, device="cuda", dtype=torch.float32).to(dtype)
                weight = torch.linspace(0.1, 2.3, 257, device="cuda", dtype=torch.float32).to(dtype)
                expected = rms_norm_golden(x, weight)
                x32 = x.float()
                normalized32 = x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + 1e-5)
                cast_after_multiply = (normalized32 * weight.float()).to(dtype)
                self.assertTrue(bool((expected != cast_after_multiply).any().item()))
                actual = rms_norm(x, weight, implementation="cuda")
                rtol, atol = _tolerances(torch, dtype)
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                self.assertGreater(int((actual != cast_after_multiply).count_nonzero().item()), 0)

    def test_one_dimensional_three_dimensional_zero_rows_and_inplace_out(self):
        import torch

        from xpu_kernels import rms_norm
        from xpu_kernels.golden import rms_norm_golden

        for shape in ((127,), (2, 3, 257), (0, 3, 257)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, device="cuda", dtype=torch.float32)
                weight = torch.randn(shape[-1], device=x.device, dtype=x.dtype)
                expected = rms_norm_golden(x, weight)
                actual = rms_norm(x, weight, out=x, implementation="cuda")
                self.assertIs(actual, x)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_non_default_stream(self):
        import torch

        from xpu_kernels import rms_norm
        from xpu_kernels.golden import rms_norm_golden

        # Warm JIT before checking ordering, so host compilation cannot mask a wrong stream.
        x = torch.zeros((3, 4097), device="cuda", dtype=torch.float32)
        weight = torch.ones(4097, device=x.device, dtype=x.dtype)
        rms_norm(x, weight, implementation="cuda")
        torch.cuda.synchronize()
        stream = torch.cuda.Stream(device=x.device)
        with torch.cuda.stream(stream):
            if hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(1_000_000)
            x.fill_(0.25)
            weight.fill_(1.5)
            output = rms_norm(x, weight, implementation="cuda")
            expected = rms_norm_golden(x, weight)
        stream.synchronize()
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)

    def test_current_device_guard_when_multiple_gpus_exist(self):
        import torch

        if torch.cuda.device_count() < 2:
            self.skipTest("device-guard test requires two visible CUDA devices")
        from xpu_kernels import rms_norm
        from xpu_kernels.golden import rms_norm_golden

        original = torch.cuda.current_device()
        tensor_device = (original + 1) % torch.cuda.device_count()
        x = torch.randn(3, 127, device=f"cuda:{tensor_device}")
        weight = torch.randn(127, device=x.device)
        result = rms_norm(x, weight, implementation="cuda")
        self.assertEqual(torch.cuda.current_device(), original)
        self.assertEqual(result.device, x.device)
        torch.testing.assert_close(result, rms_norm_golden(x, weight), rtol=1e-5, atol=1e-6)

    def test_validation_refuses_dtype_shape_and_device_mismatch(self):
        import torch

        from xpu_kernels import rms_norm

        x = torch.ones(3, 127, device="cuda")
        weight = torch.ones(127, device=x.device)
        with self.assertRaises(TypeError):
            rms_norm(x.double(), weight.double(), implementation="cuda")
        with self.assertRaises(ValueError):
            rms_norm(x, weight, out=torch.empty(127, 3, device=x.device), implementation="cuda")
        with self.assertRaises(ValueError):
            rms_norm(x, weight.cpu(), implementation="cuda")
        with self.assertRaises(ValueError):
            rms_norm(x.t(), torch.ones(3, device=x.device), implementation="cuda")

    @unittest.skipUnless(HAS_TRITON, "Triton is not installed")
    def test_cuda_triton_and_default_match_golden(self):
        import torch

        from xpu_kernels import rms_norm
        from xpu_kernels.bench import _tolerances
        from xpu_kernels.golden import rms_norm_golden

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.randn(3, 4097, device="cuda", dtype=dtype)
            weight = torch.randn(4097, device=x.device, dtype=dtype)
            expected = rms_norm_golden(x, weight)
            for implementation in (None, "cuda", "triton"):
                with self.subTest(dtype=dtype, implementation=implementation):
                    output = rms_norm(x, weight, implementation=implementation)
                    rtol, atol = _tolerances(torch, dtype)
                    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
