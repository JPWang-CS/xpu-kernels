from __future__ import annotations

import importlib.util
import os
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


def _npu_available():
    if not HAS_TORCH or os.environ.get("XPU_KERNELS_TEST_DEVICE") != "npu":
        return False
    if importlib.util.find_spec("torch_npu") is None:
        return False
    try:
        import torch
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class MhcGoldenTests(unittest.TestCase):
    def test_matches_independent_autograd_3d_and_4d(self):
        import torch

        from xpu_kernels.mhc_post_backward import mhc_post_backward_golden

        for shape in ((7, 4, 5), (2, 3, 4, 5)):
            lead, d = shape[:-2], shape[-1]
            x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            hr = torch.randn((*lead, 4, 4), dtype=torch.float64, requires_grad=True)
            ho = torch.randn((*lead, d), dtype=torch.float64, requires_grad=True)
            hp = torch.randn((*lead, 4), dtype=torch.float64, requires_grad=True)
            gy = torch.randn(shape, dtype=torch.float64)
            y = torch.einsum("...ij,...id->...jd", hr, x) + hp.unsqueeze(-1) * ho.unsqueeze(-2)
            grads = torch.autograd.grad((y * gy).sum(), (x, hr, ho, hp))
            got = mhc_post_backward_golden(gy, x.detach(), hr.detach(), ho.detach(), hp.detach())
            for actual, expected in zip(got, grads):
                torch.testing.assert_close(actual.double(), expected.to(actual.dtype).double())

    def test_rejects_bad_inputs_and_cpu_dispatch(self):
        import torch

        from xpu_kernels.mhc_post_backward import mhc_post_backward

        x = torch.ones((8, 4, 3), dtype=torch.float16)
        gy, hr = torch.ones_like(x), torch.ones((8, 4, 4), dtype=torch.float32)
        ho, hp = torch.ones((8, 3), dtype=x.dtype), torch.ones((8, 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "h_res"):
            mhc_post_backward(gy, x, hr.half(), ho, hp)
        with self.assertRaisesRegex(ValueError, "grad_y"):
            mhc_post_backward(gy.float(), x, hr, ho, hp)
        with self.assertRaisesRegex(TypeError, "out"):
            mhc_post_backward(gy, x, hr, ho, hp, out=[])
        with self.assertRaisesRegex(RuntimeError, "golden fallback"):
            mhc_post_backward(gy, x, hr, ho, hp)


@unittest.skipUnless(_npu_available(), "set XPU_KERNELS_TEST_DEVICE=npu and provide an available torch_npu device")
class MhcNpuKernelTests(unittest.TestCase):
    def _check(self, shape, dtype, non_default_stream=False):
        import torch
        import torch_npu  # noqa: F401

        from xpu_kernels.mhc_post_backward import mhc_post_backward, mhc_post_backward_golden

        lead, d = shape[:-2], shape[-1]
        x = torch.randn(shape, device="npu", dtype=dtype)
        gy = torch.randn_like(x)
        hr = torch.randn((*lead, 4, 4), device="npu", dtype=torch.float32)
        ho = torch.randn((*lead, d), device="npu", dtype=dtype)
        hp = torch.randn((*lead, 4), device="npu", dtype=torch.float32)
        outs = (torch.empty_like(x), torch.empty_like(hr), torch.empty_like(ho), torch.empty_like(hp))
        module = torch.npu
        stream = module.Stream() if non_default_stream else None
        if stream is not None:
            stream.wait_stream(module.current_stream())
        if stream is None:
            actual = mhc_post_backward(gy, x, hr, ho, hp, out=outs)
        else:
            with module.stream(stream):
                actual = mhc_post_backward(gy, x, hr, ho, hp, out=outs)
                actual = mhc_post_backward(gy, x, hr, ho, hp, out=outs)
        self.assertTrue(all(a is b for a, b in zip(actual, outs)))
        self.assertIsInstance(actual, tuple)
        expected = mhc_post_backward_golden(gy, x, hr, ho, hp)
        module.synchronize() if stream is None else stream.synchronize()
        rtol, atol = (1e-2, 2e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)
        torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol)
        torch.testing.assert_close(actual[2], expected[2], rtol=rtol, atol=atol)
        torch.testing.assert_close(actual[1], expected[1], rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(actual[3], expected[3], rtol=3e-3, atol=3e-3)

    def test_three_shapes_two_dtypes(self):
        import torch

        for shape in ((512, 4, 2560), (513, 4, 257), (1024, 4, 5120)):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(shape=shape, dtype=dtype):
                    self._check(shape, dtype)

    def test_batched_shape_non_default_stream_and_out_reuse(self):
        import torch

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._check((2, 256, 4, 257), dtype, non_default_stream=True)
        self._check((2, 1024, 4, 6144), torch.bfloat16, non_default_stream=True)


if __name__ == "__main__":
    unittest.main()
