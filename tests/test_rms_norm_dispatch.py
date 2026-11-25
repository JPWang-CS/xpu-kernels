import unittest

from xpu_kernels.rms_norm_dispatch import optimized_rms_norm_scope


class _FakeDevice:
    def __init__(self, device_type):
        self.type = device_type


class _FakeTensor:
    def __init__(self, rows, width, *, dtype="torch.bfloat16", device_type="npu", contiguous=True):
        self.shape = (rows, width)
        self.dtype = dtype
        self.device = _FakeDevice(device_type)
        self._contiguous = contiguous

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def is_contiguous(self):
        return self._contiguous


def _case(rows=2048, width=20480, **kwargs):
    return _FakeTensor(rows, width, **kwargs), _FakeTensor(width, 1, **kwargs)


class OptimizedRmsNormScopeTests(unittest.TestCase):
    def test_exact_supported_ascend950_shapes_are_selected(self):
        for rows, hardware in ((2048, "Ascend950DT"), (4096, "Ascend950PR")):
            with self.subTest(rows=rows, hardware=hardware):
                x, weight = _case(rows=rows)
                self.assertTrue(optimized_rms_norm_scope(x, weight, 1e-5, device_name=hardware))

    def test_adjacent_row_counts_are_rejected(self):
        for rows in (2047, 2049, 4095, 4097):
            with self.subTest(rows=rows):
                x, weight = _case(rows=rows)
                self.assertFalse(optimized_rms_norm_scope(x, weight, device_name="Ascend950DT"))

    def test_width_and_epsilon_must_match_exact_scope(self):
        for width in (20479, 20481, 4096):
            with self.subTest(width=width):
                x, weight = _case(width=width)
                self.assertFalse(optimized_rms_norm_scope(x, weight, device_name="Ascend950DT"))
        x, weight = _case()
        for eps in (1e-5 + 1e-12, 1e-4, 0.0):
            with self.subTest(eps=eps):
                self.assertFalse(optimized_rms_norm_scope(x, weight, eps, device_name="Ascend950DT"))

    def test_dtype_and_layout_must_be_bfloat16_and_contiguous(self):
        for dtype in ("torch.float16", "torch.float32", "torch.int8"):
            with self.subTest(dtype=dtype):
                x, weight = _case(dtype=dtype)
                self.assertFalse(optimized_rms_norm_scope(x, weight, device_name="Ascend950DT"))
        x, weight = _case()
        x._contiguous = False
        self.assertFalse(optimized_rms_norm_scope(x, weight, device_name="Ascend950DT"))
        x, weight = _case()
        weight._contiguous = False
        self.assertFalse(optimized_rms_norm_scope(x, weight, device_name="Ascend950DT"))

    def test_hardware_and_gpu_device_must_be_ascend950_npu(self):
        for hardware in ("Ascend910B", "AscendA3", "NVIDIA H100", "cuda"):
            with self.subTest(hardware=hardware):
                x, weight = _case()
                self.assertFalse(optimized_rms_norm_scope(x, weight, device_name=hardware))
        x, weight = _case(device_type="cuda")
        self.assertFalse(optimized_rms_norm_scope(x, weight, device_name="Ascend950DT"))


if __name__ == "__main__":
    unittest.main()
