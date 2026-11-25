import json
import tempfile
import unittest
from pathlib import Path

from xpu_kernels.bench import (
    NUMERICAL_CONTRACT,
    _execution_path,
    _output_allocation_metadata,
    attach_existing_baseline,
    validate_benchmark_record,
)


class _FakeDevice:
    type = "npu"


class _FakeTensor:
    device = _FakeDevice()
    dtype = "torch.bfloat16"

    def __init__(self, rows=2048, width=20480):
        self.shape = (rows, width)

    def numel(self):
        return self.shape[0] * self.shape[1]

    def is_contiguous(self):
        return True


def _record(**overrides):
    row = {
        "schema_version": 5,
        "operator": "rms_norm",
        "implementation": "auto",
        "numerical_contract": NUMERICAL_CONTRACT,
        "execution_path": "2slot",
        "timestamp_utc": "2026-11-25T00:00:00+00:00",
        "device": "npu",
        "device_index": 0,
        "hardware": "Ascend950DT",
        "shape": [2048, 20480],
        "dtype": "bfloat16",
        "measurement_method": "torch.npu.Event.elapsed_time",
        "measurement_scope": "device_stream_elapsed_per_call",
        "execution_mode": "eager",
        "graph_capture": False,
        "warmup_iterations": 1,
        "batches": 1,
        "iterations_per_batch": 1,
        "latency_ms_median": 1.0,
        "latency_ms_samples": [1.0],
        "correctness_reference": "xpu_kernels.rms_norm_golden.rms_norm_golden",
        "correctness": {"status": "PASS", "max_abs_error": 0.0, "rtol": 1e-2, "atol": 7.8125e-3},
        "native_baseline": {"status": "not_requested"},
        "baseline_best_ms": None,
        "speedup_vs_baseline": None,
        "python": "3.14",
        "torch": "test",
        "runtime": {"torch_npu": "test", "cann_version": "test"},
        "output_allocation": {
            "strategy": "preallocated_y_rstd_per_call",
            "api_call": "rms_norm(out=preallocated, implementation='auto')",
            "event_timing_note": "torch.npu.Event measures the same current stream; Y=preallocated and rstd=per_call, so host allocation time is not isolated.",
        },
    }
    row.update(overrides)
    return row


class RmsNormMetricsTests(unittest.TestCase):
    def test_mainline_out_metadata_describes_internal_allocations_and_copy(self):
        metadata = _output_allocation_metadata("npu", "native", False, "mainline")
        self.assertEqual(metadata["strategy"], "operator_allocates_y_rstd_per_call_copy_to_out")
        self.assertIn("internal Y=per_call", metadata["event_timing_note"])
        self.assertIn("rstd=per_call", metadata["event_timing_note"])
        self.assertIn("copied to the destination out", metadata["event_timing_note"])

        optimized = _output_allocation_metadata("npu", "auto", False, "2slot")
        self.assertEqual(optimized["strategy"], "preallocated_y_rstd_per_call")

        native_reference = _output_allocation_metadata("npu", "native", True, "mainline")
        self.assertEqual(native_reference["strategy"], "operator_allocates_y_rstd_per_call")

    def test_execution_path_records_mainline_and_slot_variants(self):
        x = _FakeTensor()
        weight = _FakeTensor(1, 20480)
        self.assertEqual(_execution_path("npu", "native", x, weight, 1e-5, "Ascend950DT"), "mainline")
        self.assertEqual(_execution_path("npu", "auto", x, weight, 1e-5, "Ascend950DT"), "2slot")
        x = _FakeTensor(4096, 20480)
        self.assertEqual(_execution_path("npu", "ascendc", x, weight, 1e-5, "Ascend950DT"), "3slot")
        self.assertEqual(_execution_path("npu", "auto", x, weight, 1e-5, "Ascend910B"), "mainline")

    def test_schema5_requires_current_contract_and_path(self):
        validate_benchmark_record(_record())
        with self.assertRaises(ValueError):
            validate_benchmark_record(_record(numerical_contract="old-contract"))
        with self.assertRaises(ValueError):
            validate_benchmark_record(_record(execution_path="standalone"))

    def test_new_contract_does_not_attach_old_schema_baseline(self):
        current = _record()
        old = _record()
        old.pop("numerical_contract")
        old.pop("execution_path")
        old["schema_version"] = 4
        old["output_allocation"] = {
            "strategy": "preallocated_out",
            "api_call": "rms_norm(out=preallocated)",
            "event_timing_note": "torch.npu.Event measures device-stream elapsed time; CPU submission gaps may appear in the interval, and host output-allocation time is not isolated.",
        }
        old["latency_ms_median"] = 0.5
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "old.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
            result = attach_existing_baseline([current], Path(temp))[0]
        self.assertIsNone(result["baseline_best_ms"])
        self.assertIsNone(result["speedup_vs_baseline"])

    def test_schema5_baseline_requires_matching_allocation_strategy(self):
        current = _record()
        prior = _record(
            latency_ms_median=0.5,
            output_allocation={
                "strategy": "operator_allocates_y_rstd_per_call",
                "api_call": "torch_npu.npu_rms_norm",
                "event_timing_note": "torch.npu.Event measures the same current stream; Y=per_call and rstd=per_call, so host allocation time is not isolated.",
            },
        )
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "different-allocation.jsonl").write_text(json.dumps(prior) + "\n", encoding="utf-8")
            result = attach_existing_baseline([current], Path(temp))[0]
        self.assertIsNone(result["baseline_best_ms"])

        prior["output_allocation"] = dict(current["output_allocation"])
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "same-allocation.jsonl").write_text(json.dumps(prior) + "\n", encoding="utf-8")
            result = attach_existing_baseline([current], Path(temp))[0]
        self.assertEqual(result["baseline_best_ms"], 0.5)


if __name__ == "__main__":
    unittest.main()
