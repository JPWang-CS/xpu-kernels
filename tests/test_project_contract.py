from __future__ import annotations

import tempfile
import unittest
import os
import sys
import importlib.util
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from xpu_kernels.bench import (
    _event_samples,
    _benchmark_out,
    _make_kernel_call,
    _native_baseline,
    _native_rms_norm_reference,
    attach_existing_baseline,
    validate_benchmark_record,
)
from xpu_kernels.cli import _csv_shapes, _parser, _profile_command, _profile_subprocess_env, _validate_test_report


class ProjectContractTests(unittest.TestCase):
    def test_cann_version_reads_toolkit_install_record_and_build(self):
        from xpu_kernels.bench import _cann_version

        with tempfile.TemporaryDirectory() as temp:
            info = Path(temp) / "aarch64-linux" / "ascend_toolkit_install.info"
            info.parent.mkdir()
            info.write_text("version=9.2.0\ninnerversion=V100R001C12B096\n", encoding="utf-8")
            with patch.dict(os.environ, {"ASCEND_HOME_PATH": temp, "CANN_VERSION": ""}):
                self.assertEqual(_cann_version(), "9.2.0 (V100R001C12B096)")

    def test_cli_exposes_all_required_commands(self):
        parser = _parser()
        choices = parser._subparsers._group_actions[0].choices
        for command in ("check-env", "build", "example", "test", "bench", "profile"):
            with self.subTest(command=command):
                self.assertIn(command, choices)

    def test_shape_parser_supports_multiple_cases_and_rejects_bad_width(self):
        self.assertEqual(_csv_shapes("1x257,16x4096"), [(1, 257), (16, 4096)])
        with self.assertRaises(Exception):
            _csv_shapes("4x0")

    def test_benchmark_schema_accepts_a_complete_record(self):
        record = {
            "schema_version": 2,
            "timestamp_utc": "2026-11-11T00:00:00+00:00",
            "device": "cuda",
            "hardware": "L20X",
            "shape": [16, 4096],
            "dtype": "bfloat16",
            "device_index": 0,
            "measurement_method": "torch.cuda.Event.elapsed_time",
            "measurement_scope": "device_stream_elapsed_per_call",
            "execution_mode": "eager",
            "graph_capture": False,
            "warmup_iterations": 2,
            "batches": 3,
            "iterations_per_batch": 10,
            "latency_ms_median": 0.25,
            "latency_ms_samples": [0.24, 0.25, 0.26],
            "correctness_reference": "xpu_kernels.golden.rms_norm_golden (not a performance baseline)",
            "native_baseline": {"status": "not_requested"},
            "baseline_best_ms": 0.3,
            "speedup_vs_baseline": 1.2,
            "python": "3.12.0",
            "torch": "2.13.0",
            "runtime": {"cuda": "12.6", "triton": "3.7.1", "compute_capability": [9, 0]},
        }
        validate_benchmark_record(record)

    def test_benchmark_schema_rejects_incomplete_samples(self):
        record = {
            "schema_version": 2,
            "timestamp_utc": "2026-11-11T00:00:00+00:00",
            "device": "cuda",
            "hardware": "L20X",
            "shape": [16, 4096],
            "dtype": "bfloat16",
            "device_index": 0,
            "measurement_method": "torch.cuda.Event.elapsed_time",
            "measurement_scope": "device_stream_elapsed_per_call",
            "execution_mode": "eager",
            "graph_capture": False,
            "warmup_iterations": 2,
            "batches": 3,
            "iterations_per_batch": 10,
            "latency_ms_median": 0.25,
            "latency_ms_samples": [0.24],
            "correctness_reference": "xpu_kernels.golden.rms_norm_golden (not a performance baseline)",
            "native_baseline": {"status": "not_requested"},
            "baseline_best_ms": None,
            "speedup_vs_baseline": None,
            "python": "3.12.0",
            "torch": "2.13.0",
            "runtime": {"cuda": "12.6", "triton": "3.7.1", "compute_capability": [9, 0]},
        }
        with self.assertRaises(ValueError):
            validate_benchmark_record(record)

    def test_schema4_native_record_identifies_output_allocation(self):
        record = {
            "schema_version": 4, "operator": "rms_norm", "implementation": "native",
            "timestamp_utc": "2026-11-11T00:00:00+00:00", "device": "npu", "device_index": 0,
            "hardware": "Ascend950DT", "shape": [4, 128], "dtype": "bfloat16",
            "measurement_method": "torch.npu.Event.elapsed_time",
            "measurement_scope": "device_stream_elapsed_per_call", "execution_mode": "eager",
            "graph_capture": False, "warmup_iterations": 1, "batches": 2,
            "iterations_per_batch": 3, "latency_ms_median": 0.1,
            "latency_ms_samples": [0.09, 0.11],
            "correctness_reference": "xpu_kernels.bench._native_rms_norm_reference",
            "correctness": {"status": "PASS", "max_abs_error": 0.0, "rtol": 0.01, "atol": 0.0078125},
            "output_allocation": {
                "strategy": "operator_allocates_output_per_call",
                "api_call": "torch_npu.npu_rms_norm",
                "event_timing_note": "torch.npu.Event may include CPU submission gaps; host output allocation is not isolated.",
            },
            "native_baseline": {"status": "self"}, "baseline_best_ms": None,
            "speedup_vs_baseline": None, "python": "3.11", "torch": "2.9",
            "runtime": {"torch_npu": "2.9", "cann_version": "9.2"},
        }
        validate_benchmark_record(record)

    def test_delayed_implementation_calls_keep_their_own_output_and_backend(self):
        calls = []

        def dispatch(x, weight, out, implementation):
            calls.append((implementation, out))
            return out

        first_out, second_out = object(), object()
        first = _make_kernel_call(dispatch, "ascendc", "x", "w", first_out)
        second = _make_kernel_call(dispatch, "other", "x", "w", second_out)
        first()
        second()
        self.assertEqual(calls, [("ascendc", first_out), ("other", second_out)])

    def test_native_npu_pair_allocates_output_per_call_without_changing_default_or_gpu(self):
        allocated = object()
        fake_torch = SimpleNamespace(empty_like=lambda x: allocated)
        x = object()
        self.assertIsNone(_benchmark_out(fake_torch, x, "npu", True))
        self.assertIs(_benchmark_out(fake_torch, x, "npu", False), allocated)
        self.assertIs(_benchmark_out(fake_torch, x, "cuda", True), allocated)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
    def test_native_reference_applies_gamma_before_output_cast(self):
        import torch

        x = torch.tensor([[1.1, 0.7, -1.3]], dtype=torch.bfloat16)
        weight = torch.tensor([1.3, -0.8, 0.6], dtype=torch.bfloat16)
        x32, w32 = x.float(), weight.float()
        fused = (x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + 1e-5) * w32).to(x.dtype)
        native_reference = _native_rms_norm_reference(torch, x, weight, 1e-5)
        torch.testing.assert_close(native_reference, fused)

    def test_baseline_matches_device_hardware_shape_and_dtype(self):
        import json
        from pathlib import Path

        row = {
            "schema_version": 2, "device": "cuda", "hardware": "L20X", "shape": [16, 4096],
            "dtype": "float16", "measurement_method": "torch.cuda.Event.elapsed_time",
            "execution_mode": "eager", "device_index": 0, "runtime": {"cuda": "12.6", "triton": "3.7.1"},
            "warmup_iterations": 5, "batches": 10, "iterations_per_batch": 20,
            "torch": "2.13", "python": "3.12", "latency_ms_median": 0.2,
        }
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "old.jsonl").write_text(json.dumps({**row, "latency_ms_median": 0.3}) + "\n", encoding="utf-8")
            result = attach_existing_baseline([row], Path(temp))[0]
        self.assertEqual(result["baseline_best_ms"], 0.3)
        self.assertAlmostEqual(result["speedup_vs_baseline"], 1.5)

    def test_baseline_requires_matching_software_versions(self):
        import json

        row = {
            "schema_version": 2, "device": "cuda", "device_index": 0, "hardware": "L20X",
            "shape": [2, 512], "dtype": "float16", "measurement_method": "torch.cuda.Event.elapsed_time",
            "execution_mode": "eager", "runtime": {"cuda": "12.6", "triton": "3.8.0"},
            "warmup_iterations": 5, "batches": 10, "iterations_per_batch": 20,
            "torch": "2.13", "python": "3.12", "latency_ms_median": 0.2,
        }
        prior = {**row, "runtime": {"cuda": "12.6", "triton": "3.7.1"}, "latency_ms_median": 0.3}
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "prior.jsonl").write_text(json.dumps(prior) + "\n", encoding="utf-8")
            result = attach_existing_baseline([row], Path(temp))[0]
        self.assertIsNone(result["baseline_best_ms"])

    def test_baseline_requires_matching_allocation_timing_and_correctness(self):
        import json

        row = {
            "schema_version": 4, "operator": "rms_norm", "implementation": "ascendc",
            "device": "npu", "device_index": 0, "hardware": "Ascend950DT",
            "shape": [16, 4096], "dtype": "bfloat16",
            "measurement_method": "torch.npu.Event.elapsed_time", "execution_mode": "eager",
            "warmup_iterations": 5, "batches": 10, "iterations_per_batch": 20,
            "runtime": {"cann_version": "9.2", "torch_npu": "2.9"},
            "torch": "2.9", "python": "3.11", "latency_ms_median": 0.2,
            "correctness": {"status": "PASS"},
            "output_allocation": {
                "strategy": "out_none_per_call", "api_call": "rms_norm(out=None)",
                "event_timing_note": "CPU submission gaps may appear in the NPU Event interval.",
            },
        }
        legacy_preallocated = {
            **row, "schema_version": 3, "latency_ms_median": 0.3,
        }
        legacy_preallocated.pop("output_allocation")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "old.jsonl")
            path.write_text(json.dumps(legacy_preallocated) + "\n", encoding="utf-8")
            result = attach_existing_baseline([dict(row)], Path(temp))[0]
            self.assertIsNone(result["baseline_best_ms"])

            matching = {
                **row, "latency_ms_median": 0.4,
                "output_allocation": dict(row["output_allocation"]),
            }
            path.write_text(json.dumps(matching) + "\n", encoding="utf-8")
            result = attach_existing_baseline([dict(row)], Path(temp))[0]
            self.assertEqual(result["baseline_best_ms"], 0.4)

            for key, mismatch in (
                ("warmup_iterations", 6), ("batches", 11), ("iterations_per_batch", 21),
            ):
                with self.subTest(parameter=key):
                    prior = {**matching, key: mismatch}
                    path.write_text(json.dumps(prior) + "\n", encoding="utf-8")
                    result = attach_existing_baseline([dict(row)], Path(temp))[0]
                    self.assertIsNone(result["baseline_best_ms"])

            failed_correctness = {**matching, "correctness": {"status": "FAIL"}}
            path.write_text(json.dumps(failed_correctness) + "\n", encoding="utf-8")
            result = attach_existing_baseline([dict(row)], Path(temp))[0]
            self.assertIsNone(result["baseline_best_ms"])

    def test_event_timer_batches_device_intervals_without_wall_clock(self):
        calls = []

        class Event:
            def __init__(self, enable_timing):
                self.end = False

            def record(self, stream):
                self.end = True

            def synchronize(self):
                pass

            def elapsed_time(self, end):
                return 8.0

        Module = SimpleNamespace(Event=Event)

        class Stream:
            def synchronize(self):
                pass

        samples = _event_samples(lambda: calls.append(1), Module, Stream(), 2, 3, 4)
        self.assertEqual(len(calls), 14)
        self.assertEqual(samples, [2.0, 2.0, 2.0])

    def test_msprof_uses_application_option_without_extra_python_token(self):
        command = _profile_command(
            "msprof", "msprof", Path("results/profiles"),
            ["/usr/bin/python", "-m", "xpu_kernels.profile_workload", "--device", "npu"],
            "--output --application",
        )
        self.assertEqual(command[:2], ["msprof", f"--output={Path('results/profiles')}"])
        self.assertTrue(command[2].startswith("--application=/usr/bin/python -m "))

    def test_profile_child_pythonpath_includes_venv_and_preserves_existing_entries(self):
        from xpu_kernels import cli

        original_pythonpath = os.environ.get("PYTHONPATH")
        with patch.object(cli.sys, "path", ["", "/project/src", "/venv/lib/python3.11/site-packages"]), patch.dict(
            os.environ, {"PYTHONPATH": "/existing/imports"}, clear=False
        ):
            env = _profile_subprocess_env()
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep), [
            "/project/src", "/venv/lib/python3.11/site-packages", "/existing/imports",
        ])
        self.assertEqual(os.environ.get("PYTHONPATH"), original_pythonpath)

    def test_ncu_profile_is_narrow_and_does_not_control_clocks(self):
        command = _profile_command("ncu", "ncu", Path("profiles"), ["python", "-m", "work"])
        self.assertIn("--launch-count", command)
        self.assertIn("regex:_rms_norm_kernel", command)
        self.assertIn("SpeedOfLight", command)
        self.assertIn("LaunchStats", command)
        self.assertEqual(command[command.index("--clock-control") + 1], "none")
        self.assertNotIn("full", command)

    def test_environment_reports_configured_private_cxx(self):
        from xpu_kernels.environment import inspect_environment

        with patch.dict(os.environ, {"CXX": "/private/gcc13/bin/g++"}), patch(
            "xpu_kernels.environment.shutil.which", side_effect=lambda name: name if name == "/private/gcc13/bin/g++" else None
        ):
            result = inspect_environment("npu")
        self.assertEqual(result["cxx"], "/private/gcc13/bin/g++")
        self.assertEqual(result["cxx_resolved"], "/private/gcc13/bin/g++")

    def test_torch_npu_baseline_uses_native_29_signature(self):
        calls = []

        def native(x, weight, epsilon=1e-6):
            calls.append((x, weight, epsilon))
            return "normalized", "rstd"

        torch_npu = SimpleNamespace(npu_rms_norm=native)
        fake_torch = SimpleNamespace(ops=SimpleNamespace())
        x, weight = SimpleNamespace(shape=(4, 128)), object()
        with patch.dict(sys.modules, {"torch_npu": torch_npu}):
            call, metadata = _native_baseline("npu", fake_torch, x, weight, 1e-5)
            result = call()
        self.assertEqual(result, "normalized")
        self.assertEqual(calls, [(x, weight, 1e-5)])
        self.assertEqual(metadata["name"], "torch_npu.npu_rms_norm")

    def test_schema_rejects_host_wall_clock_measurement(self):
        record = {
            "schema_version": 2,
            "timestamp_utc": "2026-11-11T00:00:00+00:00",
            "device": "cuda",
            "device_index": 0,
            "hardware": "L20X",
            "shape": [1, 128],
            "dtype": "float16",
            "measurement_method": "time.perf_counter",
            "measurement_scope": "host_e2e",
            "execution_mode": "eager",
            "graph_capture": False,
            "warmup_iterations": 1,
            "batches": 1,
            "iterations_per_batch": 1,
            "latency_ms_median": 1.0,
            "latency_ms_samples": [1.0],
            "correctness_reference": "golden",
            "native_baseline": {"status": "not_requested"},
            "baseline_best_ms": None,
            "speedup_vs_baseline": None,
            "python": "3.11",
            "torch": "2.9",
            "runtime": {},
        }
        with self.assertRaises(ValueError):
            validate_benchmark_record(record)

    def test_test_command_rejects_zero_tests_and_skipped_device_case(self):
        with self.assertRaises(RuntimeError):
            _validate_test_report("Ran 0 tests in 0.0s\nOK", "npu")
        with self.assertRaises(RuntimeError):
            _validate_test_report("test_multishape_multidtype_against_golden (test_rms_norm.NpuKernelTests) ... skipped\nRan 1 test in 0.1s\nOK", "npu")

    def test_test_report_accepts_compile_logs_between_case_and_status(self):
        output = "\n".join((
            "test_multishape_multidtype_against_golden (test_rms_norm.NpuKernelTests.test_multishape_multidtype_against_golden) ...",
            "[1/2] c++ -c xpu_kernels_ascend.cpp",
            "[2/2] c++ -shared xpu_kernels_ascend.so",
            "ok",
            "Ran 20 tests in 1.2s",
            "OK",
        ))
        _validate_test_report(output, "npu")

    def test_installed_package_resolves_bundled_native_resources(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from xpu_kernels.backends import npu

        with TemporaryDirectory() as temp:
            package = Path(temp) / "xpu_kernels"
            native = package / "_native"
            (native / "csrc" / "ascend").mkdir(parents=True)
            (native / "csrc" / "ascend" / "rms_norm.cpp").touch()
            (native / "kernels" / "include" / "xpu_kernels").mkdir(parents=True)
            (native / "kernels" / "include" / "xpu_kernels" / "rms_norm_ascend.hpp").touch()
            installed_module = package / "backends" / "npu.py"
            installed_module.parent.mkdir(parents=True)
            with patch.object(npu, "__file__", str(installed_module)):
                self.assertEqual(npu._project_root(), native)


if __name__ == "__main__":
    unittest.main()
