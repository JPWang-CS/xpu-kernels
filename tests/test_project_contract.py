from __future__ import annotations

import tempfile
import unittest
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from xpu_kernels.bench import attach_existing_baseline, validate_benchmark_record, _event_samples, _native_baseline
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

    def test_baseline_matches_device_hardware_shape_and_dtype(self):
        import json
        from pathlib import Path

        row = {
            "schema_version": 2, "device": "cuda", "hardware": "L20X", "shape": [16, 4096],
            "dtype": "float16", "measurement_method": "torch.cuda.Event.elapsed_time",
            "execution_mode": "eager", "device_index": 0, "runtime": {"cuda": "12.6", "triton": "3.7.1"},
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
            "torch": "2.13", "python": "3.12", "latency_ms_median": 0.2,
        }
        prior = {**row, "runtime": {"cuda": "12.6", "triton": "3.7.1"}, "latency_ms_median": 0.3}
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "prior.jsonl").write_text(json.dumps(prior) + "\n", encoding="utf-8")
            result = attach_existing_baseline([row], Path(temp))[0]
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
