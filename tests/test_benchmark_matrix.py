import unittest
from xpu_kernels.suites import RMSNORM_SHAPES, TAIL_SHAPES
from xpu_kernels.bench import attach_existing_baseline

class MatrixTests(unittest.TestCase):
    def test_shape_matrix_includes_model_widths_and_tails(self):
        self.assertEqual(len(RMSNORM_SHAPES), 48)
        self.assertEqual(len(set(RMSNORM_SHAPES)), 48)
        self.assertTrue({4096,5120,7168,8192,16384}.issubset({s[1] for s in RMSNORM_SHAPES}))
        self.assertTrue({4095,4097,8191,8193,16383,16385}.issubset({s[1] for s in TAIL_SHAPES}))

    def test_history_never_mixes_cuda_and_triton(self):
        import tempfile,json
        from pathlib import Path
        row = dict(schema_version=3,operator="rms_norm",implementation="cuda",device="cuda",hardware="L20X",
                   shape=[128,4096],dtype="float32",measurement_method="torch.cuda.Event.elapsed_time",
                   execution_mode="eager",device_index=0,runtime={"cuda":"12.6","triton":"3.7.1"},
                   torch="2.13",python="3.12",latency_ms_median=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,"old.jsonl").write_text(json.dumps({**row,"implementation":"triton","latency_ms_median":0.001})+"\n")
            result=attach_existing_baseline([row],Path(tmp))[0]
        self.assertIsNone(result["baseline_best_ms"])
