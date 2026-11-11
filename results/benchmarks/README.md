# Benchmark records

Each `xpu-kernels bench` run writes one JSONL file here. Keep the raw records with the hardware name, device index, shape, dtype, software versions, Event timing method, warmup/batch parameters, per-batch samples, and matching historical best. Golden is recorded as a correctness reference only; it is never treated as the fastest baseline. A native PyTorch or torch_npu result is recorded separately when that API is available.

Only sanitized, compact JSONL and this schema note are intended for Git. Profiler traces, compiler output, and temporary build files remain ignored.
