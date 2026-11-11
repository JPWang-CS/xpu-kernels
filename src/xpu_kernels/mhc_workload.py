"""Correctness, repeated device timing and profiler entry for mHC backward."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

SHAPES = tuple((t, 4, d) for t in (2048, 8192, 32768) for d in (2560, 5120, 6144, 7168)) + (
    (2, 1024, 4, 6144), (3, 1024, 4, 6144), (2, 4096, 4, 6144), (3, 8192, 4, 6144))


def inputs(shape, dtype_name):
    import torch
    from .environment import require_device
    require_device("npu")
    shape = tuple(shape)
    if len(shape) == 2:
        shape = (shape[0], 4, shape[1])
    if len(shape) not in (3, 4) or shape[-2] != 4:
        raise ValueError("mHC shape must be T x 4 x D or B x S x 4 x D")
    torch.manual_seed(5341)
    dtype = getattr(torch, dtype_name)
    lead, width = shape[:-2], shape[-1]
    x = torch.randn(shape, device="npu", dtype=dtype)
    return (torch.randn_like(x), x,
            torch.randn((*lead, 4, 4), device="npu", dtype=torch.float32),
            torch.randn((*lead, width), device="npu", dtype=dtype),
            torch.randn((*lead, 4), device="npu", dtype=torch.float32))


def check_case(values, implementation="auto"):
    import torch
    from .mhc_post_backward import mhc_post_backward, mhc_post_backward_golden
    actual = mhc_post_backward(*values,implementation=implementation)
    # Sample boundary and interior tokens; full-size FP64 CPU copies can exceed RAM.
    tokens = values[1].numel() // (4 * values[1].shape[-1])
    indices = sorted({0, 1, tokens // 3, tokens // 2, tokens - 2, tokens - 1})
    # h_res has two feature axes, h_out/h_post have one.
    gy, x, hr, ho, hp = values
    small = (gy.reshape(tokens,4,-1)[indices].contiguous(), x.reshape(tokens,4,-1)[indices].contiguous(),
             hr.reshape(tokens,4,4)[indices].contiguous(), ho.reshape(tokens,-1)[indices].contiguous(),
             hp.reshape(tokens,4)[indices].contiguous())
    reference = mhc_post_backward_golden(*small)
    actual_sample = (actual[0].reshape(tokens,4,-1)[indices], actual[1].reshape(tokens,4,4)[indices],
                     actual[2].reshape(tokens,-1)[indices], actual[3].reshape(tokens,4)[indices])
    torch.npu.synchronize()
    errors = []
    for i, (a, b) in enumerate(zip(actual_sample, reference)):
        rtol, atol = ((1e-2, 2e-2) if a.dtype == torch.bfloat16 else (3e-3, 3e-3)) if i in (0, 2) else (3e-3, 3e-3)
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        errors.append({"output": ("grad_x", "grad_h_res", "grad_h_out", "grad_h_post")[i],
                       "max_abs_error": (a.float() - b.float()).abs().max().item(), "rtol": rtol, "atol": atol})
    return actual, errors


def benchmark(args):
    import torch
    import torch_npu
    from .bench import _event_samples, _cann_version
    from .cli import _root
    from .mhc_post_backward import mhc_post_backward
    shapes = args.shapes or SHAPES
    implementations = tuple((args.implementations or "native,optimized").split(","))
    if any(i not in ("auto", "native", "optimized") for i in implementations):
        raise ValueError("mHC implementations: native,optimized or auto")
    dtypes = ("bfloat16",) if args.dtypes == "float32,bfloat16,float16" else tuple(args.dtypes.split(","))
    if any(d not in ("float16", "bfloat16") for d in dtypes):
        raise ValueError("mHC backward supports FP16/BF16")
    if args.warmup < 0 or args.batches < 1 or args.iterations_per_batch < 1:
        raise ValueError("invalid warmup/batch/iteration settings")
    rows = []
    with torch.inference_mode():
        for shape in shapes:
            for dtype in dtypes:
                values = inputs(shape, dtype)
                for implementation in implementations:
                    out, errors = check_case(values, implementation)
                    del out
                    # Both native and optimized allocate their outputs inside each call.
                    op = lambda: mhc_post_backward(*values, implementation=implementation)
                    samples = _event_samples(op, torch.npu, torch.npu.current_stream(), args.warmup, args.batches, args.iterations_per_batch)
                    row = dict(schema_version=1, operator="mhc_post_backward", implementation=implementation,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(), hardware=torch.npu.get_device_name(),
                        device="npu", device_index=torch.npu.current_device(), shape=list(values[1].shape), dtype=dtype,
                        h_res=True, measurement_method="torch.npu.Event.elapsed_time",
                        measurement_scope="device_stream_elapsed_per_call", execution_mode="eager", graph_capture=False,
                        allocation="both implementations allocate outputs per call", cache_policy="same-buffer steady state",
                        warmup_iterations=args.warmup, batches=args.batches, iterations_per_batch=args.iterations_per_batch,
                        latency_ms_median=statistics.median(samples), latency_ms_samples=samples,
                        correctness={"status":"PASS", "reference":"CPU FP64 explicit equations, six boundary/interior token samples", "outputs":errors},
                        python=platform.python_version(), torch=torch.__version__,
                        runtime={"torch_npu":torch_npu.__version__,"cann_version":_cann_version()},
                        native_api="torch_npu.npu_mhc_post_backward -> aclnnMhcPostBackward")
                    rows.append(row)
                    print(f"PASS {list(values[1].shape)} {dtype} {implementation}: {row['latency_ms_median']*1000:.3f} us",flush=True)
                del values
                torch.npu.synchronize()
    output = args.output or (_root()/"results/benchmarks"/f"mhc-post-backward-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl")
    if not output.is_absolute():
        output = _root()/output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows),encoding="utf-8")
    print(f"SAVED {len(rows)} measurements: {output}",flush=True)
    return 0


def run_cli(args):
    from .environment import require_device
    device = getattr(args, "device", None)
    if getattr(args,"backend",None) == "gpu" or (device is not None and device != "npu"):
        raise ValueError("mHC post backward currently has an Ascend 950 implementation")
    require_device("npu")
    implementation=getattr(args,"implementation",None) or "auto"
    if implementation not in ("auto","native","optimized"):
        raise ValueError("mHC --implementation accepts auto, native or optimized")
    if args.command == "build":
        from .backends.npu import _load_extension
        print(_load_extension().__file__)
        return 0
    if args.command == "example":
        import torch
        shape = (args.shape or [(512,4,2560)])[0]
        with torch.inference_mode():
            for dtype in ("float16","bfloat16"):
                _, errors = check_case(inputs(shape,dtype),implementation)
                print(json.dumps({"status":"PASS","shape":shape,"dtype":dtype,"outputs":errors}))
        return 0
    if args.command == "test":
        from .cli import _root
        env=os.environ.copy();env["XPU_KERNELS_TEST_DEVICE"]="npu"
        result=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_mhc_post_backward.py","-v"],cwd=_root(),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(result.stdout,end="")
        if result.returncode:
            return result.returncode
        # Device pass requires an actual NPU case rather than only CPU references.
        lines=result.stdout.splitlines()
        starts=[i for i,line in enumerate(lines) if "MhcNpuKernelTests" in line and line.startswith("test_")]
        passed=0
        for start in starts:
            case=[]
            for i in range(start,len(lines)):
                line=lines[i].strip()
                if i>start and line.startswith("test_"):break
                case.append(line)
                if line=="ok" or line.endswith("... ok"):
                    passed+=1;break
                if "skipped" in line or line in ("ERROR","FAIL"):break
        if not starts or passed != len(starts):
            raise RuntimeError("mHC device test did not execute")
        return 0
    if args.command == "bench":
        return benchmark(args)
    if args.command == "profile":
        from .cli import _root, _profile_command, _profile_subprocess_env
        if args.tool != "msprof":
            raise ValueError("NPU mHC profiling uses msprof")
        binary=shutil.which("msprof")
        if not binary:raise RuntimeError("msprof is not available")
        shape=(args.shape or [(1024,4,2560)])[0]
        check_case(inputs(shape,args.dtype),implementation)
        output=args.output if args.output.is_absolute() else _root()/args.output
        output.mkdir(parents=True,exist_ok=True)
        workload=[sys.executable,"-m","xpu_kernels.mhc_workload","--shape","x".join(map(str,shape)),"--dtype",args.dtype,"--implementation",implementation]
        help_text=subprocess.run([binary,"--help"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout
        command=_profile_command("msprof",binary,output,workload,help_text)
        print("PROFILE_COMMAND="+json.dumps(command))
        return subprocess.run(command,cwd=_root(),env=_profile_subprocess_env()).returncode
    raise ValueError(f"unsupported command {args.command}")


def main():
    import torch
    from .mhc_post_backward import mhc_post_backward
    parser=argparse.ArgumentParser()
    parser.add_argument("--shape",default="1024x4x2560")
    parser.add_argument("--dtype",choices=("float16","bfloat16"),default="bfloat16")
    parser.add_argument("--implementation",choices=("auto","native","optimized"),default="auto")
    parser.add_argument("--matrix", action="store_true", help="profile the optimized-scope shape matrix")
    parser.add_argument("--profile-warmup", type=int, default=20)
    parser.add_argument("--profile-repeats", type=int, default=5)
    args=parser.parse_args()
    if args.profile_warmup < 0 or args.profile_repeats < 1:
        parser.error("invalid profile iteration count")
    with torch.inference_mode():
        shapes = SHAPES if args.matrix else (tuple(map(int,args.shape.split("x"))),)
        for shape in shapes:
            values = inputs(shape, args.dtype)
            print("PROFILE_CASE="+json.dumps({"shape":shape, "implementation":args.implementation,
                "warmup":args.profile_warmup, "samples":args.profile_repeats}), flush=True)
            for _ in range(args.profile_warmup + args.profile_repeats):
                result = mhc_post_backward(*values, implementation=args.implementation)
                torch.npu.synchronize()
                del result
            del values
    print("MHC_PROFILE_COMPLETE",flush=True)


if __name__ == "__main__":
    main()
