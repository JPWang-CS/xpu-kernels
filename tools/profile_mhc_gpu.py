"""Capture or extract the GPU mHC matrix using official Nsight Compute metrics."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from xpu_kernels.mhc_workload import SHAPES


def extract(path,checks,output):
    text=path.read_text(encoding="utf-8-sig")
    header=next((i for i,line in enumerate(text.splitlines()) if line.startswith('"ID","Process ID"')),None)
    if header is None:
        raise ValueError("NCU CSV header was not found")
    rows=list(csv.DictReader(io.StringIO("\n".join(text.splitlines()[header:]))))
    rows=[r for r in rows if r.get("Kernel Name")=="mhc_post_bwd_cuda_kernel" and r.get("Metric Name")=="gpu__time_duration.sum"]
    rows.sort(key=lambda r:int(r["ID"]))
    if len(rows)!=len(SHAPES)*25:
        raise ValueError(f"expected 400 target kernel samples; found {len(rows)}")
    evidence={tuple(r["shape"]):r for r in (json.loads(line) for line in checks.read_text(encoding="utf-8").splitlines())}
    results=[];raw=[]
    for index,shape in enumerate(SHAPES):
        check=evidence[shape]
        if check["correctness"]["status"]!="PASS" or check["implementation"]!="tilelang" or check["dtype"]!="bfloat16":
            raise ValueError("GPU correctness evidence does not match this kernel")
        tokens=shape[0] if len(shape)==3 else shape[0]*shape[1]
        chunk=rows[index*25:(index+1)*25]
        values=[]
        for iteration,row in enumerate(chunk):
            if int(re.findall(r"\d+",row["Grid Size"])[0])!=tokens:
                raise ValueError("NCU grid does not match the matrix execution order")
            scale={"ns":0.001,"us":1.0,"nsecond":0.001,"usecond":1.0}.get(row["Metric Unit"])
            if scale is None:
                raise ValueError(f"unsupported NCU unit: {row['Metric Unit']}")
            value=float(row["Metric Value"].replace(",",""))*scale
            values.append(value)
            raw.append({"shape":"x".join(map(str,shape)),"iteration":iteration,
                "phase":"warmup" if iteration<20 else "measured","launch_id":row["ID"],
                "kernel":row["Kernel Name"],"grid":row["Grid Size"],"duration_us":value})
        results.append({"schema_version":1,"operator":"mhc_post_backward","implementation":"tilelang",
            "hardware":check["hardware"],"device":"cuda","shape":list(shape),"dtype":"bfloat16","h_res":True,
            "measurement_method":"Nsight Compute gpu__time_duration.sum","measurement_scope":"device_kernel_duration",
            "latency_us_median":statistics.median(values[20:]),"samples_us":values[20:],"warmup_iterations":20,
            "replay_mode":"kernel","cache_control":"none","clock_control":"none",
            "correctness":check["correctness"],"torch":check["torch"],"runtime":check["runtime"],
            "source_api":"DeepSeek TileKernels mhc_post_bwd_cuda",
            "source_csv":"docs/performance/data/mhc-gpu-tilelang.csv"})
    target=ROOT/"docs/performance/data/mhc-gpu-tilelang.csv"
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in results),encoding="utf-8")
    print("Saved16 GPU kernel medians,80 measured samples and400 complete launch rows")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"results/profiles/mhc-gpu")
    parser.add_argument("--checks",type=Path,default=ROOT/"results/benchmarks/mhc-post-backward-gpu-tilelang.jsonl")
    parser.add_argument("--records",type=Path,default=ROOT/"results/benchmarks/mhc-post-backward-gpu-ncu.jsonl")
    args=parser.parse_args()
    if args.from_csv is None:
        from xpu_kernels.environment import require_device
        from xpu_kernels.cli import _profile_subprocess_env
        require_device("cuda")
        binary=shutil.which("ncu")
        if not binary:parser.error("ncu was not found")
        args.output.mkdir(parents=True,exist_ok=False)
        env=_profile_subprocess_env()
        env["NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT"]="1"
        env["XDG_CONFIG_HOME"]=str(args.output.resolve()/"config")
        Path(env["XDG_CONFIG_HOME"]).mkdir()
        args.from_csv=args.output/"matrix.csv"
        command=[binary,"--kernel-name","regex:mhc_post_bwd_cuda_kernel","--replay-mode","kernel",
            "--cache-control","none","--clock-control","none","--metrics","gpu__time_duration.sum",
            "--csv","--log-file",str(args.from_csv.resolve()),sys.executable,"-B","-u","-m",
            "xpu_kernels.mhc_workload","--device","cuda","--matrix","--implementation","tilelang",
            "--profile-warmup","20","--profile-repeats","5"]
        (args.output/"command.json").write_text(json.dumps(command),encoding="utf-8")
        with (args.output/"application.log").open("w",encoding="utf-8") as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    extract(args.from_csv,args.checks,args.records)


if __name__=="__main__":main()
