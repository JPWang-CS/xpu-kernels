# xpu-kernels agent entry

Before editing or validating this project, read [skills/xpu-kernels/SKILL.md](skills/xpu-kernels/SKILL.md). It is the single project skill and defines the RMSNorm and mHC backward contracts, backend commands, correctness gate, benchmark method, and profiling workflow.

`pyproject.toml` is the dependency authority; `requirements.txt` only forwards to the editable project install. Keep build and runtime caches under this project. Preserve private `envs/`, `runs/`, `artifacts/`, `.runtime/`, and `tools/remote/` contents; they are excluded from Git.
