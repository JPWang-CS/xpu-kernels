# xpu-kernels agent entry

Before editing or validating this project, read [skills/xpu-kernels/SKILL.md](skills/xpu-kernels/SKILL.md). It is the single project skill and defines the RMSNorm and mHC backward contracts, backend commands, correctness gate, benchmark method, and profiling workflow.

`pyproject.toml` is the dependency authority; `requirements.txt` only forwards to the editable project install. Keep build and runtime caches under this project. Preserve private `envs/`, `runs/`, `artifacts/`, `.runtime/`, and `tools/remote/` contents; they are excluded from Git.

Before adding or reorganizing an operator, follow [docs/repository-layout.md](docs/repository-layout.md). Use the same snake_case operator name in the public API, backend modules, native source directory, golden, tests, and documentation. Shared loading/build code stays shared; each Ascend operator owns one flat directory with `kernel.hpp` as its entry. Register its workload in the existing CLI. Run `python tools/check_repository_layout.py` and the relevant correctness/package checks. Update the operator guide and homepage from accepted measurements whenever public implementations or results change. Preserve original benchmark records and third-party provenance.

Performance charts use the shared Python/Matplotlib renderer in `tools/render_homepage_charts.py`, its single `STYLE`, and committed benchmark data. Keep one figure per operator with the same fonts, colors, axes and number format. New operators add datasets and shapes to that renderer. Regenerate all figures after shared style changes; verify with `--check` and a visual review.
