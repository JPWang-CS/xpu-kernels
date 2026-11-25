"""Check operator naming, required files, and DeepJIT include closure without devices."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path = ROOT) -> list[str]:
    errors = []
    package = root / "src/xpu_kernels"
    exports = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))
    operators = [alias.name for node in exports.body if isinstance(node, ast.ImportFrom)
                 and node.level == 1 for alias in node.names if not alias.name.startswith("_")]
    if not operators:
        errors.append("no public operators found")
    for op in operators:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", op):
            errors.append(f"invalid operator name: {op}")
        for relative in (f"src/xpu_kernels/{op}.py", f"src/xpu_kernels/{op}_golden.py",
                         f"src/xpu_kernels/{op}_workload.py", f"tests/test_{op}.py",
                         f"docs/operators/{op}/README.md", f"docs/operators/{op}/optimization.md"):
            if not (root / relative).is_file():
                errors.append(f"missing {relative}")
        document_dir = root / "docs/operators" / op
        if (root / f"docs/operators/{op}.md").exists():
            errors.append(f"operator docs must use their own folder: {op}")
        if document_dir.is_dir():
            extra = {p.name for p in document_dir.glob("*.md")} - {"README.md", "optimization.md"}
            if extra:
                errors.append(f"operator docs use README.md and optimization.md: {op}: {sorted(extra)}")
        implementations = list((package / "backends").glob(f"{op}_*.py"))
        if not implementations:
            errors.append(f"no backend for {op}")
        for path in implementations:
            backend = path.stem[len(op) + 1:]
            if backend not in ("cuda", "triton", "ascend"):
                errors.append(f"unexpected backend name: {path.name}")
            if backend == "ascend":
                for relative in (f"csrc/ascend/{op}.cpp", f"kernels/include/xpu_kernels/{op}/kernel.hpp",
                                 f"kernels/include/xpu_kernels/{op}/LICENSE"):
                    if not (root / relative).is_file():
                        errors.append(f"missing {relative}")
                loader = (package / "backends/_ascend.py").read_text(encoding="utf-8")
                bindings = (root / "csrc/ascend/bindings.cpp").read_text(encoding="utf-8")
                if f'"{op}.cpp"' not in loader:
                    errors.append(f"Ascend loader does not build {op}.cpp")
                if f'module.def("{op}"' not in bindings:
                    errors.append(f"missing Ascend binding: {op}")
    headers = root / "kernels/include"
    for path in (headers / "xpu_kernels").iterdir():
        if not path.is_dir() or path.name not in operators:
            errors.append(f"kernel entry must live in its operator directory: {path.name}")
        elif any(child.is_dir() for child in path.iterdir()):
            errors.append(f"nested operator kernel directory: {path.name}")
    for path in (root / "kernels").rglob("*.hpp"):
        for include in re.findall(r"#\s*include\s*<([^>]+)>", path.read_text(encoding="utf-8")):
            if include.startswith("xpu_kernels/") and not (headers / include).is_file():
                errors.append(f"broken include in {path.name}: {include}")
    for path in (package / "backends").glob("*.py"):
        if path.stem.startswith("_"):
            continue
        if not any(path.stem == f"{op}_{backend}" for op in operators for backend in ("cuda", "triton", "ascend")):
            errors.append(f"backend must be named <operator>_<backend>.py: {path.name}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    errors = check(parser.parse_args().root)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Operator layout and local DeepJIT includes: PASS")


if __name__ == "__main__":
    main()
