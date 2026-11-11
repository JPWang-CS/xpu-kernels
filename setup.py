from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent


class BuildPyWithNativeResources(build_py):
    """Include JIT host/device sources and upstream DeepJIT headers in wheels."""

    def run(self):
        super().run()
        target = Path(self.build_lib) / "xpu_kernels" / "_native"
        resources = {
            ROOT / "csrc" / "ascend": target / "csrc" / "ascend",
            ROOT / "csrc" / "cuda": target / "csrc" / "cuda",
            ROOT / "kernels" / "include": target / "kernels" / "include",
            ROOT / "third-party" / "deep-jit" / "include": target / "third-party" / "deep-jit" / "include",
        }
        for source, destination in resources.items():
            if not source.is_dir():
                raise RuntimeError(f"required native package resource is missing: {source}")
            shutil.copytree(source, destination, dirs_exist_ok=True)
        deep_jit_license = ROOT / "third-party" / "deep-jit" / "LICENSE"
        if deep_jit_license.is_file():
            shutil.copy2(deep_jit_license, target / "third-party" / "deep-jit" / "LICENSE")


setup(cmdclass={"build_py": BuildPyWithNativeResources})
