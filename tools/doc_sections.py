"""Keep measured sections reproducible while preserving hand-written operator explanations."""
from __future__ import annotations

import posixpath
from pathlib import Path
import re

BEGIN = "<!-- BEGIN measured-performance -->"
END = "<!-- END measured-performance -->"
ROUTES = {
    "docs/operators/rms_norm.md": "docs/operators/rms_norm/README.md",
    "docs/operators/mhc_post_backward.md": "docs/operators/mhc_post_backward/README.md",
    "docs/performance/rms_norm.md": "docs/operators/rms_norm/optimization.md#实测结果",
    "docs/performance/mhc_post_backward.md": "docs/operators/mhc_post_backward/optimization.md#实测结果",
}


def relocate_links(text: str, source: str, destination: str) -> str:
    def replace(match):
        link = match.group(1)
        if link.startswith(("#", "http:", "https:", "mailto:")) or "://" in link:
            return match.group(0)
        path, sep, fragment = link.partition("#")
        original = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
        target = ROUTES.get(original, original)
        target_path, target_sep, target_fragment = target.partition("#")
        updated = posixpath.relpath(target_path, posixpath.dirname(destination))
        if target_sep or sep:
            updated += "#" + (target_fragment if target_sep else fragment)
        return "](" + updated + ")"
    return re.sub(r"\]\(([^)]+)\)", replace, text)


def measured_body(text: str, source: str, destination: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    text = "\n".join(lines).strip()
    text = re.sub(r"^(#{2,5}) ", r"\1# ", text, flags=re.M)
    return relocate_links(text, source, destination)


def update_measured_section(path: Path, body: str) -> str:
    original = path.read_text(encoding="utf-8")
    if original.count(BEGIN) != 1 or original.count(END) != 1:
        raise ValueError(f"{path.name} needs exactly one measured-performance marker pair")
    start, stop = original.index(BEGIN) + len(BEGIN), original.index(END)
    if stop < start:
        raise ValueError("measured-performance markers are reversed")
    return original[:start] + "\n\n" + body.strip() + "\n\n" + original[stop:]
