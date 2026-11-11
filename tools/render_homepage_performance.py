"""Update the README performance block from verified public device records."""
from pathlib import Path
import argparse
from render_native_comparisons import BEGIN, END, homepage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "README.md"
    original = path.read_text(encoding="utf-8")
    if original.count(BEGIN) != 1 or original.count(END) != 1:
        raise ValueError("README needs exactly one performance marker pair")
    start, stop = original.index(BEGIN)+len(BEGIN), original.index(END)
    if stop < start:
        raise ValueError("performance markers are reversed")
    updated = original[:start]+"\n"+homepage()+"\n"+original[stop:]
    if args.check:
        if updated != original:
            raise SystemExit("README performance is out of date")
    elif updated != original:
        if path.read_text(encoding="utf-8") != original:
            raise RuntimeError("README changed during rendering")
        path.write_text(updated, encoding="utf-8")
    print("README performance matches the measured records")


if __name__ == "__main__":
    main()
