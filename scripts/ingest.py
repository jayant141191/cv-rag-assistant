"""Index CV files from the command line.

Usage:
    python scripts/ingest.py data/sample_cvs            # a folder
    python scripts/ingest.py path/to/cv.pdf other.docx  # individual files
    python scripts/ingest.py data/sample_cvs --reset    # wipe the index first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_rag.loaders import SUPPORTED_EXTENSIONS  # noqa: E402
from cv_rag.vectorstore import CVVectorStore  # noqa: E402


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
        else:
            print(f"skip: {p} (unsupported)")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="CV files or folders")
    parser.add_argument("--reset", action="store_true", help="Delete everything in the index first")
    args = parser.parse_args()

    store = CVVectorStore()
    if args.reset:
        store.reset()
        print("Index reset.")

    files = collect(args.paths)
    if not files:
        sys.exit("No supported CV files found.")
    for f in files:
        try:
            info = store.add_path(f)
            print(f"indexed: {f.name:<35} candidate={info['candidate']:<25} chunks={info['chunks']}")
        except Exception as exc:
            print(f"FAILED:  {f.name}: {exc}")
    print(f"\nCandidates in index: {', '.join(store.list_candidates())}")


if __name__ == "__main__":
    main()
