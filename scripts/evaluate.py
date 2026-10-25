"""Evaluate the RAG pipeline against a labelled question set.

Usage:
    python scripts/evaluate.py                       # defaults: sample CVs + bundled dataset
    python scripts/evaluate.py --no-judge            # skip LLM-as-judge (cheaper, retrieval metrics only)
    python scripts/evaluate.py --fail-under 4.0      # exit 1 if mean correctness < 4.0 (for CI)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv_rag.evaluation.runner import run_evaluation  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=ROOT / "evaluation/datasets/cv_eval.jsonl")
    p.add_argument("--cv-dir", default=ROOT / "data/sample_cvs")
    p.add_argument("--out-dir", default=ROOT / "evaluation/results")
    p.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring")
    p.add_argument("--no-reingest", action="store_true", help="Reuse the existing eval collection")
    p.add_argument("--fail-under", type=float, default=None,
                   help="Fail (exit 1) if mean judge correctness is below this value")
    args = p.parse_args()

    result = run_evaluation(args.dataset, args.cv_dir, args.out_dir,
                            use_judge=not args.no_judge, reingest=not args.no_reingest)
    overall = result["summary"]["overall"]
    print("\n=== Overall ===")
    for k, v in overall.items():
        print(f"{k:<24} {'–' if v is None else v}")
    print(f"\nReport: {result['report_path']}")

    if args.fail_under is not None:
        score = overall.get("correctness")
        if score is None or score < args.fail_under:
            sys.exit(f"Mean correctness {score} is below threshold {args.fail_under}")


if __name__ == "__main__":
    main()
