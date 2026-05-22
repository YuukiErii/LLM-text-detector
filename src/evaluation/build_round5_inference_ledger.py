import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))

from build_round5_flip_ledger import align_split, save_jsonl, split_flip_counts, override_candidate_counts


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "round5_teacher_test_ledger.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "evaluation" / "round5_teacher_test_ledger_summary.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Build Round5 inference ledger for a labeled split.")
    parser.add_argument("--split_name", default="teacher_test")
    parser.add_argument("--source", required=True)
    parser.add_argument("--step7", required=True)
    parser.add_argument("--round4", required=True)
    parser.add_argument("--guard", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


def main():
    args = parse_args()
    rows, diagnostics = align_split(
        args.split_name,
        {
            "source": Path(args.source),
            "step7": Path(args.step7),
            "round4": Path(args.round4),
            "guard": Path(args.guard),
        },
    )
    save_jsonl(rows, Path(args.output))
    summary = {
        "split_name": args.split_name,
        "num_rows": len(rows),
        "diagnostics": diagnostics,
        "split_flip_counts": split_flip_counts(rows),
        "override_candidate_counts": override_candidate_counts(rows),
        "inputs": {
            "source": args.source,
            "step7": args.step7,
            "round4": args.round4,
            "guard": args.guard,
        },
        "output": args.output,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 70)
    print("Round5 inference ledger built")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
