import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.round3_fusion_utils import (  # noqa: E402
    align_prediction_runs,
    apply_precision_guard_rules,
    metrics_for_labels,
    override_delta_summary,
    precision_guard_prediction_rows,
    save_jsonl,
    write_json,
)


DEFAULT_RULES = PROJECT_ROOT / "outputs" / "models" / "round3_precision_guard" / "rules.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with Round3 precision-guarded ensemble rules.")
    parser.add_argument("--runs", nargs="+", required=True, help="Prediction specs in NAME=PATH format.")
    parser.add_argument("--split_name", default="predict")
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--delta_json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    _run_names, rows = align_prediction_runs(args.runs, split_name=args.split_name)
    preds, probs, decisions = apply_precision_guard_rules(rows, rules)
    out_rows = precision_guard_prediction_rows(rows, preds, probs, decisions)
    save_jsonl(out_rows, Path(args.output))

    metrics = metrics_for_labels([row["label"] for row in rows], preds, probs)
    delta = override_delta_summary(rows, preds, rules.get("step7_run", "step7"))
    if args.metrics:
        write_json(metrics, Path(args.metrics))
    if args.delta_json:
        write_json(delta, Path(args.delta_json))

    print("=" * 70)
    print("Round3 precision-guarded predictions written")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.output}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Confusion: {metrics['confusion_matrix']}")
    print(f"Overrides: {delta['overrides']} fixed_FN={delta['fixed_step7_fn']} induced_FP={delta['induced_fp']}")


if __name__ == "__main__":
    main()
