import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))

from tune_round5_residual_override import (
    align_split,
    apply_rules,
    metrics_for,
    override_delta,
    prediction_rows,
    save_jsonl,
    write_json,
)


DEFAULT_RULES = PROJECT_ROOT / "outputs" / "models" / "round5_residual_override" / "rules.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "predictions" / "round5_teacher_test_predictions.jsonl"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "evaluation" / "round5_teacher_test_comparison.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round5_teacher_test_comparison.md"


def baseline_metrics(rows):
    return metrics_for(
        rows,
        [int(row["step7_prediction"]) for row in rows],
        [float(row["step7_prob"]) for row in rows],
    )


def write_markdown(report, path: Path) -> None:
    lines = [
        "# Round5 Teacher-Test Comparison",
        "",
        "This file applies the already-selected Round5 residual override rule. It does not tune thresholds on teacher-test.",
        "",
        "## Selected Rule",
        "",
        "```json",
        json.dumps(report["rules"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "| Run | n | Accuracy | Precision | Recall | F1 | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in [("Step7 baseline", report["baseline_metrics"]), ("Round5 override", report["round5_metrics"])]:
        lines.append(
            f"| {name} | {metrics['num_samples']} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['false_positives']} | {metrics['false_negatives']} |"
        )
    delta = report["override_delta"]
    lines.extend(
        [
            "",
            "## Override Delta",
            "",
            f"- overrides: {delta['overrides']}",
            f"- fixed Step7 FN: {delta['fixed_step7_fn']}",
            f"- induced FP: {delta['induced_fp']}",
            f"- broke Step7 correct: {delta['broke_step7_correct']}",
            "",
            "## Decision Bands",
            "",
            "- `<= 274 / 300`: reject, keep Step7",
            "- `275-284 / 300`: partial improvement, not 95%",
            "- `>= 285 / 300`: reaches 95%, promote if leakage/tuning rules were respected",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply selected Round5 residual override rules to a labeled split.")
    parser.add_argument("--split_name", default="teacher_test")
    parser.add_argument("--step7", required=True)
    parser.add_argument("--round5", required=True)
    parser.add_argument("--human_guard", required=True)
    parser.add_argument("--flip_guard", required=True)
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    return parser.parse_args()


def main():
    args = parse_args()
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    rows = align_split(
        args.split_name,
        {
            "step7": Path(args.step7),
            "round5": Path(args.round5),
            "human_guard": Path(args.human_guard),
            "flip_guard": Path(args.flip_guard),
        },
    )
    if not rows:
        raise ValueError("No aligned rows found.")
    preds, probs, decisions = apply_rules(rows, rules)
    out_rows = prediction_rows(rows, preds, probs, decisions)
    save_jsonl(out_rows, Path(args.output))

    report = {
        "rules": rules,
        "split_name": args.split_name,
        "output": args.output,
        "baseline_metrics": baseline_metrics(rows),
        "round5_metrics": metrics_for(rows, preds, probs),
        "override_delta": override_delta(rows, preds),
        "inputs": {
            "step7": args.step7,
            "round5": args.round5,
            "human_guard": args.human_guard,
            "flip_guard": args.flip_guard,
        },
    }
    write_json(report, Path(args.metrics))
    write_markdown(report, Path(args.report_md))

    metrics = report["round5_metrics"]
    delta = report["override_delta"]
    print("=" * 70)
    print("Round5 residual override applied")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.output}")
    print(f"Accuracy: {metrics['accuracy']:.4f} F1: {metrics['f1']:.4f}")
    print(
        f"Overrides: {delta['overrides']} fixed_FN={delta['fixed_step7_fn']} "
        f"induced_FP={delta['induced_fp']}"
    )


if __name__ == "__main__":
    main()
