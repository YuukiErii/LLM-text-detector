import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))

from tune_round7_exact_safe_override import (
    apply_rules,
    load_jsonl,
    metrics_for,
    override_delta,
    prediction_rows,
    prob,
    save_jsonl,
    write_json,
)


DEFAULT_RULES = PROJECT_ROOT / "outputs" / "models" / "round7_exact_safe_override" / "rules.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "predictions" / "round7_teacher_test_predictions.jsonl"
DEFAULT_COMPARISON = PROJECT_ROOT / "outputs" / "evaluation" / "round7_teacher_test_comparison.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round7_teacher_test_comparison.md"
DEFAULT_LEDGER_SUMMARY = PROJECT_ROOT / "outputs" / "evaluation" / "round7_teacher_test_ledger_summary.json"


def baseline_metrics(rows: Sequence[Dict]) -> Dict:
    return metrics_for(
        rows,
        [int(row["step7_pred"]) for row in rows],
        [prob(row, "step7_prob") for row in rows],
    )


def branch_metrics(rows: Sequence[Dict]) -> Dict:
    return metrics_for(
        rows,
        [int(row["round4_pred"]) for row in rows],
        [prob(row, "round4_prob") for row in rows],
    )


def bucket_counts(rows: Sequence[Dict], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))


def candidate_summary(rows: Sequence[Dict]) -> Dict:
    exact_rows = [
        row
        for row in rows
        if int(row.get("step7_pred", 0)) == 0
        and int(row.get("round4_pred", 0)) == 1
        and row.get("flip_type") in {"fixed_fn_candidate", "induced_fp"}
    ]
    return {
        "num_rows": len(rows),
        "exact_override_candidates": len(exact_rows),
        "exact_candidate_flip_counts": dict(sorted(Counter(str(row.get("flip_type") or "unknown") for row in exact_rows).items())),
        "exact_candidate_round4_bucket_counts": bucket_counts(exact_rows, "round4_bucket"),
    }


def applied_override_summary(out_rows: Sequence[Dict]) -> Dict:
    override_rows = [row for row in out_rows if row.get("round7_override")]
    return {
        "num_overrides": len(override_rows),
        "override_flip_counts": dict(sorted(Counter(str(row.get("flip_type") or "unknown") for row in override_rows).items())),
        "override_round4_bucket_counts": bucket_counts(override_rows, "round4_bucket"),
    }


def correct_count(metrics: Dict) -> int:
    matrix = metrics["confusion_matrix"]
    return int(matrix[0][0] + matrix[1][1])


def decision_for(metrics: Dict, baseline: Dict) -> str:
    correct = correct_count(metrics)
    baseline_correct = correct_count(baseline)
    if correct >= 285:
        return "ROUND7_TEACHER_TEST_DECISION = 95_percent_reached; promote only after final leakage review."
    if correct > baseline_correct:
        return "ROUND7_TEACHER_TEST_DECISION = partial_success; Round7 beats Step7 but remains below 95_percent."
    return "ROUND7_TEACHER_TEST_DECISION = reject; keep Step7 ensemble as final."


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round7 Teacher-Test Comparison",
        "",
        "This file applies the already-frozen Round7 exact-selector override rule.",
        "It does not tune thresholds or search rules on teacher-test.",
        "",
        "## Selected Rule",
        "",
        "```json",
        json.dumps(report["rules"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "| Run | Correct | n | Accuracy | Precision | Recall | F1 | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in [
        ("Step7 baseline", report["baseline_metrics"]),
        ("Round4 signal branch", report["round4_branch_metrics"]),
        ("Round7 override", report["round7_metrics"]),
    ]:
        lines.append(
            f"| {name} | {correct_count(metrics)} | {metrics['num_samples']} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['false_positives']} | {metrics['false_negatives']} |"
        )
    delta = report["override_delta"]
    ledger = report["ledger_summary"]
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
            "## Candidate Summary",
            "",
            "| Item | Count |",
            "| --- | ---: |",
            f"| exact Step7-human -> Round4-LLM candidates | {ledger['candidate_summary']['exact_override_candidates']} |",
            f"| selected Round7 overrides | {ledger['applied_override_summary']['num_overrides']} |",
        ]
    )
    for name, count in ledger["candidate_summary"]["exact_candidate_flip_counts"].items():
        lines.append(f"| exact candidate `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## Decision Bands",
            "",
            "- `<= 274 / 300`: reject, keep Step7",
            "- `275-284 / 300`: partial improvement, not 95%",
            "- `>= 285 / 300`: reaches 95%, promote only if leakage/tuning rules were respected",
            "",
            "## Decision",
            "",
            "```text",
            report["decision"],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply the frozen Round7 exact safe override rule to a labeled split.")
    parser.add_argument("--split_name", default="teacher_test")
    parser.add_argument("--input", required=True, help="Round7-scored aligned rows.")
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--comparison", default=str(DEFAULT_COMPARISON))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--ledger_summary", default=str(DEFAULT_LEDGER_SUMMARY))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    if not rows:
        raise ValueError("No Round7 scored rows found.")
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    preds, probs, decisions = apply_rules(rows, rules)
    out_rows = prediction_rows(rows, preds, probs, decisions)
    save_jsonl(out_rows, Path(args.output))

    ledger_summary = {
        "split_name": args.split_name,
        "input": args.input,
        "output": args.output,
        "candidate_summary": candidate_summary(rows),
        "applied_override_summary": applied_override_summary(out_rows),
    }
    baseline = baseline_metrics(rows)
    round7 = metrics_for(rows, preds, probs)
    report = {
        "rules": rules,
        "split_name": args.split_name,
        "input": args.input,
        "output": args.output,
        "baseline_metrics": baseline,
        "round4_branch_metrics": branch_metrics(rows),
        "round7_metrics": round7,
        "override_delta": override_delta(rows, preds),
        "ledger_summary": ledger_summary,
    }
    report["decision"] = decision_for(round7, baseline)
    write_json(ledger_summary, Path(args.ledger_summary))
    write_json(report, Path(args.comparison))
    write_markdown(report, Path(args.report_md))

    metrics = report["round7_metrics"]
    delta = report["override_delta"]
    print("=" * 70)
    print("Round7 exact safe override applied")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.output}")
    print(f"Correct: {correct_count(metrics)} / {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} F1: {metrics['f1']:.4f}")
    print(
        f"Overrides: {delta['overrides']} fixed_FN={delta['fixed_step7_fn']} "
        f"induced_FP={delta['induced_fp']}"
    )
    print(report["decision"])


if __name__ == "__main__":
    main()
