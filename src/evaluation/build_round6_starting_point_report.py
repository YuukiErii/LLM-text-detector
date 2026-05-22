import argparse
import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TEACHER_COMPARISON = PROJECT_ROOT / "outputs" / "evaluation" / "round5_teacher_test_comparison.json"
DEFAULT_TEACHER_LEDGER_SUMMARY = PROJECT_ROOT / "outputs" / "evaluation" / "round5_teacher_test_ledger_summary.json"
DEFAULT_BASELINE_REPORT = PROJECT_ROOT / "outputs" / "evaluation" / "round5_baseline_frozen_report.json"
DEFAULT_GATE_REPORT = PROJECT_ROOT / "outputs" / "evaluation" / "round5_gate_report.md"
DEFAULT_RULES = PROJECT_ROOT / "outputs" / "models" / "round5_residual_override" / "rules.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round6_starting_point_report.md"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "outputs" / "evaluation" / "round6_starting_point_report.json"


REUSABLE_ARTIFACTS = [
    "outputs/predictions/round5_teacher_test_predictions.jsonl",
    "outputs/evaluation/round5_teacher_test_comparison.json",
    "outputs/evaluation/round5_teacher_test_ledger_summary.json",
    "outputs/models/round5_residual_override/rules.json",
    "outputs/models/round5_flip_guard/flip_guard.pkl",
    "outputs/models/round4_deberta_weighted_residual/best_model",
    "outputs/evaluation/round5_flip_ledger.jsonl",
    "data/processed/round5_flip_guard_train.jsonl",
    "data/processed/round5_flip_guard_dev_hardpos.jsonl",
    "data/processed/round5_flip_guard_dev_hardneg.jsonl",
]


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def correct_count(metrics: Dict) -> int:
    return int(round(float(metrics["accuracy"]) * int(metrics["num_samples"])))


def metric_row(name: str, metrics: Dict) -> str:
    return (
        f"| {name} | {correct_count(metrics)} / {metrics['num_samples']} | "
        f"{metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
        f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | "
        f"{metrics['false_positives']} | {metrics['false_negatives']} |"
    )


def false_positives(metrics: Dict) -> int:
    if "false_positives" in metrics:
        return int(metrics["false_positives"])
    return int(metrics["confusion_matrix"][0][1])


def false_negatives(metrics: Dict) -> int:
    if "false_negatives" in metrics:
        return int(metrics["false_negatives"])
    return int(metrics["confusion_matrix"][1][0])


def non_teacher_rows(baseline: Dict) -> List[str]:
    rows = []
    for split_name, split in baseline["splits"].items():
        step7 = split["step7"]
        round4 = split["round4"]
        rows.append(
            f"| {split_name} | Step7 | {step7['f1']:.4f} | "
            f"{step7['false_positives']} | {step7['false_negatives']} |"
        )
        rows.append(
            f"| {split_name} | Round4 DeBERTa | {round4['f1']:.4f} | "
            f"{round4['false_positives']} | {round4['false_negatives']} |"
        )
    return rows


def artifact_status(paths: List[str]) -> List[Dict]:
    out = []
    for rel_path in paths:
        path = PROJECT_ROOT / rel_path
        out.append(
            {
                "path": rel_path,
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            }
        )
    return out


def write_markdown(report: Dict, path: Path) -> None:
    teacher = report["teacher_test"]
    lines = [
        "# Round6 Starting Point Report",
        "",
        "Date: 2026-05-22",
        "",
        "This report freezes the Round6 starting point. Teacher-test labels remain diagnostic-only and must not be used for training, threshold selection, guard calibration, router tuning, stacker training, or model selection.",
        "",
        "## Starting Decision",
        "",
        "```text",
        "FINAL_MODEL_BEFORE_ROUND6 = Step7 ensemble",
        "ROUND5_PROMOTED = no",
        "ROUND6_NEXT_STEP = build non-teacher safe/unsafe override dataset",
        "```",
        "",
        "## Teacher-Test Diagnostic Summary",
        "",
        "| Run | Correct / 300 | Accuracy | Precision | Recall | F1 | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        metric_row("Step7 baseline", teacher["step7_baseline"]),
        metric_row("Round5 override", teacher["round5_override"]),
        "",
        "Round4 DeBERTa branch is retained only as a signal branch:",
        "",
        "| Branch | Correct / 300 | Accuracy | FP | FN | Decision |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| Round4 DeBERTa | {teacher['round4_deberta_branch']['correct']} / 300 | {teacher['round4_deberta_branch']['accuracy']:.4f} | {teacher['round4_deberta_branch']['false_positives']} | {teacher['round4_deberta_branch']['false_negatives']} | unsafe as global classifier |",
        "",
        "## Teacher-Test Override-Candidate Aggregate",
        "",
        "| Candidate type | Count |",
        "| --- | ---: |",
    ]
    for name, count in teacher["override_candidates"].items():
        lines.append(f"| {name} | {count} |")

    lines.extend(
        [
            "",
            "These are aggregate diagnostics only. Round6 must not export teacher-test text, labels, sample ids, or row-level threshold conditions into training, validation, or selection flows.",
            "",
            "## Non-Teacher Round5 Gate Summary",
            "",
            "| Split | Run | F1 | FP | FN |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(non_teacher_rows(report["round5_baseline_frozen_report"]))

    lines.extend(
        [
            "",
            "Round5 selected override rule:",
            "",
            "```json",
            json.dumps(report["round5_rules"], ensure_ascii=False, indent=2),
            "```",
            "",
            "Round5 non-teacher override delta:",
            "",
            "| Split | Overrides | Fixed Step7 FN | Induced FP |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for split_name, delta in report["round5_non_teacher_override_delta"].items():
        lines.append(
            f"| {split_name} | {delta['overrides']} | {delta['fixed_step7_fn']} | {delta['induced_fp']} |"
        )

    lines.extend(["", "## Reusable Artifacts", "", "| Path | Exists |", "| --- | --- |"])
    for item in report["artifact_status"]:
        lines.append(f"| `{item['path']}` | {item['exists']} |")

    lines.extend(
        [
            "",
            "## Round6 Entry Gates",
            "",
            "| Gate | Required before teacher-test |",
            "| --- | --- |",
            "| hardneg induced FP | 0 |",
            "| internal induced FP | <= 1 |",
            "| internal F1 | >= 0.9564 hard minimum, >= 0.9570 preferred |",
            "| hardpos fixed Step7 FN | >= 57 hard minimum, >= 70 target |",
            "| override rule | non-empty |",
            "| teacher-test leakage | exact duplicate = 0, no teacher labels/text in tuning |",
            "",
            "## Phase 0 Decision",
            "",
            "```text",
            "ROUND6_PHASE0_STATUS = complete",
            "PROMOTE_TO_PHASE1_DATASET_BUILD = yes",
            "TEACHER_TEST_SELECTION_ALLOWED = no",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args) -> Dict:
    teacher_comparison = read_json(Path(args.teacher_comparison))
    teacher_ledger = read_json(Path(args.teacher_ledger_summary))
    baseline = read_json(Path(args.baseline_report))
    rules = read_json(Path(args.rules))

    branch_metrics = read_json(PROJECT_ROOT / "outputs" / "predictions" / "round5_round4_deberta_teacher_test_metrics.json")
    candidate_counts = teacher_ledger["override_candidate_counts"]["teacher_test"]
    report = {
        "teacher_test": {
            "step7_baseline": teacher_comparison["baseline_metrics"],
            "round5_override": teacher_comparison["round5_metrics"],
            "round4_deberta_branch": {
                "correct": correct_count(branch_metrics),
                "accuracy": branch_metrics["accuracy"],
                "false_positives": false_positives(branch_metrics),
                "false_negatives": false_negatives(branch_metrics),
            },
            "override_candidates": {
                "safe fixed-FN candidate": int(candidate_counts.get("fixed_fn_candidate", 0)),
                "unsafe induced-FP candidate": int(candidate_counts.get("induced_fp", 0)),
                "total Step7-human -> Round4-LLM candidate": int(candidate_counts.get("total", 0)),
            },
        },
        "round5_baseline_frozen_report": baseline,
        "round5_rules": rules,
        "round5_non_teacher_override_delta": {
            "internal_test": {"overrides": 1, "fixed_step7_fn": 1, "induced_fp": 0},
            "hardpos": {"overrides": 57, "fixed_step7_fn": 57, "induced_fp": 0},
            "hardneg": {"overrides": 0, "fixed_step7_fn": 0, "induced_fp": 0},
        },
        "artifact_status": artifact_status(REUSABLE_ARTIFACTS),
        "strict_boundary": "teacher-test may be used only for final diagnostics, not for Round6 training or selection",
    }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Build Round6 starting-point report.")
    parser.add_argument("--teacher_comparison", default=str(DEFAULT_TEACHER_COMPARISON))
    parser.add_argument("--teacher_ledger_summary", default=str(DEFAULT_TEACHER_LEDGER_SUMMARY))
    parser.add_argument("--baseline_report", default=str(DEFAULT_BASELINE_REPORT))
    parser.add_argument("--gate_report", default=str(DEFAULT_GATE_REPORT))
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--report_json", default=str(DEFAULT_REPORT_JSON))
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(args)
    write_json(report, Path(args.report_json))
    write_markdown(report, Path(args.report_md))
    print("=" * 70)
    print("Round6 starting point report built")
    print("=" * 70)
    print(f"Report: {args.report_md}")
    print(f"JSON: {args.report_json}")
    print("Step7 teacher-test correct:", correct_count(report["teacher_test"]["step7_baseline"]))
    print("Round5 teacher-test correct:", correct_count(report["teacher_test"]["round5_override"]))
    print("Round4 branch teacher-test correct:", report["teacher_test"]["round4_deberta_branch"]["correct"])


if __name__ == "__main__":
    main()
