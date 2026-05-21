import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.round3_fusion_utils import (  # noqa: E402
    HIGH_RISK_BUCKETS,
    apply_precision_guard_rules,
    baseline_metrics,
    fmt,
    load_split_sets,
    metrics_for_labels,
    metrics_table_lines,
    override_delta_summary,
    precision_guard_prediction_rows,
    safe_name,
    save_jsonl,
    split_metrics,
    write_json,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round3_precision_guard"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round3_precision_guard_tuning_report.md"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"


def float_grid(value: str) -> List[float]:
    return [float(part) for part in str(value).split(",") if part.strip()]


def int_grid(value: str) -> List[int]:
    return [int(part) for part in str(value).split(",") if part.strip()]


def split_rows(rows: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for row in rows:
        out.setdefault(str(row.get("split_name", "unknown")), []).append(row)
    return out


def metrics_by_split(rows: Sequence[Dict], preds: Sequence[int], probs: Sequence[float]) -> Dict[str, Dict]:
    by_split = {}
    for split_name, split_group in split_rows(rows).items():
        indices = [idx for idx, row in enumerate(rows) if row.get("split_name") == split_name]
        by_split[split_name] = metrics_for_labels(
            [int(rows[i]["label"]) for i in indices],
            [int(preds[i]) for i in indices],
            [float(probs[i]) for i in indices],
        )
    return dict(sorted(by_split.items()))


def deltas_by_split(rows: Sequence[Dict], preds: Sequence[int], step7_run: str) -> Dict[str, Dict]:
    out = {}
    for split_name, split_group in split_rows(rows).items():
        indices = [idx for idx, row in enumerate(rows) if row.get("split_name") == split_name]
        out[split_name] = override_delta_summary([rows[i] for i in indices], [preds[i] for i in indices], step7_run)
    return dict(sorted(out.items()))


def candidate_rules(args, disabled: bool = False) -> List[Dict]:
    if disabled:
        return [
            {
                "step7_run": args.step7_run,
                "oof_run": args.oof_run,
                "roberta_run": args.roberta_run,
                "electra_run": args.electra_run,
                "oof_threshold": 1.1,
                "roberta_threshold": 1.1,
                "electra_threshold": 1.1,
                "high_risk_threshold_add": 0.0,
                "max_disagreement": 1.1,
                "min_votes": 3,
                "min_words": 0,
                "high_risk_buckets": sorted(HIGH_RISK_BUCKETS),
                "disabled_baseline": True,
            }
        ]

    rules = []
    for oof_th, rob_th, ele_th, add, max_dis, min_votes, min_words in itertools.product(
        float_grid(args.oof_thresholds),
        float_grid(args.roberta_thresholds),
        float_grid(args.electra_thresholds),
        float_grid(args.high_risk_threshold_adds),
        float_grid(args.max_disagreements),
        int_grid(args.min_votes),
        int_grid(args.min_words),
    ):
        rules.append(
            {
                "step7_run": args.step7_run,
                "oof_run": args.oof_run,
                "roberta_run": args.roberta_run,
                "electra_run": args.electra_run,
                "oof_threshold": oof_th,
                "roberta_threshold": rob_th,
                "electra_threshold": ele_th,
                "high_risk_threshold_add": add,
                "max_disagreement": max_dis,
                "min_votes": min_votes,
                "min_words": min_words,
                "high_risk_buckets": sorted(HIGH_RISK_BUCKETS),
                "disabled_baseline": False,
            }
        )
    return rules


def evaluate_candidate(rows: Sequence[Dict], rules: Dict, baseline_by_split: Dict[str, Dict], args) -> Dict:
    preds, probs, decisions = apply_precision_guard_rules(rows, rules)
    by_split = metrics_by_split(rows, preds, probs)
    deltas = deltas_by_split(rows, preds, args.step7_run)

    guard = safe_name(args.guard_split)
    valid = safe_name(args.valid_split)
    round2 = safe_name(args.round2_split)
    constraints = []
    if guard in by_split:
        constraints.append(by_split[guard]["false_positives"] <= baseline_by_split[guard]["false_positives"] + args.guard_fp_tolerance)
        constraints.append(by_split[guard]["recall"] >= baseline_by_split[guard]["recall"])
    if valid in by_split:
        constraints.append(by_split[valid]["false_positives"] <= baseline_by_split[valid]["false_positives"] + args.valid_fp_tolerance)
        constraints.append(by_split[valid]["f1"] >= baseline_by_split[valid]["f1"] - args.valid_f1_tolerance)

    fixed_guard = deltas.get(guard, {}).get("fixed_step7_fn", 0)
    fixed_round2 = deltas.get(round2, {}).get("fixed_step7_fn", 0)
    fixed_valid = deltas.get(valid, {}).get("fixed_step7_fn", 0)
    induced_guard = deltas.get(guard, {}).get("induced_fp", 0)
    induced_valid = deltas.get(valid, {}).get("induced_fp", 0)
    total_overrides = sum(block["overrides"] for block in deltas.values())
    score = (
        3.0 * fixed_guard
        + 1.4 * fixed_round2
        + 0.5 * fixed_valid
        - 5.0 * induced_guard
        - 2.0 * induced_valid
        - 0.03 * total_overrides
    )
    return {
        "rules": rules,
        "constraints_passed": bool(all(constraints)) if constraints else True,
        "score": float(score),
        "metrics_by_split": by_split,
        "deltas_by_split": deltas,
        "override_count": total_overrides,
    }


def write_report(report: Dict, path: Path) -> None:
    selected = report["selected"]
    lines = [
        "# Round3 Phase E Precision-Guard Tuning Report",
        "",
        "This tuning step keeps Step7 as the default decision and searches only",
        "human-to-LLM override rules on non-teacher-test splits. Teacher-test labels",
        "are not used for rule selection.",
        "",
        "## Selected Rule",
        "",
        "```json",
        report["selected_rules_json"],
        "```",
        "",
        f"- constraints passed: {selected['constraints_passed']}",
        f"- score: {selected['score']:.4f}",
        f"- total overrides on tuning splits: {selected['override_count']}",
        "",
        "## Selected Metrics",
        "",
    ]
    lines.extend(metrics_table_lines(selected["metrics_by_split"]))
    lines.extend(["", "## Step7 Baselines", ""])
    lines.extend(metrics_table_lines(report["baseline_by_split"]))
    lines.extend(["", "## Override Delta By Split", ""])
    lines.append("| Split | Overrides | Fixed Step7 FN | Induced FP |")
    lines.append("| --- | ---: | ---: | ---: |")
    for split_name, block in selected["deltas_by_split"].items():
        lines.append(
            f"| {split_name} | {block['overrides']} | {block['fixed_step7_fn']} | {block['induced_fp']} |"
        )
    lines.extend(["", "## Top Feasible Rules", ""])
    lines.append("| Rank | Score | Overrides | Guard FP | Guard FN | Round2 Fixed FN | Rule |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    guard = safe_name(report["config"]["guard_split"])
    round2 = safe_name(report["config"]["round2_split"])
    for idx, item in enumerate(report["top_feasible"], start=1):
        guard_metrics = item["metrics_by_split"].get(guard, {})
        round2_delta = item["deltas_by_split"].get(round2, {})
        rule = item["rules"]
        rule_short = (
            f"oof>={rule['oof_threshold']}, rob>={rule['roberta_threshold']}, "
            f"ele>={rule['electra_threshold']}, votes>={rule['min_votes']}, "
            f"risk+{rule['high_risk_threshold_add']}, dis<={rule['max_disagreement']}"
        )
        lines.append(
            f"| {idx} | {item['score']:.2f} | {item['override_count']} | "
            f"{guard_metrics.get('false_positives', 'NA')} | {guard_metrics.get('false_negatives', 'NA')} | "
            f"{round2_delta.get('fixed_step7_fn', 0)} | `{rule_short}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            "## Output Files",
            "",
            f"- Rules: `{report['rules_path']}`",
            f"- JSON report: `{report['json_report']}`",
        ]
    )
    for split_name, path_value in report["tuning_prediction_files"].items():
        lines.append(f"- {split_name} tuned predictions: `{path_value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Round3 precision-guard override rules.")
    parser.add_argument(
        "--tune_set",
        action="append",
        nargs="+",
        required=True,
        help="SPLIT_NAME followed by NAME=PATH prediction specs. Include step7/oof/roberta/electra.",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction_dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--step7_run", default="step7")
    parser.add_argument("--oof_run", default="oof")
    parser.add_argument("--roberta_run", default="roberta")
    parser.add_argument("--electra_run", default="electra")
    parser.add_argument("--guard_split", default="guard_dev")
    parser.add_argument("--valid_split", default="valid")
    parser.add_argument("--round2_split", default="round2_dev")
    parser.add_argument("--guard_fp_tolerance", type=int, default=0)
    parser.add_argument("--valid_fp_tolerance", type=int, default=0)
    parser.add_argument("--valid_f1_tolerance", type=float, default=0.002)
    parser.add_argument("--oof_thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--roberta_thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--electra_thresholds", default="0.55,0.70,0.85,1.10")
    parser.add_argument("--high_risk_threshold_adds", default="0.00,0.05,0.10,0.15")
    parser.add_argument("--max_disagreements", default="0.25,0.35,0.50,0.75,1.10")
    parser.add_argument("--min_votes", default="2,3")
    parser.add_argument("--min_words", default="0,16,32")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    run_names, tune_rows = load_split_sets(args.tune_set)
    required = {safe_name(args.step7_run), safe_name(args.oof_run), safe_name(args.roberta_run), safe_name(args.electra_run)}
    missing = required - set(run_names)
    if missing:
        raise ValueError(f"Tuning sets are missing required run names: {sorted(missing)}")

    baseline_by_split = {
        split_name: baseline_metrics(rows, args.step7_run)
        for split_name, rows in split_rows(tune_rows).items()
    }

    candidates = []
    for rules in candidate_rules(args, disabled=True) + candidate_rules(args):
        candidates.append(evaluate_candidate(tune_rows, rules, baseline_by_split, args))
    feasible = [item for item in candidates if item["constraints_passed"]]
    selected = max(feasible, key=lambda item: (item["score"], -item["override_count"])) if feasible else max(
        candidates, key=lambda item: (item["score"], -item["override_count"])
    )

    rules = dict(selected["rules"])
    rules["selection_constraints_passed"] = selected["constraints_passed"]
    rules["selection_score"] = selected["score"]
    rules["tuned_on_splits"] = sorted(split_rows(tune_rows))
    rules["run_names"] = run_names

    preds, probs, decisions = apply_precision_guard_rules(tune_rows, rules)
    tuning_prediction_files = {}
    for split_name, rows in split_rows(tune_rows).items():
        indices = [idx for idx, row in enumerate(tune_rows) if row.get("split_name") == split_name]
        split_path = prediction_dir / f"round3_precision_guard_{split_name}_tuned_predictions.jsonl"
        save_jsonl(
            precision_guard_prediction_rows(
                [tune_rows[i] for i in indices],
                [preds[i] for i in indices],
                [probs[i] for i in indices],
                [decisions[i] for i in indices],
            ),
            split_path,
        )
        tuning_prediction_files[split_name] = str(split_path)

    decision = (
        "The selected precision guard can advance to Phase F as the final Round3 candidate."
        if selected["constraints_passed"] and selected["override_count"] > 0
        else "No non-baseline override rule beat the constraints safely; Phase F should retain Step7 as the strict final system."
    )
    top_feasible = sorted(feasible, key=lambda item: (item["score"], -item["override_count"]), reverse=True)[:12]
    selected_rules_json = __import__("json").dumps(rules, ensure_ascii=False, indent=2)
    rules_path = output_dir / "rules.json"
    json_report_path = output_dir / "precision_guard_tuning_report.json"
    report = {
        "config": vars(args),
        "run_names": run_names,
        "baseline_by_split": baseline_by_split,
        "selected": selected,
        "selected_rules_json": selected_rules_json,
        "top_feasible": top_feasible,
        "num_candidates": len(candidates),
        "num_feasible": len(feasible),
        "decision": decision,
        "rules_path": str(rules_path),
        "json_report": str(json_report_path),
        "tuning_prediction_files": tuning_prediction_files,
    }
    write_json(rules, rules_path)
    write_json(report, json_report_path)
    write_report(report, Path(args.report_md))

    print("=" * 70)
    print("Round3 precision guard rules tuned")
    print("=" * 70)
    print(f"Tuning rows: {len(tune_rows)}")
    print(f"Candidates: {len(candidates)} feasible={len(feasible)}")
    print(f"Selected overrides: {selected['override_count']}")
    print(f"Selected score: {selected['score']:.2f}")
    print(f"Rules: {rules_path}")
    print(f"Report: {args.report_md}")


if __name__ == "__main__":
    main()
