import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round4_residual_override"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round4_residual_override_tuning_report.md"

HIGH_RISK_BUCKETS = {
    "poetry_classical",
    "poetry_freeverse",
    "literary_old_prose",
    "literary_short_fragment",
    "academic_formal",
}


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}, line {line_id}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def row_id(row: Dict, index: int) -> str:
    return str(row.get("id", index))


def prob_value(row: Dict, *keys: str) -> float:
    for key in keys:
        if row.get(key) is not None:
            return float(row[key])
    return 0.0


def parse_split_spec(values: Sequence[str]) -> Tuple[str, Dict[str, Path]]:
    if len(values) < 2:
        raise ValueError("--tune_set requires SPLIT followed by NAME=PATH specs.")
    split_name = values[0]
    specs = {}
    for value in values[1:]:
        if "=" not in value:
            raise ValueError(f"Prediction spec must be NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        specs[name.strip()] = Path(path.strip())
    required = {"step7", "round4", "guard"}
    missing = required - set(specs)
    if missing:
        raise ValueError(f"{split_name} missing specs: {sorted(missing)}")
    return split_name, specs


def align_split(split_name: str, specs: Dict[str, Path]) -> List[Dict]:
    step7_rows = load_jsonl(specs["step7"])
    round4_rows = load_jsonl(specs["round4"])
    guard_rows = load_jsonl(specs["guard"])
    round4_by_id = {row_id(row, idx): row for idx, row in enumerate(round4_rows)}
    guard_by_id = {row_id(row, idx): row for idx, row in enumerate(guard_rows)}

    aligned = []
    missing_round4 = 0
    missing_guard = 0
    for idx, step7 in enumerate(step7_rows):
        sample_id = row_id(step7, idx)
        round4 = round4_by_id.get(sample_id)
        guard = guard_by_id.get(sample_id)
        if round4 is None:
            missing_round4 += 1
            continue
        if guard is None:
            missing_guard += 1
            continue
        text = step7.get("text") or round4.get("text") or guard.get("text") or ""
        bucket = (
            step7.get("round4_bucket")
            or step7.get("bucket")
            or round4.get("round4_bucket")
            or round4.get("bucket")
            or (assign_bucket(text) if text else "general_prose")
        )
        aligned.append(
            {
                "id": sample_id,
                "split_name": split_name,
                "label": step7.get("label", round4.get("label")),
                "text": text,
                "bucket": bucket,
                "round4_bucket": round4.get("round4_bucket", step7.get("round4_bucket")),
                "round4_tag": round4.get("round4_tag", step7.get("round4_tag")),
                "step7_prediction": int(step7.get("prediction")),
                "step7_prob": prob_value(step7, "probability", "prob_llm"),
                "round4_prediction": int(round4.get("prediction")),
                "round4_prob": prob_value(round4, "probability", "prob_llm", "p_deberta"),
                "p_human_style": prob_value(guard, "p_human_style"),
            }
        )
    if missing_round4 or missing_guard:
        print(
            f"[Warning] {split_name}: skipped rows missing round4={missing_round4}, "
            f"guard={missing_guard}"
        )
    return aligned


def load_tune_rows(tune_sets: Sequence[Sequence[str]]) -> List[Dict]:
    rows = []
    for values in tune_sets:
        split_name, specs = parse_split_spec(values)
        rows.extend(align_split(split_name, specs))
    return rows


def labels_for(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([int(row["label"]) for row in rows], dtype=int)


def metrics_for(rows: Sequence[Dict], preds: Sequence[int], probs: Sequence[float]) -> Dict:
    labels = labels_for(rows)
    preds = np.array(preds, dtype=int)
    probs = np.array(probs, dtype=float)
    out = {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }
    cm = out["confusion_matrix"]
    out["false_positives"] = int(cm[0][1])
    out["false_negatives"] = int(cm[1][0])
    if len(set(labels.tolist())) < 2:
        out["roc_auc"] = None
        return out
    try:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["roc_auc"] = None
    return out


def split_rows(rows: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for row in rows:
        out[str(row.get("split_name", "unknown"))].append(row)
    return dict(out)


def word_count(text: str) -> int:
    return len(str(text or "").split())


def apply_rules(rows: Sequence[Dict], rules: Dict) -> Tuple[List[int], List[float], List[Dict]]:
    preds = []
    probs = []
    decisions = []
    high_risk = set(rules.get("high_risk_buckets", sorted(HIGH_RISK_BUCKETS)))
    for row in rows:
        pred = int(row["step7_prediction"])
        prob = float(row["step7_prob"])
        bucket = str(row.get("round4_bucket") or row.get("bucket"))
        threshold = float(rules["round4_threshold"])
        if bucket in high_risk:
            threshold += float(rules.get("high_risk_threshold_add", 0.0))
        should_override = (
            int(row["step7_prediction"]) == 0
            and float(row["round4_prob"]) >= threshold
            and float(row["round4_prob"]) - float(row["step7_prob"]) >= float(rules["min_delta"])
            and float(row["p_human_style"]) < float(rules["human_style_veto_threshold"])
            and word_count(str(row.get("text", ""))) >= int(rules["min_words"])
        )
        if should_override:
            pred = 1
            prob = max(prob, float(row["round4_prob"]))
        preds.append(pred)
        probs.append(prob)
        decisions.append(
            {
                "override": bool(should_override),
                "effective_round4_threshold": threshold,
                "bucket": bucket,
            }
        )
    return preds, probs, decisions


def baseline_metrics_by_split(rows: Sequence[Dict]) -> Dict[str, Dict]:
    out = {}
    for split_name, split_group in split_rows(rows).items():
        out[split_name] = metrics_for(
            split_group,
            [int(row["step7_prediction"]) for row in split_group],
            [float(row["step7_prob"]) for row in split_group],
        )
    return out


def override_delta(rows: Sequence[Dict], preds: Sequence[int]) -> Dict:
    fixed_fn = 0
    induced_fp = 0
    broke_correct = 0
    overrides = 0
    for row, pred in zip(rows, preds):
        label = int(row["label"])
        step7_pred = int(row["step7_prediction"])
        pred = int(pred)
        if pred != step7_pred:
            overrides += 1
            if step7_pred != label and pred == label and label == 1:
                fixed_fn += 1
            if step7_pred == label and pred != label and label == 0:
                induced_fp += 1
            if step7_pred == label and pred != label:
                broke_correct += 1
    return {
        "overrides": overrides,
        "fixed_step7_fn": fixed_fn,
        "induced_fp": induced_fp,
        "broke_step7_correct": broke_correct,
    }


def candidate_rules(args) -> List[Dict]:
    rules = [
        {
            "round4_threshold": 1.1,
            "min_delta": 1.1,
            "human_style_veto_threshold": 0.0,
            "high_risk_threshold_add": 0.0,
            "min_words": 0,
            "high_risk_buckets": sorted(HIGH_RISK_BUCKETS),
            "disabled_baseline": True,
        }
    ]
    for threshold, delta, veto, risk_add, min_words in itertools.product(
        float_grid(args.round4_thresholds),
        float_grid(args.min_deltas),
        float_grid(args.human_style_veto_thresholds),
        float_grid(args.high_risk_threshold_adds),
        int_grid(args.min_words),
    ):
        rules.append(
            {
                "round4_threshold": threshold,
                "min_delta": delta,
                "human_style_veto_threshold": veto,
                "high_risk_threshold_add": risk_add,
                "min_words": min_words,
                "high_risk_buckets": sorted(HIGH_RISK_BUCKETS),
                "disabled_baseline": False,
            }
        )
    return rules


def float_grid(value: str) -> List[float]:
    return [float(part) for part in str(value).split(",") if part.strip()]


def int_grid(value: str) -> List[int]:
    return [int(part) for part in str(value).split(",") if part.strip()]


def evaluate_candidate(rows: Sequence[Dict], rules: Dict, baseline: Dict[str, Dict], args) -> Dict:
    preds, probs, decisions = apply_rules(rows, rules)
    by_split = {}
    deltas = {}
    for split_name, split_group in split_rows(rows).items():
        indices = [idx for idx, row in enumerate(rows) if row.get("split_name") == split_name]
        split_preds = [preds[idx] for idx in indices]
        split_probs = [probs[idx] for idx in indices]
        by_split[split_name] = metrics_for(split_group, split_preds, split_probs)
        deltas[split_name] = override_delta(split_group, split_preds)

    constraints = []
    if args.hardneg_split in by_split:
        constraints.append(
            by_split[args.hardneg_split]["false_positives"]
            <= baseline[args.hardneg_split]["false_positives"] + args.hardneg_fp_tolerance
        )
    if args.internal_split in by_split:
        constraints.append(by_split[args.internal_split]["f1"] >= baseline[args.internal_split]["f1"] - args.internal_f1_tolerance)
        constraints.append(
            by_split[args.internal_split]["false_positives"]
            <= baseline[args.internal_split]["false_positives"] + args.internal_fp_tolerance
        )
    if args.hardpos_split in by_split:
        constraints.append(by_split[args.hardpos_split]["recall"] >= baseline[args.hardpos_split]["recall"])

    score = 0.0
    for split_name, delta in deltas.items():
        if split_name == args.hardpos_split:
            score += 3.0 * delta["fixed_step7_fn"] - 0.02 * delta["overrides"]
        elif split_name == args.hardneg_split:
            score -= 7.0 * delta["induced_fp"] + 0.02 * delta["overrides"]
        elif split_name == args.internal_split:
            score += 1.0 * delta["fixed_step7_fn"] - 4.0 * delta["induced_fp"] - 0.02 * delta["overrides"]
        else:
            score += 0.5 * delta["fixed_step7_fn"] - 2.0 * delta["induced_fp"]

    return {
        "rules": rules,
        "score": float(score),
        "constraints_passed": bool(all(constraints)) if constraints else True,
        "metrics_by_split": by_split,
        "deltas_by_split": deltas,
        "total_overrides": int(sum(delta["overrides"] for delta in deltas.values())),
    }


def prediction_rows(rows: Sequence[Dict], preds: Sequence[int], probs: Sequence[float], decisions: Sequence[Dict]) -> List[Dict]:
    out = []
    for row, pred, prob, decision in zip(rows, preds, probs, decisions):
        item = dict(row)
        item["prediction"] = int(pred)
        item["probability"] = float(prob)
        item["prob_llm"] = float(prob)
        item["round4_override"] = bool(decision["override"])
        item["effective_round4_threshold"] = float(decision["effective_round4_threshold"])
        out.append(item)
    return out


def write_report(report: Dict, path: Path) -> None:
    selected = report["selected"]
    lines = [
        "# Round4 Residual Override Tuning Report",
        "",
        "Teacher-test labels are not used here. The default prediction is Step7; rules only allow conservative human -> LLM overrides.",
        "",
        "## Selected Rule",
        "",
        "```json",
        json.dumps(selected["rules"], ensure_ascii=False, indent=2),
        "```",
        "",
        f"- constraints passed: {selected['constraints_passed']}",
        f"- score: {selected['score']:.4f}",
        f"- total overrides: {selected['total_overrides']}",
        "",
        "## Metrics",
        "",
        "| Split | n | Accuracy | Precision | Recall | F1 | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name, metrics in selected["metrics_by_split"].items():
        lines.append(
            f"| {split_name} | {metrics['num_samples']} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['false_positives']} | {metrics['false_negatives']} |"
        )
    lines.extend(["", "## Step7 Baseline", ""])
    lines.append("| Split | n | Accuracy | Precision | Recall | F1 | FP | FN |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for split_name, metrics in report["baseline_by_split"].items():
        lines.append(
            f"| {split_name} | {metrics['num_samples']} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['false_positives']} | {metrics['false_negatives']} |"
        )
    lines.extend(["", "## Override Delta", ""])
    lines.append("| Split | Overrides | Fixed Step7 FN | Induced FP | Broke Step7 Correct |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for split_name, delta in selected["deltas_by_split"].items():
        lines.append(
            f"| {split_name} | {delta['overrides']} | {delta['fixed_step7_fn']} | "
            f"{delta['induced_fp']} | {delta['broke_step7_correct']} |"
        )
    lines.extend(["", "## Decision", "", report["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Round4 residual override rules.")
    parser.add_argument("--tune_set", action="append", nargs="+", required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction_dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--internal_split", default="internal_test")
    parser.add_argument("--hardpos_split", default="hardpos")
    parser.add_argument("--hardneg_split", default="hardneg")
    parser.add_argument("--hardneg_fp_tolerance", type=int, default=0)
    parser.add_argument("--internal_fp_tolerance", type=int, default=0)
    parser.add_argument("--internal_f1_tolerance", type=float, default=0.002)
    parser.add_argument("--round4_thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--min_deltas", default="0.00,0.05,0.10,0.15,0.20")
    parser.add_argument("--human_style_veto_thresholds", default="0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--high_risk_threshold_adds", default="0.00,0.05,0.10,0.15")
    parser.add_argument("--min_words", default="0,16,32,48")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    rows = load_tune_rows(args.tune_set)
    if not rows:
        raise ValueError("No aligned tuning rows found.")
    baseline = baseline_metrics_by_split(rows)

    candidates = [evaluate_candidate(rows, rules, baseline, args) for rules in candidate_rules(args)]
    feasible = [item for item in candidates if item["constraints_passed"]]
    ranked = sorted(feasible or candidates, key=lambda item: (item["constraints_passed"], item["score"], -item["total_overrides"]), reverse=True)
    selected = ranked[0]
    rules_path = output_dir / "rules.json"
    write_json(selected["rules"], rules_path)

    preds, probs, decisions = apply_rules(rows, selected["rules"])
    tuned_files = {}
    for split_name, split_group in split_rows(rows).items():
        indices = [idx for idx, row in enumerate(rows) if row.get("split_name") == split_name]
        out_rows = prediction_rows(
            split_group,
            [preds[idx] for idx in indices],
            [probs[idx] for idx in indices],
            [decisions[idx] for idx in indices],
        )
        out_path = prediction_dir / f"round4_residual_override_{split_name}_predictions.jsonl"
        save_jsonl(out_rows, out_path)
        tuned_files[split_name] = str(out_path)

    decision = (
        "Advance only if this rule improves hard-positive recall without increasing hard-negative FP "
        "and does not regress internal-test F1 beyond tolerance."
    )
    report = {
        "selected": selected,
        "baseline_by_split": baseline,
        "rules_path": str(rules_path),
        "tuned_prediction_files": tuned_files,
        "num_candidates": len(candidates),
        "num_feasible": len(feasible),
        "decision": decision,
        "config": vars(args),
    }
    write_json(report, output_dir / "residual_override_tuning_report.json")
    write_report(report, Path(args.report_md))

    print("=" * 70)
    print("Round4 residual override tuning complete")
    print("=" * 70)
    print(f"Aligned rows: {len(rows)}")
    print(f"Candidates: {len(candidates)} feasible={len(feasible)}")
    print(f"Rules: {rules_path}")
    print(f"Report: {args.report_md}")
    for split_name, metrics in selected["metrics_by_split"].items():
        delta = selected["deltas_by_split"][split_name]
        print(
            f"{split_name}: f1={metrics['f1']:.4f} FP={metrics['false_positives']} FN={metrics['false_negatives']} "
            f"overrides={delta['overrides']} fixed_FN={delta['fixed_step7_fn']} induced_FP={delta['induced_fp']}"
        )


if __name__ == "__main__":
    main()
