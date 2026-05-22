import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round6_safe_override"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round6_safe_override_tuning_report.md"


BUCKET_POLICIES = {
    "old_short": {"base": {"literary_old_prose", "literary_short_fragment"}, "general": set()},
    "short_only": {"base": {"literary_short_fragment"}, "general": set()},
    "general_strict": {"base": set(), "general": {"general_prose"}},
    "old_short_plus_general_strict": {
        "base": {"literary_old_prose", "literary_short_fragment"},
        "general": {"general_prose"},
    },
}


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}, line {line_id}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prob(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key)
        if value in [None, ""]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def split_rows(rows: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for row in rows:
        out[str(row.get("split") or row.get("split_name") or "unknown")].append(row)
    return dict(out)


def labels_for(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([int(row["label"]) for row in rows], dtype=int)


def metrics_for(rows: Sequence[Dict], preds: Sequence[int], probs: Sequence[float]) -> Dict:
    labels = labels_for(rows)
    preds = np.array(preds, dtype=int)
    probs = np.array(probs, dtype=float)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    out = {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": cm,
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
    }
    if len(set(labels.tolist())) < 2:
        out["roc_auc"] = None
    else:
        try:
            out["roc_auc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            out["roc_auc"] = None
    return out


def baseline_metrics_by_split(rows: Sequence[Dict]) -> Dict[str, Dict]:
    out = {}
    for split_name, group in split_rows(rows).items():
        out[split_name] = metrics_for(
            group,
            [int(row["step7_pred"]) for row in group],
            [prob(row, "step7_prob") for row in group],
        )
    return out


def row_is_general_strict_allowed(row: Dict, rules: Dict) -> bool:
    if rules.get("disabled_baseline"):
        return False
    bucket = str(row.get("round4_bucket") or row.get("bucket") or "unknown")
    policy = BUCKET_POLICIES[rules["bucket_policy"]]
    if bucket in policy["base"]:
        return True
    if bucket in policy["general"]:
        return (
            prob(row, "p_safe_override") >= float(rules["general_p_safe_min"])
            and prob(row, "p_unsafe_override", 1.0) <= float(rules["general_p_unsafe_max"])
            and prob(row, "round4_prob") >= float(rules["general_round4_threshold"])
            and prob(row, "prob_delta") >= float(rules["general_min_delta"])
        )
    return False


def apply_rules(rows: Sequence[Dict], rules: Dict) -> Tuple[List[int], List[float], List[Dict]]:
    preds = []
    probs = []
    decisions = []
    for row in rows:
        pred = int(row["step7_pred"])
        final_prob = prob(row, "step7_prob")
        should_override = (
            int(row.get("step7_pred", 0)) == 0
            and int(row.get("round4_pred", 0)) == 1
            and row_is_general_strict_allowed(row, rules)
            and prob(row, "p_safe_override") >= float(rules["p_safe_min"])
            and prob(row, "p_unsafe_override", 1.0) <= float(rules["p_unsafe_max"])
            and prob(row, "round4_prob") >= float(rules["round4_threshold"])
            and prob(row, "prob_delta") >= float(rules["min_delta"])
        )
        if should_override:
            pred = 1
            final_prob = max(final_prob, prob(row, "round4_prob"))
        preds.append(pred)
        probs.append(final_prob)
        decisions.append(
            {
                "round6_override": bool(should_override),
                "round6_policy": rules["bucket_policy"],
                "round6_bucket": str(row.get("round4_bucket") or row.get("bucket") or "unknown"),
            }
        )
    return preds, probs, decisions


def override_delta(rows: Sequence[Dict], preds: Sequence[int]) -> Dict:
    fixed_fn = induced_fp = broke_step7_correct = overrides = 0
    bucket_counts = Counter()
    fixed_by_bucket = Counter()
    induced_by_bucket = Counter()
    for row, pred in zip(rows, preds):
        label = int(row["label"])
        step7_pred = int(row["step7_pred"])
        pred = int(pred)
        if pred == step7_pred:
            continue
        overrides += 1
        bucket = str(row.get("round4_bucket") or row.get("bucket") or "unknown")
        bucket_counts[bucket] += 1
        if step7_pred != label and pred == label and label == 1:
            fixed_fn += 1
            fixed_by_bucket[bucket] += 1
        if step7_pred == label and pred != label and label == 0:
            induced_fp += 1
            induced_by_bucket[bucket] += 1
        if step7_pred == label and pred != label:
            broke_step7_correct += 1
    return {
        "overrides": overrides,
        "fixed_step7_fn": fixed_fn,
        "induced_fp": induced_fp,
        "broke_step7_correct": broke_step7_correct,
        "overrides_by_bucket": dict(bucket_counts),
        "fixed_step7_fn_by_bucket": dict(fixed_by_bucket),
        "induced_fp_by_bucket": dict(induced_by_bucket),
    }


def float_grid(value: str) -> List[float]:
    return [float(part) for part in str(value).split(",") if part.strip()]


def candidate_rules(args) -> List[Dict]:
    rules = [
        {
            "p_safe_min": 1.1,
            "p_unsafe_max": -1.0,
            "round4_threshold": 1.1,
            "min_delta": 1.1,
            "bucket_policy": "disabled",
            "general_p_safe_min": 1.1,
            "general_p_unsafe_max": -1.0,
            "general_round4_threshold": 1.1,
            "general_min_delta": 1.1,
            "disabled_baseline": True,
        }
    ]
    for p_safe, p_unsafe, round4_threshold, min_delta, policy in itertools.product(
        float_grid(args.p_safe_min),
        float_grid(args.p_unsafe_max),
        float_grid(args.round4_threshold),
        float_grid(args.min_delta),
        [part.strip() for part in args.bucket_policy.split(",") if part.strip()],
    ):
        if policy not in BUCKET_POLICIES:
            raise ValueError(f"Unknown bucket policy: {policy}")
        rules.append(
            {
                "p_safe_min": p_safe,
                "p_unsafe_max": p_unsafe,
                "round4_threshold": round4_threshold,
                "min_delta": min_delta,
                "bucket_policy": policy,
                "general_p_safe_min": min(0.99, p_safe + args.general_strict_p_safe_margin),
                "general_p_unsafe_max": max(0.0, p_unsafe - args.general_strict_p_unsafe_margin),
                "general_round4_threshold": min(0.99, round4_threshold + args.general_strict_round4_margin),
                "general_min_delta": min_delta + args.general_strict_delta_margin,
                "disabled_baseline": False,
            }
        )
    return rules


def evaluate_candidate(rows: Sequence[Dict], rules: Dict, baseline: Dict[str, Dict], args) -> Dict:
    preds, probs, _ = apply_rules(rows, rules)
    by_split = {}
    deltas = {}
    row_splits = split_rows(rows)
    for split_name, group in row_splits.items():
        indices = [idx for idx, row in enumerate(rows) if (row.get("split") or row.get("split_name")) == split_name]
        split_preds = [preds[idx] for idx in indices]
        split_probs = [probs[idx] for idx in indices]
        by_split[split_name] = metrics_for(group, split_preds, split_probs)
        deltas[split_name] = override_delta(group, split_preds)

    constraints = [sum(delta["overrides"] for delta in deltas.values()) > 0]
    if args.hardneg_split in by_split:
        constraints.append(deltas[args.hardneg_split]["induced_fp"] <= args.hardneg_induced_fp_tolerance)
        constraints.append(by_split[args.hardneg_split]["false_positives"] <= baseline[args.hardneg_split]["false_positives"])
    if args.internal_split in by_split:
        constraints.append(deltas[args.internal_split]["induced_fp"] <= args.internal_induced_fp_tolerance)
        constraints.append(by_split[args.internal_split]["f1"] >= args.min_internal_f1)
        constraints.append(
            by_split[args.internal_split]["false_positives"]
            <= baseline[args.internal_split]["false_positives"] + args.internal_fp_tolerance
        )
    if args.hardpos_split in by_split:
        constraints.append(deltas[args.hardpos_split]["fixed_step7_fn"] >= args.min_hardpos_fixed_fn)

    score = 0.0
    for split_name, delta in deltas.items():
        if split_name == args.hardpos_split:
            score += 5.0 * delta["fixed_step7_fn"] - 0.01 * delta["overrides"]
        elif split_name == args.hardneg_split:
            score -= 25.0 * delta["induced_fp"] + 0.05 * delta["overrides"]
        elif split_name == args.internal_split:
            score += 2.0 * delta["fixed_step7_fn"] - 20.0 * delta["induced_fp"] - 0.02 * delta["overrides"]
    return {
        "rules": rules,
        "score": float(score),
        "constraints_passed": bool(all(constraints)),
        "metrics_by_split": by_split,
        "deltas_by_split": deltas,
        "total_overrides": int(sum(delta["overrides"] for delta in deltas.values())),
    }


def prediction_rows(rows: Sequence[Dict], preds: Sequence[int], probs: Sequence[float], decisions: Sequence[Dict]) -> List[Dict]:
    out = []
    for row, pred, final_prob, decision in zip(rows, preds, probs, decisions):
        item = dict(row)
        item["prediction"] = int(pred)
        item["probability"] = float(final_prob)
        item["prob_llm"] = float(final_prob)
        item.update(decision)
        out.append(item)
    return out


def write_report(report: Dict, path: Path) -> None:
    selected = report["selected"]
    lines = [
        "# Round6 Safe Override Tuning Report",
        "",
        "Teacher-test labels are not used here. Step7 remains the default prediction; this search only allows local Step7-human -> LLM overrides.",
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
    lines.extend(["", "## Override Delta", ""])
    lines.append("| Split | Overrides | Fixed Step7 FN | Induced FP | Broke Step7 Correct |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for split_name, delta in selected["deltas_by_split"].items():
        lines.append(
            f"| {split_name} | {delta['overrides']} | {delta['fixed_step7_fn']} | "
            f"{delta['induced_fp']} | {delta['broke_step7_correct']} |"
        )
    lines.extend(["", "## Search Summary", ""])
    lines.append(f"- candidate rules: {report['num_candidates']}")
    lines.append(f"- feasible rules: {report['num_feasible']}")
    lines.extend(["", "## Decision", "", "```text", report["decision"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_split_spec(values: Sequence[str]) -> Tuple[str, Path]:
    if len(values) != 2:
        raise ValueError("--tune_set requires SPLIT PATH")
    return values[0], Path(values[1])


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Round6 safe override rules.")
    parser.add_argument("--tune_set", action="append", nargs=2, required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction_dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--internal_split", default="internal_test")
    parser.add_argument("--hardpos_split", default="hardpos")
    parser.add_argument("--hardneg_split", default="hardneg")
    parser.add_argument("--min_internal_f1", type=float, default=0.9564)
    parser.add_argument("--internal_fp_tolerance", type=int, default=2)
    parser.add_argument("--internal_induced_fp_tolerance", type=int, default=1)
    parser.add_argument("--hardneg_induced_fp_tolerance", type=int, default=0)
    parser.add_argument("--min_hardpos_fixed_fn", type=int, default=70)
    parser.add_argument("--p_safe_min", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--p_unsafe_max", default="0.20,0.25,0.30,0.35,0.40,0.45")
    parser.add_argument("--round4_threshold", default="0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--min_delta", default="0.00,0.05,0.10,0.15")
    parser.add_argument(
        "--bucket_policy",
        default="old_short,short_only,general_strict,old_short_plus_general_strict",
    )
    parser.add_argument("--general_strict_p_safe_margin", type=float, default=0.10)
    parser.add_argument("--general_strict_p_unsafe_margin", type=float, default=0.10)
    parser.add_argument("--general_strict_round4_margin", type=float, default=0.05)
    parser.add_argument("--general_strict_delta_margin", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for values in args.tune_set:
        split_name, path = parse_split_spec(values)
        for row in load_jsonl(path):
            item = dict(row)
            item["split"] = split_name
            rows.append(item)
    if not rows:
        raise ValueError("No Round6 tuning rows found.")

    baseline = baseline_metrics_by_split(rows)
    candidates = [evaluate_candidate(rows, rules, baseline, args) for rules in candidate_rules(args)]
    feasible = [item for item in candidates if item["constraints_passed"]]
    ranked = sorted(
        feasible or candidates,
        key=lambda item: (item["constraints_passed"], item["score"], item["total_overrides"]),
        reverse=True,
    )
    selected = ranked[0]
    rules_path = output_dir / "rules.json"
    write_json(selected["rules"], rules_path)

    preds, probs, decisions = apply_rules(rows, selected["rules"])
    tuned_files = {}
    for split_name, group in split_rows(rows).items():
        indices = [idx for idx, row in enumerate(rows) if row.get("split") == split_name]
        out_rows = prediction_rows(
            group,
            [preds[idx] for idx in indices],
            [probs[idx] for idx in indices],
            [decisions[idx] for idx in indices],
        )
        out_path = prediction_dir / f"round6_safe_override_{split_name}_predictions.jsonl"
        save_jsonl(out_rows, out_path)
        tuned_files[split_name] = str(out_path)

    decision = (
        "PROMOTE_TO_ROUND6_GATE_REPORT = yes"
        if selected["constraints_passed"] and selected["total_overrides"] > 0
        else "PROMOTE_TO_ROUND6_GATE_REPORT = no; keep Step7 and improve the safe selector or data."
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
    write_json(report, output_dir / "tuning_report.json")
    write_report(report, Path(args.report_md))

    print("=" * 70)
    print("Round6 safe override tuning complete")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Candidates: {len(candidates)} feasible={len(feasible)}")
    print(f"Rules: {rules_path}")
    print(f"Report: {args.report_md}")
    for split_name, metrics in selected["metrics_by_split"].items():
        delta = selected["deltas_by_split"][split_name]
        print(
            f"{split_name}: f1={metrics['f1']:.4f} FP={metrics['false_positives']} "
            f"FN={metrics['false_negatives']} overrides={delta['overrides']} "
            f"fixed_FN={delta['fixed_step7_fn']} induced_FP={delta['induced_fp']}"
        )
    print(decision)


if __name__ == "__main__":
    main()
