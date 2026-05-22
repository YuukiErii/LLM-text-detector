import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round5_residual_override"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round5_residual_override_tuning_report.md"

BUCKET_GROUPS = {
    "all": [],
    "planned_high_risk": [
        "academic_formal",
        "literary_old_prose",
        "literary_short_fragment",
        "poetry_classical",
        "poetry_freeverse",
    ],
    "literary_poetry": [
        "literary_old_prose",
        "literary_short_fragment",
        "poetry_classical",
        "poetry_freeverse",
    ],
    "old_short": ["literary_old_prose", "literary_short_fragment"],
    "poetry": ["poetry_classical", "poetry_freeverse"],
    "academic": ["academic_formal"],
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


def row_id(row: Dict, index: int) -> str:
    value = row.get("id")
    if value is None or value == "":
        return str(index)
    return str(value)


def prob_value(row: Dict, *keys: str) -> float:
    for key in keys:
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
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
    required = {"step7", "round5", "human_guard", "flip_guard"}
    missing = required - set(specs)
    if missing:
        raise ValueError(f"{split_name} missing specs: {sorted(missing)}")
    return split_name, specs


def indexed(rows: Sequence[Dict]) -> Dict[str, Dict]:
    return {row_id(row, idx): row for idx, row in enumerate(rows)}


def align_split(split_name: str, specs: Dict[str, Path]) -> List[Dict]:
    step7_rows = load_jsonl(specs["step7"])
    round5_rows = load_jsonl(specs["round5"])
    human_guard_rows = load_jsonl(specs["human_guard"])
    flip_guard_rows = load_jsonl(specs["flip_guard"])
    round5_by_id = indexed(round5_rows)
    human_guard_by_id = indexed(human_guard_rows)
    flip_guard_by_id = indexed(flip_guard_rows)

    aligned = []
    missing_round5 = 0
    missing_human_guard = 0
    missing_flip_guard = 0
    for idx, step7 in enumerate(step7_rows):
        sample_id = row_id(step7, idx)
        round5 = round5_by_id.get(sample_id)
        human_guard = human_guard_by_id.get(sample_id)
        flip_guard = flip_guard_by_id.get(sample_id)
        if round5 is None:
            missing_round5 += 1
            continue
        if human_guard is None:
            missing_human_guard += 1
            continue
        if flip_guard is None:
            missing_flip_guard += 1
            continue
        text = str(step7.get("text") or round5.get("text") or flip_guard.get("text") or "")
        bucket = (
            step7.get("round4_bucket")
            or step7.get("bucket")
            or round5.get("round4_bucket")
            or round5.get("bucket")
            or flip_guard.get("round4_bucket")
            or flip_guard.get("bucket")
            or (assign_bucket(text) if text else "general_prose")
        )
        text_bucket = (
            step7.get("bucket")
            or round5.get("bucket")
            or flip_guard.get("bucket")
            or (assign_bucket(text) if text else "general_prose")
        )
        label = step7.get("label", round5.get("label"))
        if label not in [0, 1]:
            continue
        aligned.append(
            {
                "id": sample_id,
                "pair_id": step7.get("pair_id") or round5.get("pair_id") or flip_guard.get("pair_id") or "",
                "split_name": split_name,
                "label": int(label),
                "text": text,
                "domain": step7.get("domain") or round5.get("domain") or flip_guard.get("domain") or "unknown",
                "generator": step7.get("generator") or round5.get("generator") or flip_guard.get("generator") or "unknown",
                "source": step7.get("source") or round5.get("source") or flip_guard.get("source") or "unknown",
                "bucket": str(text_bucket),
                "round4_bucket": str(bucket),
                "round4_tag": round5.get("round4_tag") or flip_guard.get("round4_tag") or step7.get("round4_tag") or "",
                "step7_prediction": int(step7.get("prediction")),
                "step7_prob": prob_value(step7, "prob_llm", "probability", "score"),
                "round5_prediction": int(round5.get("prediction")),
                "round5_prob": prob_value(round5, "prob_llm", "probability", "p_deberta", "score"),
                "round5_prob_delta": prob_value(round5, "prob_llm", "probability", "p_deberta", "score")
                - prob_value(step7, "prob_llm", "probability", "score"),
                "p_human_style": prob_value(human_guard, "p_human_style"),
                "human_style_veto": int(human_guard.get("human_style_veto", 0) or 0),
                "p_unsafe_override": prob_value(flip_guard, "p_unsafe_override"),
                "flip_guard_veto": int(flip_guard.get("flip_guard_veto", 0) or 0),
            }
        )
    if missing_round5 or missing_human_guard or missing_flip_guard:
        print(
            f"[Warning] {split_name}: skipped rows missing round5={missing_round5}, "
            f"human_guard={missing_human_guard}, flip_guard={missing_flip_guard}"
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
    allowed_buckets = set(rules.get("allowed_buckets") or [])
    for row in rows:
        pred = int(row["step7_prediction"])
        prob = float(row["step7_prob"])
        bucket = str(row.get("round4_bucket") or row.get("bucket") or "unknown")
        bucket_allowed = True if not allowed_buckets else bucket in allowed_buckets
        should_override = (
            int(row["step7_prediction"]) == 0
            and int(row["round5_prediction"]) == 1
            and bucket_allowed
            and float(row["round5_prob"]) >= float(rules["round5_threshold"])
            and float(row["round5_prob_delta"]) >= float(rules["min_delta"])
            and float(row["p_unsafe_override"]) <= float(rules["flip_guard_unsafe_max"])
            and float(row["p_human_style"]) <= float(rules["human_style_veto_max"])
            and word_count(str(row.get("text", ""))) >= int(rules["min_words"])
        )
        if should_override:
            pred = 1
            prob = max(prob, float(row["round5_prob"]))
        preds.append(pred)
        probs.append(prob)
        decisions.append(
            {
                "override": bool(should_override),
                "bucket_allowed": bool(bucket_allowed),
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
    broke_step7_correct = 0
    overrides = 0
    bucket_counts = Counter()
    fixed_by_bucket = Counter()
    induced_by_bucket = Counter()
    for row, pred in zip(rows, preds):
        label = int(row["label"])
        step7_pred = int(row["step7_prediction"])
        pred = int(pred)
        if pred != step7_pred:
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


def int_grid(value: str) -> List[int]:
    return [int(part) for part in str(value).split(",") if part.strip()]


def bucket_group_grid(value: str) -> List[str]:
    names = [part.strip() for part in str(value).split(",") if part.strip()]
    unknown = sorted(set(names) - set(BUCKET_GROUPS))
    if unknown:
        raise ValueError(f"Unknown bucket groups: {unknown}. Available: {sorted(BUCKET_GROUPS)}")
    return names


def candidate_rules(args) -> List[Dict]:
    rules = [
        {
            "round5_threshold": 1.1,
            "min_delta": 1.1,
            "flip_guard_unsafe_max": -1.0,
            "human_style_veto_max": -1.0,
            "min_words": 0,
            "bucket_group": "disabled",
            "allowed_buckets": [],
            "disabled_baseline": True,
        }
    ]
    for threshold, delta, flip_max, human_max, min_words, bucket_group in itertools.product(
        float_grid(args.round5_thresholds),
        float_grid(args.min_deltas),
        float_grid(args.flip_guard_unsafe_max),
        float_grid(args.human_style_veto_max),
        int_grid(args.min_words),
        bucket_group_grid(args.bucket_groups),
    ):
        rules.append(
            {
                "round5_threshold": threshold,
                "min_delta": delta,
                "flip_guard_unsafe_max": flip_max,
                "human_style_veto_max": human_max,
                "min_words": min_words,
                "bucket_group": bucket_group,
                "allowed_buckets": BUCKET_GROUPS[bucket_group],
                "disabled_baseline": False,
            }
        )
    return rules


def evaluate_candidate(rows: Sequence[Dict], rules: Dict, baseline: Dict[str, Dict], args) -> Dict:
    preds, probs, _ = apply_rules(rows, rules)
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
        constraints.append(deltas[args.hardneg_split]["induced_fp"] <= args.hardneg_induced_fp_tolerance)
        constraints.append(by_split[args.hardneg_split]["false_positives"] <= baseline[args.hardneg_split]["false_positives"])
    if args.internal_split in by_split:
        constraints.append(by_split[args.internal_split]["f1"] >= args.min_internal_f1)
        constraints.append(
            by_split[args.internal_split]["false_positives"]
            <= baseline[args.internal_split]["false_positives"] + args.internal_fp_tolerance
        )
        constraints.append(deltas[args.internal_split]["induced_fp"] <= args.internal_induced_fp_tolerance)
    if args.hardpos_split in by_split:
        constraints.append(deltas[args.hardpos_split]["fixed_step7_fn"] >= args.min_hardpos_fixed_fn)
    constraints.append(sum(delta["overrides"] for delta in deltas.values()) > 0)

    score = 0.0
    for split_name, delta in deltas.items():
        if split_name == args.hardpos_split:
            score += 5.0 * delta["fixed_step7_fn"] - 0.02 * delta["overrides"]
        elif split_name == args.hardneg_split:
            score -= 20.0 * delta["induced_fp"] + 0.05 * delta["overrides"]
        elif split_name == args.internal_split:
            score += 2.0 * delta["fixed_step7_fn"] - 10.0 * delta["induced_fp"] - 0.02 * delta["overrides"]
        else:
            score += delta["fixed_step7_fn"] - 5.0 * delta["induced_fp"]

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
        item["round5_override"] = bool(decision["override"])
        item["round5_override_bucket_allowed"] = bool(decision["bucket_allowed"])
        item["round5_override_bucket"] = decision["bucket"]
        out.append(item)
    return out


def write_report(report: Dict, path: Path) -> None:
    selected = report["selected"]
    lines = [
        "# Round5 Residual Override Tuning Report",
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
    lines.extend(["", "## Bucket Delta", ""])
    lines.append("| Split | Type | Bucket | Count |")
    lines.append("| --- | --- | --- | ---: |")
    for split_name, delta in selected["deltas_by_split"].items():
        for key, label in [
            ("overrides_by_bucket", "override"),
            ("fixed_step7_fn_by_bucket", "fixed_fn"),
            ("induced_fp_by_bucket", "induced_fp"),
        ]:
            for bucket, count in sorted(delta.get(key, {}).items()):
                lines.append(f"| {split_name} | {label} | {bucket} | {count} |")
    lines.extend(["", "## Decision", "", report["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Round5 FP-safe residual override rules.")
    parser.add_argument("--tune_set", action="append", nargs="+", required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction_dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--internal_split", default="internal_test")
    parser.add_argument("--hardpos_split", default="hardpos")
    parser.add_argument("--hardneg_split", default="hardneg")
    parser.add_argument("--min_internal_f1", type=float, default=0.9544)
    parser.add_argument("--internal_fp_tolerance", type=int, default=2)
    parser.add_argument("--internal_induced_fp_tolerance", type=int, default=2)
    parser.add_argument("--hardneg_induced_fp_tolerance", type=int, default=0)
    parser.add_argument("--min_hardpos_fixed_fn", type=int, default=30)
    parser.add_argument("--round5_thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--min_deltas", default="0.00,0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--flip_guard_unsafe_max", default="0.20,0.25,0.29,0.30,0.35,0.40")
    parser.add_argument("--human_style_veto_max", default="0.65,0.70,0.75,0.80")
    parser.add_argument("--min_words", default="0,16,32,48")
    parser.add_argument(
        "--bucket_groups",
        default="all,planned_high_risk,literary_poetry,old_short,poetry,academic",
    )
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
    for split_name, split_group in split_rows(rows).items():
        indices = [idx for idx, row in enumerate(rows) if row.get("split_name") == split_name]
        out_rows = prediction_rows(
            split_group,
            [preds[idx] for idx in indices],
            [probs[idx] for idx in indices],
            [decisions[idx] for idx in indices],
        )
        out_path = prediction_dir / f"round5_residual_override_{split_name}_predictions.jsonl"
        save_jsonl(out_rows, out_path)
        tuned_files[split_name] = str(out_path)

    decision = (
        "PROMOTE_TO_PHASE6_GATE_REPORT = yes"
        if selected["constraints_passed"] and selected["total_overrides"] > 0
        else "PROMOTE_TO_PHASE6_GATE_REPORT = no; keep Step7 as final and return to residual data or guard work."
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
    print("Round5 residual override tuning complete")
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
    print(decision)


if __name__ == "__main__":
    main()
