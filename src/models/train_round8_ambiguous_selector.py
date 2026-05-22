import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features  # noqa: E402


DEFAULT_TRAIN = PROJECT_ROOT / "outputs" / "predictions" / "round8_ambiguous_selector_train_scored.jsonl"
DEFAULT_DEV = PROJECT_ROOT / "outputs" / "predictions" / "round8_ambiguous_selector_dev_scored.jsonl"
DEFAULT_PROBE = PROJECT_ROOT / "outputs" / "predictions" / "round8_ambiguous_selector_probe_scored.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round8_ambiguous_selector"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round8_ambiguous_selector_report.md"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labeled_rows(path: Path) -> List[Dict]:
    rows = []
    for row in load_records(path):
        if row.get("label") in [0, 1] and isinstance(row.get("text"), str) and row.get("text").strip():
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"No labeled rows found in {path}")
    return rows


def numeric(row: Dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in [None, ""]:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def has_numeric(row: Dict, key: str) -> float:
    return 0.0 if row.get(key) in [None, ""] else 1.0


def entropy(values: Sequence[float]) -> float:
    eps = 1e-9
    parts = []
    for value in values:
        p = min(1.0 - eps, max(eps, float(value)))
        parts.append(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))
    return float(mean(parts)) if parts else 0.0


def lexical_shape_features(text: str) -> Dict[str, float]:
    text = str(text or "")
    chars = max(1, len(text))
    words = WORD_RE.findall(text)
    lower_words = [word.lower() for word in words]
    long_words = [word for word in words if len(word) >= 9]
    return {
        "uppercase_char_ratio": sum(1 for char in text if char.isalpha() and char.isupper()) / chars,
        "digit_char_ratio": sum(1 for char in text if char.isdigit()) / chars,
        "space_char_ratio": sum(1 for char in text if char.isspace()) / chars,
        "comma_count": float(text.count(",")),
        "colon_count": float(text.count(":")),
        "semicolon_count": float(text.count(";")),
        "paren_count": float(text.count("(") + text.count(")")),
        "quote_count_extra": float(text.count('"') + text.count("'")),
        "long_word_ratio": len(long_words) / max(1, len(words)),
        "avg_word_len": sum(len(word) for word in words) / max(1, len(words)),
        "first_person_count": float(sum(1 for word in lower_words if word in {"i", "me", "my", "mine", "we", "our"})),
    }


def feature_dict(row: Dict) -> Dict:
    text = str(row.get("text") or "")
    bucket = assign_bucket(text)
    p_step7 = numeric(row, "p_step7", numeric(row, "probability"))
    p_tfidf = numeric(row, "p_tfidf")
    p_deberta_step7 = numeric(row, "p_deberta_step7", numeric(row, "p_deberta"))
    p_residual = numeric(row, "p_residual_deberta")
    p_style = numeric(row, "p_stylometry")
    probs = [p_step7, p_tfidf, p_deberta_step7, p_residual, p_style]

    features = text_features(text)
    features.pop("bucket", None)
    features.update(lexical_shape_features(text))
    features.update(
        {
            "bucket": bucket,
            "round4_bucket": str(row.get("round4_bucket") or bucket),
            "domain": str(row.get("domain") or "unknown"),
            "step7_probability_bucket": str(row.get("step7_probability_bucket") or "unknown"),
            "p_step7": p_step7,
            "p_tfidf": p_tfidf,
            "p_deberta_step7": p_deberta_step7,
            "p_residual_deberta": p_residual,
            "p_stylometry": p_style,
            "has_p_tfidf": has_numeric(row, "p_tfidf"),
            "has_p_deberta_step7": has_numeric(row, "p_deberta_step7") or has_numeric(row, "p_deberta"),
            "has_p_residual_deberta": has_numeric(row, "p_residual_deberta"),
            "has_p_stylometry": has_numeric(row, "p_stylometry"),
            "step7_margin_055": abs(p_step7 - 0.55),
            "step7_margin_050": abs(p_step7 - 0.50),
            "residual_margin_050": abs(p_residual - 0.50),
            "stylometry_margin_050": abs(p_style - 0.50),
            "residual_minus_step7": p_residual - p_step7,
            "stylometry_minus_step7": p_style - p_step7,
            "residual_minus_stylometry": p_residual - p_style,
            "tfidf_minus_step7": p_tfidf - p_step7,
            "step7_deberta_minus_tfidf": p_deberta_step7 - p_tfidf,
            "prob_mean": float(mean(probs)),
            "prob_std": float(pstdev(probs)) if len(probs) > 1 else 0.0,
            "prob_min": float(min(probs)),
            "prob_max": float(max(probs)),
            "prob_range": float(max(probs) - min(probs)),
            "prob_entropy": entropy(probs),
            "pred_step7": float(row.get("step7_pred", int(p_step7 >= 0.55))),
            "pred_residual": float(row.get("residual_deberta_pred", int(p_residual >= 0.5))),
            "pred_stylometry": float(row.get("stylometry_pred", int(p_style >= 0.5))),
            "branch_disagreement_count": numeric(row, "branch_disagreement_count"),
        }
    )
    return features


def labels_for(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([int(row["label"]) for row in rows], dtype=int)


def build_model(c_value: float, class_weight: str, seed: int) -> Pipeline:
    resolved_weight = None if class_weight == "none" else class_weight
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight=resolved_weight,
                    max_iter=2000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def metrics_for_preds(rows: Sequence[Dict], preds: Sequence[int], probs: Sequence[float]) -> Dict:
    y_true = labels_for(rows)
    y_pred = np.array(preds, dtype=int)
    y_prob = np.array(probs, dtype=float)
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    out = {
        "num_samples": int(len(rows)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "false_positives": fp,
        "false_negatives": fn,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = None
    return out


def baseline_preds(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([int(row.get("step7_pred", float(row["p_step7"]) >= 0.55)) for row in rows], dtype=int)


def baseline_probs(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([float(row.get("p_step7", row.get("probability", 0.0))) for row in rows], dtype=float)


def system_output(rows: Sequence[Dict], selector_probs: Sequence[float], confidence_threshold: float, low: float, high: float) -> Dict:
    labels = labels_for(rows)
    step7 = baseline_preds(rows)
    step7_probs = baseline_probs(rows)
    selector_probs = np.array(selector_probs, dtype=float)
    selector_preds = (selector_probs >= 0.5).astype(int)
    selector_conf = np.maximum(selector_probs, 1.0 - selector_probs)
    ambiguous = np.array([low <= prob <= high for prob in step7_probs], dtype=bool)
    use_selector = ambiguous & (selector_conf >= confidence_threshold)
    final_preds = np.where(use_selector, selector_preds, step7)
    final_probs = np.where(use_selector, selector_probs, step7_probs)

    fixed_errors = int(((step7 != labels) & (final_preds == labels)).sum())
    broken_correct = int(((step7 == labels) & (final_preds != labels)).sum())
    return {
        "predictions": final_preds,
        "probabilities": final_probs,
        "selector_predictions": selector_preds,
        "selector_confidence": selector_conf,
        "use_selector": use_selector,
        "metrics": metrics_for_preds(rows, final_preds, final_probs),
        "delta": {
            "step7_correct": int((step7 == labels).sum()),
            "final_correct": int((final_preds == labels).sum()),
            "net_correct_gain": int((final_preds == labels).sum() - (step7 == labels).sum()),
            "fixed_step7_errors": fixed_errors,
            "broken_step7_correct": broken_correct,
            "fixed_false_negatives": int(((labels == 1) & (step7 == 0) & (final_preds == 1)).sum()),
            "fixed_false_positives": int(((labels == 0) & (step7 == 1) & (final_preds == 0)).sum()),
            "new_false_positives": int(((labels == 0) & (step7 == 0) & (final_preds == 1)).sum()),
            "new_false_negatives": int(((labels == 1) & (step7 == 1) & (final_preds == 0)).sum()),
            "selector_eligible_rows": int(ambiguous.sum()),
            "selector_used_rows": int(use_selector.sum()),
            "selector_changed_rows": int((use_selector & (selector_preds != step7)).sum()),
            "selector_coverage": float(use_selector.mean()) if len(rows) else 0.0,
        },
    }


def threshold_grid() -> List[float]:
    return [round(float(value), 4) for value in np.linspace(0.50, 0.95, 46)]


def choose_confidence_threshold(rows: Sequence[Dict], selector_probs: Sequence[float], args) -> Dict:
    scored = []
    step7_metrics = metrics_for_preds(rows, baseline_preds(rows), baseline_probs(rows))
    for threshold in threshold_grid():
        output = system_output(rows, selector_probs, threshold, args.ambiguous_low, args.ambiguous_high)
        delta = output["delta"]
        fp_increase = output["metrics"]["false_positives"] - step7_metrics["false_positives"]
        feasible = (
            fp_increase <= args.max_dev_fp_increase
            and delta["net_correct_gain"] >= args.min_dev_net_gain
            and delta["selector_used_rows"] >= args.min_dev_selector_used
        )
        score = (
            1 if feasible else 0,
            delta["net_correct_gain"],
            -max(0, fp_increase),
            output["metrics"]["f1"],
            delta["fixed_false_negatives"],
            -threshold,
        )
        scored.append(
            {
                "threshold": float(threshold),
                "score": list(score),
                "constraints_passed": bool(feasible),
                "fp_increase": int(fp_increase),
                "metrics": output["metrics"],
                "delta": delta,
            }
        )
    selected = max(scored, key=lambda item: tuple(item["score"]))
    return {
        "selected_confidence_threshold": float(selected["threshold"]),
        "selected_constraints_passed": bool(selected["constraints_passed"]),
        "selected": selected,
        "top_candidates": sorted(scored, key=lambda item: tuple(item["score"]), reverse=True)[:20],
        "step7_dev_baseline": step7_metrics,
    }


def prediction_rows(rows: Sequence[Dict], selector_probs: Sequence[float], output: Dict, confidence_threshold: float) -> List[Dict]:
    out = []
    for row, prob, selector_pred, selector_conf, use_selector, final_pred, final_prob in zip(
        rows,
        selector_probs,
        output["selector_predictions"],
        output["selector_confidence"],
        output["use_selector"],
        output["predictions"],
        output["probabilities"],
    ):
        item = dict(row)
        item["p_round8_ambiguous_selector"] = float(prob)
        item["round8_ambiguous_selector_pred"] = int(selector_pred)
        item["round8_ambiguous_selector_confidence"] = float(selector_conf)
        item["round8_ambiguous_selector_confidence_threshold"] = float(confidence_threshold)
        item["round8_ambiguous_selector_used"] = bool(use_selector)
        item["round8_oneshot_prediction"] = int(final_pred)
        item["round8_oneshot_probability"] = float(final_prob)
        out.append(item)
    return out


def split_summary(rows: Sequence[Dict]) -> Dict:
    return {
        "num_rows": len(rows),
        "label_distribution": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "round8_bucket_distribution": dict(sorted(Counter(str(row.get("round8_bucket")) for row in rows).items())),
        "domain_distribution": dict(sorted(Counter(str(row.get("domain")) for row in rows).items())),
        "generator_distribution": dict(sorted(Counter(str(row.get("generator")) for row in rows).items())),
    }


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round8 Ambiguous Selector Report",
        "",
        "This selector is trained only on non-teacher Step7 ambiguous-zone rows.",
        "It is a local repair layer over the frozen Step7 baseline, not a global replacement.",
        "",
        f"Selected selector confidence threshold: `{report['confidence_threshold']:.4f}`",
        f"Dev threshold constraints passed: `{report['threshold_selection']['selected_constraints_passed']}`",
        "",
        "## Metrics",
        "",
        "| Split | System | n | F1 | FP | FN | Confusion |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for split_name, split_metrics in report["metrics"].items():
        for system_name in ["step7", "selector_direct", "round8_oneshot_local"]:
            block = split_metrics[system_name]
            lines.append(
                f"| {split_name} | {system_name} | {block['num_samples']} | {block['f1']:.4f} | "
                f"{block['false_positives']} | {block['false_negatives']} | {block['confusion_matrix']} |"
            )
    lines.extend(["", "## Delta Vs Step7", "", "| Split | Net correct | Fixed FN | New FP | Used | Changed |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for split_name, delta in report["deltas"].items():
        lines.append(
            f"| {split_name} | {delta['net_correct_gain']} | {delta['fixed_false_negatives']} | "
            f"{delta['new_false_positives']} | {delta['selector_used_rows']} | {delta['selector_changed_rows']} |"
        )
    lines.extend(["", "## Decision", "", "```text", report["decision"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Round8 ambiguous-zone selector.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev", default=str(DEFAULT_DEV))
    parser.add_argument("--probe", default=str(DEFAULT_PROBE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction_dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--ambiguous_low", type=float, default=0.35)
    parser.add_argument("--ambiguous_high", type=float, default=0.65)
    parser.add_argument("--max_dev_fp_increase", type=int, default=0)
    parser.add_argument("--min_dev_net_gain", type=int, default=1)
    parser.add_argument("--min_dev_selector_used", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {
        "train": load_labeled_rows(Path(args.train)),
        "dev": load_labeled_rows(Path(args.dev)),
        "probe": load_labeled_rows(Path(args.probe)),
    }
    if len(set(labels_for(rows_by_split["train"]).tolist())) < 2:
        raise ValueError("Ambiguous selector training rows must contain both labels.")

    model = build_model(c_value=args.C, class_weight=args.class_weight, seed=args.seed)
    model.fit([feature_dict(row) for row in rows_by_split["train"]], labels_for(rows_by_split["train"]))

    selector_probs = {
        name: model.predict_proba([feature_dict(row) for row in rows])[:, 1]
        for name, rows in rows_by_split.items()
    }
    threshold_report = choose_confidence_threshold(rows_by_split["dev"], selector_probs["dev"], args)
    confidence_threshold = float(threshold_report["selected_confidence_threshold"])

    metrics = {}
    deltas = {}
    prediction_outputs = {}
    for name, rows in rows_by_split.items():
        local = system_output(rows, selector_probs[name], confidence_threshold, args.ambiguous_low, args.ambiguous_high)
        selector_direct = (selector_probs[name] >= 0.5).astype(int)
        metrics[name] = {
            "step7": metrics_for_preds(rows, baseline_preds(rows), baseline_probs(rows)),
            "selector_direct": metrics_for_preds(rows, selector_direct, selector_probs[name]),
            "round8_oneshot_local": local["metrics"],
        }
        deltas[name] = local["delta"]
        output_path = prediction_dir / f"round8_ambiguous_selector_{name}_predictions.jsonl"
        save_jsonl(prediction_rows(rows, selector_probs[name], local, confidence_threshold), output_path)
        prediction_outputs[name] = str(output_path)

    checks = [
        threshold_report["selected_constraints_passed"],
        deltas["probe"]["net_correct_gain"] >= 0,
        deltas["probe"]["new_false_positives"] <= 1,
    ]
    decision = (
        "USE_FOR_NON_TEACHER_GATE = yes; apply as a local Step7 ambiguous-zone repair and evaluate on original internal_test plus residual_dev/probe."
        if all(checks)
        else "USE_FOR_NON_TEACHER_GATE = diagnostic_only; do not run teacher-test diagnostic unless the wider non-teacher gate passes."
    )

    artifact = {
        "model": model,
        "confidence_threshold": confidence_threshold,
        "ambiguous_low": args.ambiguous_low,
        "ambiguous_high": args.ambiguous_high,
        "feature_version": "round8_ambiguous_selector_scores_style_v1",
        "positive_label": "llm",
    }
    model_path = output_dir / "selector.pkl"
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    report = {
        "model_path": str(model_path),
        "confidence_threshold": confidence_threshold,
        "threshold_selection": threshold_report,
        "inputs": {
            "train": args.train,
            "dev": args.dev,
            "probe": args.probe,
        },
        "split_summaries": {name: split_summary(rows) for name, rows in rows_by_split.items()},
        "metrics": metrics,
        "deltas": deltas,
        "prediction_outputs": prediction_outputs,
        "decision": decision,
        "config": vars(args),
    }
    write_json(report, output_dir / "selector_report.json")
    write_markdown(report, Path(args.report_md))

    print("=" * 70)
    print("Round8 ambiguous selector trained")
    print("=" * 70)
    print(f"Train rows: {len(rows_by_split['train'])}")
    print(f"Confidence threshold: {confidence_threshold:.4f}")
    for name, block in metrics.items():
        delta = deltas[name]
        print(
            f"{name}: local_f1={block['round8_oneshot_local']['f1']:.4f} "
            f"step7_f1={block['step7']['f1']:.4f} net={delta['net_correct_gain']} "
            f"new_fp={delta['new_false_positives']} used={delta['selector_used_rows']}"
        )
    print(decision)


if __name__ == "__main__":
    main()
