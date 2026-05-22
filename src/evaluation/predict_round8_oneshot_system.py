import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import load_records  # noqa: E402
from models.train_round8_ambiguous_selector import feature_dict  # noqa: E402


DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "round8_ambiguous_selector" / "selector.pkl"


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
    return str(value) if value not in [None, ""] else f"row_{index:06d}"


def load_rows(path: Path) -> List[Dict]:
    rows = []
    for index, row in enumerate(load_records(path)):
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item["id"] = row_id(item, index)
        rows.append(item)
    return rows


def probability(row: Dict) -> float:
    for key in ["p_residual_deberta", "p_stylometry", "p_step7", "probability", "prob_llm"]:
        value = row.get(key)
        if value not in [None, ""]:
            return float(value)
    return float(row.get("prediction", 0))


def load_by_id(path_value: str) -> Dict[str, Dict]:
    if not path_value:
        return {}
    return {row_id(row, index): row for index, row in enumerate(load_rows(Path(path_value)))}


def merge_branch_predictions(rows: Sequence[Dict], residual_path: str, stylometry_path: str) -> List[Dict]:
    residual_by_id = load_by_id(residual_path)
    stylometry_by_id = load_by_id(stylometry_path)
    merged = []
    for index, row in enumerate(rows):
        item = dict(row)
        rid = row_id(item, index)
        step7_prob = float(item.get("p_step7", item.get("probability", item.get("prob_llm", 0.0))))
        item["p_step7"] = step7_prob
        item["step7_pred"] = int(item.get("step7_pred", item.get("prediction", step7_prob >= 0.55)))
        if item.get("p_deberta_step7") in [None, ""] and item.get("p_deberta") not in [None, ""]:
            item["p_deberta_step7"] = float(item["p_deberta"])

        residual = residual_by_id.get(rid)
        if residual:
            item["p_residual_deberta"] = float(residual.get("p_residual_deberta", probability(residual)))
            item["residual_deberta_pred"] = int(residual.get("residual_deberta_pred", residual.get("prediction", item["p_residual_deberta"] >= 0.5)))

        stylometry = stylometry_by_id.get(rid)
        if stylometry:
            item["p_stylometry"] = float(stylometry.get("p_stylometry", probability(stylometry)))
            item["stylometry_pred"] = int(stylometry.get("stylometry_pred", stylometry.get("prediction", item["p_stylometry"] >= 0.5)))

        if "p_residual_deberta" not in item:
            raise ValueError(f"Missing residual prediction for id={rid}")
        if "p_stylometry" not in item:
            raise ValueError(f"Missing stylometry prediction for id={rid}")
        merged.append(item)
    return merged


def metrics_for(rows: Sequence[Dict], pred_key: str, prob_key: str) -> Dict:
    if not rows or any(row.get("label") not in [0, 1] for row in rows):
        return {}
    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    preds = np.array([int(row[pred_key]) for row in rows], dtype=int)
    probs = np.array([float(row[prob_key]) for row in rows], dtype=float)
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    out = {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "false_positives": fp,
        "false_negatives": fn,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["roc_auc"] = None
    return out


def apply_system(rows: Sequence[Dict], artifact: Dict, confidence_override: float = None) -> List[Dict]:
    model = artifact["model"]
    confidence_threshold = float(confidence_override if confidence_override is not None else artifact["confidence_threshold"])
    low = float(artifact.get("ambiguous_low", 0.35))
    high = float(artifact.get("ambiguous_high", 0.65))
    selector_probs = model.predict_proba([feature_dict(row) for row in rows])[:, 1]
    out = []
    for row, selector_prob in zip(rows, selector_probs):
        item = dict(row)
        p_step7 = float(item["p_step7"])
        step7_pred = int(item["step7_pred"])
        selector_pred = int(selector_prob >= 0.5)
        selector_confidence = max(float(selector_prob), 1.0 - float(selector_prob))
        in_ambiguous_zone = low <= p_step7 <= high
        use_selector = in_ambiguous_zone and selector_confidence >= confidence_threshold
        final_pred = selector_pred if use_selector else step7_pred
        final_prob = float(selector_prob) if use_selector else p_step7
        item["p_round8_ambiguous_selector"] = float(selector_prob)
        item["round8_ambiguous_selector_pred"] = selector_pred
        item["round8_ambiguous_selector_confidence"] = float(selector_confidence)
        item["round8_ambiguous_selector_confidence_threshold"] = confidence_threshold
        item["round8_ambiguous_selector_in_zone"] = bool(in_ambiguous_zone)
        item["round8_ambiguous_selector_used"] = bool(use_selector)
        item["prediction"] = int(final_pred)
        item["probability"] = float(final_prob)
        item["prob_llm"] = float(final_prob)
        item["round8_oneshot_prediction"] = int(final_pred)
        item["round8_oneshot_probability"] = float(final_prob)
        out.append(item)
    return out


def delta_vs_step7(rows: Sequence[Dict]) -> Dict:
    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    step7 = np.array([int(row["step7_pred"]) for row in rows], dtype=int)
    final = np.array([int(row["round8_oneshot_prediction"]) for row in rows], dtype=int)
    used = np.array([bool(row["round8_ambiguous_selector_used"]) for row in rows], dtype=bool)
    selector = np.array([int(row["round8_ambiguous_selector_pred"]) for row in rows], dtype=int)
    return {
        "step7_correct": int((step7 == labels).sum()),
        "final_correct": int((final == labels).sum()),
        "net_correct_gain": int((final == labels).sum() - (step7 == labels).sum()),
        "fixed_step7_errors": int(((step7 != labels) & (final == labels)).sum()),
        "broken_step7_correct": int(((step7 == labels) & (final != labels)).sum()),
        "fixed_false_negatives": int(((labels == 1) & (step7 == 0) & (final == 1)).sum()),
        "fixed_false_positives": int(((labels == 0) & (step7 == 1) & (final == 0)).sum()),
        "new_false_positives": int(((labels == 0) & (step7 == 0) & (final == 1)).sum()),
        "new_false_negatives": int(((labels == 1) & (step7 == 1) & (final == 0)).sum()),
        "selector_eligible_rows": int(sum(bool(row["round8_ambiguous_selector_in_zone"]) for row in rows)),
        "selector_used_rows": int(used.sum()),
        "selector_changed_rows": int((used & (selector != step7)).sum()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Apply the Round8 one-shot local ambiguous selector over Step7.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--residual_predictions", required=True)
    parser.add_argument("--stylometry_predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--selector_model", default=str(DEFAULT_MODEL))
    parser.add_argument("--confidence_threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(Path(args.input))
    merged = merge_branch_predictions(rows, args.residual_predictions, args.stylometry_predictions)
    with Path(args.selector_model).open("rb") as f:
        artifact = pickle.load(f)
    output_rows = apply_system(merged, artifact, confidence_override=args.confidence_threshold)
    save_jsonl(output_rows, Path(args.output))

    metrics = {
        "input": args.input,
        "selector_model": args.selector_model,
        "output": args.output,
        "step7": metrics_for(output_rows, "step7_pred", "p_step7"),
        "round8_oneshot": metrics_for(output_rows, "round8_oneshot_prediction", "round8_oneshot_probability"),
        "delta_vs_step7": delta_vs_step7(output_rows) if output_rows and output_rows[0].get("label") in [0, 1] else {},
    }
    write_json(metrics, Path(args.metrics))

    print("=" * 70)
    print("Round8 one-shot predictions written")
    print("=" * 70)
    print(f"Rows: {len(output_rows)}")
    print(f"Output: {args.output}")
    if metrics["round8_oneshot"]:
        print(f"Step7 F1: {metrics['step7']['f1']:.4f}")
        print(f"Round8 F1: {metrics['round8_oneshot']['f1']:.4f}")
        print(f"Delta: {metrics['delta_vs_step7']}")


if __name__ == "__main__":
    main()
