import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.predict_ensemble import (
    load_records,
    normalize_records,
    predict_deberta,
    predict_tfidf,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "residual_candidate_pool_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "predictions" / "residual_candidate_pool_v1_step7_predictions.jsonl"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "predictions" / "residual_candidate_pool_v1_step7_metrics.json"
DEFAULT_TFIDF_DIR = PROJECT_ROOT / "outputs" / "models" / "tfidf_lit_academic_poetry"
DEFAULT_DEBERTA_DIR = PROJECT_ROOT / "outputs" / "models" / "deberta_lit_academic_poetry_step7_combined"


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def labeled(rows: Sequence[Dict]) -> bool:
    return bool(rows) and all(row.get("label") in [0, 1] for row in rows)


def metrics_for(rows: Sequence[Dict]) -> Dict:
    if not labeled(rows):
        return {}
    y_true = np.array([int(row["label"]) for row in rows], dtype=int)
    y_pred = np.array([int(row["step7_pred"]) for row in rows], dtype=int)
    y_prob = np.array([float(row["p_step7"]) for row in rows], dtype=float)
    metrics = {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
        "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum()),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def grouped_metrics(rows: Sequence[Dict], key: str) -> Dict[str, Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {name: metrics_for(group) for name, group in sorted(grouped.items())}


def probability_bucket(probability: float) -> str:
    if probability < 0.2:
        return "lt_0.20"
    if probability < 0.35:
        return "0.20_0.35"
    if probability < 0.45:
        return "0.35_0.45"
    if probability < 0.55:
        return "0.45_0.55"
    if probability < 0.65:
        return "0.55_0.65"
    if probability < 0.8:
        return "0.65_0.80"
    return "ge_0.80"


def enrich_predictions(
    samples: Sequence[Dict],
    p_tfidf: np.ndarray,
    p_deberta: np.ndarray,
    p_step7: np.ndarray,
    threshold: float,
    ambiguous_low: float,
    ambiguous_high: float,
) -> List[Dict]:
    rows = []
    for sample, tfidf_prob, deberta_prob, step7_prob in zip(samples, p_tfidf, p_deberta, p_step7):
        label = sample.get("label")
        pred = int(step7_prob >= threshold)
        step7_correct = None
        if label in [0, 1]:
            step7_correct = pred == int(label)

        row = dict(sample)
        row["p_tfidf"] = float(tfidf_prob)
        row["p_deberta_step7"] = float(deberta_prob)
        row["p_deberta"] = float(deberta_prob)
        row["p_step7"] = float(step7_prob)
        row["probability"] = float(step7_prob)
        row["prob_llm"] = float(step7_prob)
        row["step7_pred"] = pred
        row["prediction"] = pred
        row["step7_threshold"] = threshold
        row["step7_correct"] = step7_correct
        row["ambiguous_zone"] = bool(ambiguous_low <= step7_prob <= ambiguous_high)
        row["step7_probability_bucket"] = probability_bucket(float(step7_prob))

        if label == 0:
            row["hard_human_candidate"] = bool(step7_prob >= 0.55)
            row["very_hard_human_candidate"] = bool(step7_prob >= 0.65)
            row["hard_llm_candidate"] = False
            row["very_hard_llm_candidate"] = False
        elif label == 1:
            row["hard_human_candidate"] = False
            row["very_hard_human_candidate"] = False
            row["hard_llm_candidate"] = bool(step7_prob <= 0.45)
            row["very_hard_llm_candidate"] = bool(step7_prob <= 0.35)
        else:
            row["hard_human_candidate"] = False
            row["very_hard_human_candidate"] = False
            row["hard_llm_candidate"] = False
            row["very_hard_llm_candidate"] = False

        rows.append(row)
    return rows


def hard_residual_summary(rows: Sequence[Dict]) -> Dict:
    counts = Counter()
    bucket_counts = defaultdict(Counter)
    for row in rows:
        label = row.get("label")
        taxonomy = str(row.get("round8_bucket") or "unknown")
        if row.get("ambiguous_zone"):
            counts["ambiguous_zone"] += 1
            bucket_counts["ambiguous_zone"][taxonomy] += 1
        if label == 0 and row.get("hard_human_candidate"):
            counts["hard_human_candidates"] += 1
            bucket_counts["hard_human_candidates"][taxonomy] += 1
        if label == 0 and row.get("very_hard_human_candidate"):
            counts["very_hard_human_candidates"] += 1
            bucket_counts["very_hard_human_candidates"][taxonomy] += 1
        if label == 1 and row.get("hard_llm_candidate"):
            counts["hard_llm_candidates"] += 1
            bucket_counts["hard_llm_candidates"][taxonomy] += 1
        if label == 1 and row.get("very_hard_llm_candidate"):
            counts["very_hard_llm_candidates"] += 1
            bucket_counts["very_hard_llm_candidates"][taxonomy] += 1
        if row.get("step7_correct") is False:
            counts["step7_errors"] += 1
            bucket_counts["step7_errors"][taxonomy] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "round8_bucket_counts": {name: dict(counter.most_common()) for name, counter in sorted(bucket_counts.items())},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Score residual candidates with the frozen Step7 ensemble.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--tfidf_dir", default=str(DEFAULT_TFIDF_DIR))
    parser.add_argument("--deberta_dir", default=str(DEFAULT_DEBERTA_DIR))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--ambiguous_low", type=float, default=0.35)
    parser.add_argument("--ambiguous_high", type=float, default=0.65)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    metrics_path = Path(args.metrics)
    tfidf_dir = Path(args.tfidf_dir)
    deberta_dir = Path(args.deberta_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input: {input_path}")
    if not tfidf_dir.exists():
        raise FileNotFoundError(f"Cannot find TF-IDF dir: {tfidf_dir}")
    if not deberta_dir.exists():
        raise FileNotFoundError(f"Cannot find DeBERTa dir: {deberta_dir}")

    samples = normalize_records(load_records(input_path))
    if not samples:
        raise ValueError(f"No valid samples found in {input_path}")

    print("=" * 70)
    print("Score Round8 residual candidates with frozen Step7")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Samples: {len(samples)}")
    print(f"TF-IDF: {tfidf_dir}")
    print(f"DeBERTa: {deberta_dir}")
    print(f"alpha={args.alpha} threshold={args.threshold}")

    print("\nPredicting TF-IDF branch...")
    p_tfidf = predict_tfidf(samples, tfidf_dir)

    print("Predicting Step7 DeBERTa branch...")
    p_deberta = predict_deberta(samples, deberta_dir, batch_size=args.batch_size, max_length=args.max_length)

    p_step7 = args.alpha * p_deberta + (1.0 - args.alpha) * p_tfidf
    rows = enrich_predictions(
        samples,
        p_tfidf=p_tfidf,
        p_deberta=p_deberta,
        p_step7=p_step7,
        threshold=args.threshold,
        ambiguous_low=args.ambiguous_low,
        ambiguous_high=args.ambiguous_high,
    )

    save_jsonl(rows, output_path)
    metrics = {
        "input": str(input_path),
        "output": str(output_path),
        "tfidf_dir": str(tfidf_dir),
        "deberta_dir": str(deberta_dir),
        "alpha": args.alpha,
        "threshold": args.threshold,
        "ambiguous_zone": {
            "low": args.ambiguous_low,
            "high": args.ambiguous_high,
        },
        "overall": metrics_for(rows),
        "by_round8_bucket": grouped_metrics(rows, "round8_bucket"),
        "by_round8_family": grouped_metrics(rows, "round8_bucket_family"),
        "by_domain": grouped_metrics(rows, "domain"),
        "by_generator": grouped_metrics(rows, "generator"),
        "hard_residual_summary": hard_residual_summary(rows),
    }
    write_json(metrics, metrics_path)

    print("\nSaved:")
    print(f"  predictions: {output_path}")
    print(f"  metrics:     {metrics_path}")
    if metrics["overall"]:
        overall = metrics["overall"]
        print("\nOverall:")
        print(f"  accuracy:  {overall['accuracy']:.4f}")
        print(f"  precision: {overall['precision']:.4f}")
        print(f"  recall:    {overall['recall']:.4f}")
        print(f"  f1:        {overall['f1']:.4f}")
        print(f"  confusion: {overall['confusion_matrix']}")
    print(f"Hard residual summary: {metrics['hard_residual_summary']['counts']}")


if __name__ == "__main__":
    main()
