import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records


DEFAULT_THRESHOLDS = PROJECT_ROOT / "outputs" / "models" / "round2_bucket_thresholds.json"


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_thresholds(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "thresholds": {str(key): float(value) for key, value in data.get("thresholds", {}).items()},
        "fallback_threshold": float(data.get("fallback_threshold", data.get("default_threshold", 0.55))),
        "source": str(path),
    }


def normalize_row(row: Dict, index: int, alpha: float) -> Dict:
    text = str(row.get("text", ""))
    if row.get("p_tfidf") is not None and row.get("p_deberta") is not None:
        probability = alpha * to_float(row.get("p_deberta")) + (1.0 - alpha) * to_float(row.get("p_tfidf"))
    else:
        probability = to_float(row.get("probability", row.get("prob_llm", row.get("score"))))
    bucket = row.get("bucket") or row.get("rough_domain") or (assign_bucket(text) if text else row.get("domain", "general_prose"))
    item = dict(row)
    item["id"] = str(item.get("id", index))
    item["probability"] = float(probability)
    item["prob_llm"] = float(probability)
    item["bucket"] = str(bucket)
    return item


def evaluate(rows: List[Dict]) -> Dict:
    if not rows or any(row.get("label") not in [0, 1] for row in rows):
        return {}
    y_true = np.array([int(row["label"]) for row in rows], dtype=int)
    y_pred = np.array([int(row["prediction"]) for row in rows], dtype=int)
    y_prob = np.array([float(row["probability"]) for row in rows], dtype=float)
    metrics = {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Apply round2 bucket-routed ensemble thresholds to prediction rows.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_thresholds(Path(args.thresholds))
    rows = []
    for index, row in enumerate(load_records(Path(args.predictions))):
        item = normalize_row(row, index=index, alpha=args.alpha)
        threshold = config["thresholds"].get(item["bucket"], config["fallback_threshold"])
        item["bucket_threshold"] = threshold
        item["prediction"] = int(item["probability"] >= threshold)
        rows.append(item)

    save_jsonl(rows, Path(args.output))
    metrics = evaluate(rows)
    if args.metrics and metrics:
        metrics.update(
            {
                "prediction_input": args.predictions,
                "threshold_config": config["source"],
                "output": args.output,
            }
        )
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Bucket-routed predictions written")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.output}")
    if metrics:
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
