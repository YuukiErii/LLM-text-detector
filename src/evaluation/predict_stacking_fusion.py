import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import load_records
from models.train_stacking_fusion import feature_dict


DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "round2_stacker" / "stacking_model.pkl"


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    parser = argparse.ArgumentParser(description="Predict with a trained round2 stacking fusion model.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.model).open("rb") as f:
        model = pickle.load(f)
    rows = load_records(Path(args.predictions))
    probs = model.predict_proba([feature_dict(row) for row in rows])[:, 1]
    out_rows = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["probability"] = float(prob)
        item["prob_llm"] = float(prob)
        item["prediction"] = int(prob >= args.threshold)
        item["stacker_threshold"] = args.threshold
        out_rows.append(item)

    save_jsonl(out_rows, Path(args.output))
    metrics = evaluate(out_rows)
    if args.metrics and metrics:
        path = Path(args.metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Stacking predictions written")
    print("=" * 70)
    print(f"Rows: {len(out_rows)}")
    print(f"Output: {args.output}")
    if metrics:
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
