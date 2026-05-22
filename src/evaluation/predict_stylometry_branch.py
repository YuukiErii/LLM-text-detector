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

from models.train_stylometry_branch import predict_probs


DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "stylometry_round8" / "stylometry_branch.pkl"


def load_records(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
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
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    rows.append(item)
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict) and isinstance(row.get("text"), str)]
    if isinstance(data, dict):
        for key in ["data", "samples", "records", "items"]:
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict) and isinstance(row.get("text"), str)]
    raise ValueError(f"Unsupported input format: {path}")


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(rows: List[Dict], threshold: float) -> Dict:
    if not rows or any(row.get("label") not in [0, 1] for row in rows):
        return {}
    y_true = np.array([int(row["label"]) for row in rows], dtype=int)
    probs = np.array([float(row["p_stylometry"]) for row in rows], dtype=float)
    preds = (probs >= threshold).astype(int)
    metrics = {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run Round8 stylometry branch inference.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--metrics", default="")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    with model_path.open("rb") as f:
        artifacts = pickle.load(f)
    threshold = float(args.threshold if args.threshold is not None else artifacts.get("threshold", 0.5))

    rows = load_records(Path(args.input))
    probs = predict_probs(artifacts, [row["text"] for row in rows])
    out_rows = []
    for index, (row, prob) in enumerate(zip(rows, probs)):
        pred = int(prob >= threshold)
        item = dict(row)
        item["p_stylometry"] = float(prob)
        item["stylometry_pred"] = pred
        item["stylometry_threshold"] = threshold
        out_rows.append(item)

    save_jsonl(out_rows, Path(args.output))
    metrics = evaluate(out_rows, threshold)
    if args.metrics and metrics:
        write_json(metrics, Path(args.metrics))

    print("=" * 70)
    print("Stylometry predictions written")
    print("=" * 70)
    print(f"Rows: {len(out_rows)}")
    print(f"Output: {args.output}")
    if metrics:
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
