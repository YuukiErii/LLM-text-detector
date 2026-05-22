import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from evaluation.assign_text_bucket import load_records
from train_round4_human_style_guard import feature_dict


DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "round4_human_style_guard"


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_artifact(model_dir: Path) -> Dict:
    path = model_dir / "human_style_guard.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find guard artifact: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def guard_labels(rows: List[Dict]) -> Optional[np.ndarray]:
    labels = []
    for row in rows:
        label = row.get("label")
        if label not in [0, 1]:
            return None
        labels.append(1 if int(label) == 0 else 0)
    return np.array(labels, dtype=int)


def metrics_for(rows: List[Dict], probs: np.ndarray, threshold: float) -> Optional[Dict]:
    labels = guard_labels(rows)
    if labels is None:
        return None
    preds = (probs >= threshold).astype(int)
    metrics = {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_human_style": float(precision_score(labels, preds, zero_division=0)),
        "recall_human_style": float(recall_score(labels, preds, zero_division=0)),
        "f1_human_style": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "veto_rate": float(np.mean(preds == 1)) if len(preds) else 0.0,
        "mean_p_human_style": float(np.mean(probs)) if len(probs) else 0.0,
        "threshold": threshold,
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Round4 human-style guard probabilities.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--include_text", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_records(Path(args.input))
    rows = [row for row in rows if isinstance(row.get("text"), str) and row.get("text").strip()]
    if not rows:
        raise ValueError(f"No text rows found in {args.input}")

    artifact = load_artifact(Path(args.model_dir))
    model = artifact["model"]
    char_vectorizer = artifact["char_vectorizer"]
    dict_vectorizer = artifact["dict_vectorizer"]
    threshold = float(args.threshold if args.threshold is not None else artifact.get("threshold", 0.75))

    x_char = char_vectorizer.transform([str(row.get("text", "")) for row in rows])
    x_dict = dict_vectorizer.transform([feature_dict(row) for row in rows])
    x = hstack([x_char, x_dict])
    probs = model.predict_proba(x)[:, 1]

    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["p_human_style"] = float(prob)
        item["human_style_veto"] = int(prob >= threshold)
        item["human_style_guard_threshold"] = threshold
        if not args.include_text:
            item.pop("text", None)
        out.append(item)
    save_jsonl(out, Path(args.output))

    metrics = metrics_for(rows, probs, threshold)
    if args.metrics and metrics is not None:
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Round4 human-style guard predictions written")
    print("=" * 70)
    print(f"Rows: {len(out)}")
    print(f"Output: {args.output}")
    if metrics is not None:
        print(f"Veto rate: {metrics['veto_rate']:.4f}")
        print(f"Human-style recall: {metrics['recall_human_style']:.4f}")
        print(f"Confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
