import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text").strip():
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


def predict(samples: List[Dict], model_dir: Path, batch_size: int, max_length: int) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    best_model = model_dir / "best_model"
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer_path = tokenizer_dir if tokenizer_dir.exists() else best_model

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(best_model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = [sample["text"] for sample in samples]
    probs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1]
            probs.extend(batch_probs.detach().cpu().numpy().tolist())
    return np.array(probs, dtype=float)


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
    parser = argparse.ArgumentParser(description="Run standalone neural classifier inference.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--include_text", action="store_true")
    parser.add_argument("--prob_field", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_records(Path(args.input))
    probs = predict(samples, Path(args.model_dir), batch_size=args.batch_size, max_length=args.max_length)
    rows = []
    for index, (sample, prob) in enumerate(zip(samples, probs)):
        pred = int(prob >= args.threshold)
        row = {
            "id": str(sample.get("id", index)),
            "label": sample.get("label"),
            "prediction": pred,
            "probability": float(prob),
            "prob_llm": float(prob),
        }
        if args.prob_field:
            row[args.prob_field] = float(prob)
        for key in ["domain", "generator", "source", "pair_id", "bucket", "round2_tag"]:
            if sample.get(key) is not None:
                row[key] = sample.get(key)
        if args.include_text:
            row["text"] = sample["text"]
        rows.append(row)

    save_jsonl(rows, Path(args.output))
    metrics = evaluate(rows)
    if args.metrics and metrics:
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Neural predictions written")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.output}")
    if metrics:
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
