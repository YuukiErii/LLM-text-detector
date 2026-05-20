import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def load_json_records(path: Optional[Path]) -> List[Dict]:
    if path is None:
        return []

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse {path}, line {line_id}: {e}") from e
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "samples", "records", "items"]:
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported JSON format: {path}")


def clean_text(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def enrich_predictions(predictions: List[Dict], raw_records: List[Dict]) -> List[Dict]:
    raw_by_id = {str(i): item for i, item in enumerate(raw_records)}
    enriched = []
    for idx, pred in enumerate(predictions):
        row = dict(pred)
        sample_id = str(row.get("id", idx))
        raw = raw_by_id.get(sample_id)
        if raw is not None:
            row.setdefault("text", raw.get("text", ""))
            row.setdefault("label", raw.get("label"))
        row["id"] = sample_id
        enriched.append(row)
    return enriched


def metric_block(rows: List[Dict]) -> Dict:
    labels = np.array([int(row["label"]) for row in rows])
    preds = np.array([int(row["prediction"]) for row in rows])
    probs = np.array([to_float(row.get("probability", row.get("prob_llm"))) for row in rows])

    metrics = {
        "num_samples": len(rows),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def format_metrics(metrics: Dict) -> str:
    lines = [
        f"- Samples: {metrics['num_samples']}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
    ]
    if metrics.get("roc_auc") is not None:
        lines.append(f"- ROC-AUC: {metrics['roc_auc']:.4f}")
    lines.append(f"- Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    return "\n".join(lines)


def branch_summary(rows: List[Dict], threshold: float) -> str:
    if not rows or "p_tfidf" not in rows[0] or "p_deberta" not in rows[0]:
        return "Branch probability columns are not available."

    counter = Counter()
    branch_correct = Counter()
    for row in rows:
        label = int(row["label"])
        tfidf_pred = int(to_float(row.get("p_tfidf")) >= threshold)
        deberta_pred = int(to_float(row.get("p_deberta")) >= threshold)
        final_pred = int(row["prediction"])
        key = f"tfidf={tfidf_pred}, deberta={deberta_pred}, final={final_pred}"
        counter[key] += 1
        if tfidf_pred == label:
            branch_correct["tfidf"] += 1
        if deberta_pred == label:
            branch_correct["deberta"] += 1
        if final_pred == label:
            branch_correct["final"] += 1

    lines = ["Branch agreement patterns:"]
    for key, value in counter.most_common():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Branch accuracy using the final threshold for branch-only decisions:")
    for key in ["tfidf", "deberta", "final"]:
        lines.append(f"- {key}: {branch_correct[key] / len(rows):.4f}")
    return "\n".join(lines)


def error_rows(rows: Iterable[Dict], kind: str, limit: int) -> List[Dict]:
    if kind == "fp":
        selected = [row for row in rows if int(row["label"]) == 0 and int(row["prediction"]) == 1]
        return sorted(selected, key=lambda row: to_float(row.get("probability")), reverse=True)[:limit]
    if kind == "fn":
        selected = [row for row in rows if int(row["label"]) == 1 and int(row["prediction"]) == 0]
        return sorted(selected, key=lambda row: to_float(row.get("probability")))[:limit]
    raise ValueError(f"Unknown error kind: {kind}")


def format_error_table(rows: List[Dict], text_limit: int) -> str:
    if not rows:
        return "No examples."

    lines = ["| id | label | pred | prob | p_tfidf | p_deberta | text |", "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        text = clean_text(row.get("text", ""), text_limit)
        text = text.replace("|", "\\|")
        lines.append(
            "| {id} | {label} | {pred} | {prob:.4f} | {tfidf:.4f} | {deberta:.4f} | {text} |".format(
                id=row.get("id", ""),
                label=int(row["label"]),
                pred=int(row["prediction"]),
                prob=to_float(row.get("probability", row.get("prob_llm"))),
                tfidf=to_float(row.get("p_tfidf")),
                deberta=to_float(row.get("p_deberta")),
                text=text,
            )
        )
    return "\n".join(lines)


def build_report(rows: List[Dict], threshold: float, examples: int, text_limit: int) -> str:
    metrics = metric_block(rows)
    pred_dist = Counter(int(row["prediction"]) for row in rows)
    label_dist = Counter(int(row["label"]) for row in rows)
    fp = [row for row in rows if int(row["label"]) == 0 and int(row["prediction"]) == 1]
    fn = [row for row in rows if int(row["label"]) == 1 and int(row["prediction"]) == 0]

    sections = [
        "# Prediction Error Analysis",
        "",
        "## Overall Metrics",
        "",
        format_metrics(metrics),
        "",
        "## Label And Prediction Distribution",
        "",
        f"- True human / LLM: {label_dist.get(0, 0)} / {label_dist.get(1, 0)}",
        f"- Predicted human / LLM: {pred_dist.get(0, 0)} / {pred_dist.get(1, 0)}",
        f"- False positives: {len(fp)}",
        f"- False negatives: {len(fn)}",
        "",
        "## Branch Diagnostics",
        "",
        branch_summary(rows, threshold),
        "",
        "## Highest-Confidence False Positives",
        "",
        format_error_table(error_rows(rows, "fp", examples), text_limit),
        "",
        "## Lowest-Probability False Negatives",
        "",
        format_error_table(error_rows(rows, "fn", examples), text_limit),
        "",
    ]
    return "\n".join(sections)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze labeled prediction errors.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL from predict_ensemble.py.")
    parser.add_argument("--input", default=None, help="Optional original JSON/JSONL input with text fields.")
    parser.add_argument("--output", required=True, help="Markdown report path.")
    parser.add_argument("--threshold", type=float, default=0.48)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--text_limit", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    prediction_path = Path(args.predictions)
    input_path = Path(args.input) if args.input else None
    output_path = Path(args.output)

    predictions = load_json_records(prediction_path)
    raw_records = load_json_records(input_path) if input_path else []
    rows = enrich_predictions(predictions, raw_records)
    rows = [row for row in rows if row.get("label") in [0, 1] and row.get("prediction") in [0, 1]]
    if not rows:
        raise ValueError("No labeled prediction rows found.")

    report = build_report(rows, args.threshold, args.examples, args.text_limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote analysis report: {output_path}")


if __name__ == "__main__":
    main()
