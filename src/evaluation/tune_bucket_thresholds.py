import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records


DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "models" / "round2_bucket_thresholds.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round2_bucket_router_report.md"


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_row(row: Dict, source_name: str, index: int, default_alpha: float) -> Dict:
    text = str(row.get("text", ""))
    p_tfidf = row.get("p_tfidf")
    p_deberta = row.get("p_deberta")
    if p_tfidf is not None and p_deberta is not None:
        probability = default_alpha * to_float(p_deberta) + (1.0 - default_alpha) * to_float(p_tfidf)
    else:
        probability = to_float(row.get("probability", row.get("prob_llm", row.get("score"))))
    bucket = row.get("bucket") or row.get("rough_domain") or (assign_bucket(text) if text else row.get("domain", "general_prose"))
    return {
        "id": str(row.get("id", index)),
        "label": int(row.get("label")),
        "probability": float(probability),
        "bucket": str(bucket),
        "domain": row.get("domain", "unknown"),
        "generator": row.get("generator", "unknown"),
        "source_name": source_name,
    }


def load_prediction_rows(paths: Sequence[str], default_alpha: float) -> List[Dict]:
    rows = []
    for value in paths:
        path = Path(value)
        source_name = path.stem
        for index, row in enumerate(load_records(path)):
            if row.get("label") not in [0, 1]:
                continue
            rows.append(normalize_row(row, source_name=source_name, index=index, default_alpha=default_alpha))
    return rows


def threshold_grid(probs: Sequence[float]) -> np.ndarray:
    unique = sorted(set(float(prob) for prob in probs))
    candidates = {0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 0.95}
    for prob in unique:
        candidates.add(prob)
    for left, right in zip(unique, unique[1:]):
        candidates.add((left + right) / 2.0)
    return np.array(sorted(value for value in candidates if 0.0 <= value <= 1.0))


def metrics_for(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    preds = (probs >= threshold).astype(int)
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    return {
        "threshold": float(threshold),
        "n": int(len(labels)),
        "accuracy": 0.0 if len(labels) == 0 else float((tp + tn) / len(labels)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def choose_threshold(rows: Sequence[Dict], default_threshold: float, min_precision: float, min_recall: float) -> Dict:
    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    probs = np.array([float(row["probability"]) for row in rows], dtype=float)
    baseline = metrics_for(labels, probs, default_threshold)
    best = None
    best_any = None
    for threshold in threshold_grid(probs):
        block = metrics_for(labels, probs, float(threshold))
        rank = (block["f1"], block["accuracy"], -abs(float(threshold) - default_threshold))
        if best_any is None or rank > best_any["_rank"]:
            best_any = dict(block)
            best_any["_rank"] = rank
        if block["precision"] < min_precision or block["recall"] < min_recall:
            continue
        if best is None or rank > best["_rank"]:
            best = dict(block)
            best["_rank"] = rank
    selected = best or best_any or dict(baseline)
    selected.pop("_rank", None)
    return {
        "baseline": baseline,
        "selected": selected,
        "met_constraints": best is not None,
    }


def apply_thresholds(rows: Sequence[Dict], thresholds: Dict[str, float], fallback: float) -> Dict:
    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    probs = np.array([float(row["probability"]) for row in rows], dtype=float)
    preds = np.array([int(row["probability"] >= thresholds.get(row["bucket"], fallback)) for row in rows], dtype=int)
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    return {
        "n": int(len(rows)),
        "accuracy": 0.0 if len(rows) == 0 else float((tp + tn) / len(rows)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round2 Bucket Router Report",
        "",
        "Thresholds are tuned on validation/round2-dev style data only. Teacher-test labels are not used here.",
        "",
        "## Overall",
        "",
        "| Metric | Baseline global | Bucket routed |",
        "| --- | ---: | ---: |",
    ]
    for key in ["accuracy", "precision", "recall", "f1"]:
        lines.append(f"| {key} | {fmt(report['global_baseline'][key])} | {fmt(report['routed_train_eval'][key])} |")
    lines.append(f"| false_positives | {report['global_baseline']['false_positives']} | {report['routed_train_eval']['false_positives']} |")
    lines.append(f"| false_negatives | {report['global_baseline']['false_negatives']} | {report['routed_train_eval']['false_negatives']} |")
    lines.extend(["", "## Bucket Thresholds", ""])
    lines.extend([
        "| Bucket | n | baseline F1 | selected threshold | selected F1 | precision | recall | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for bucket, block in sorted(report["buckets"].items()):
        selected = block["selected"]
        baseline = block["baseline"]
        lines.append(
            f"| {bucket} | {selected['n']} | {fmt(baseline['f1'])} | {selected['threshold']:.6f} | "
            f"{fmt(selected['f1'])} | {fmt(selected['precision'])} | {fmt(selected['recall'])} | "
            f"{selected['false_positives']} | {selected['false_negatives']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune transparent per-bucket thresholds.")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output_md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--default_threshold", type=float, default=0.55)
    parser.add_argument("--default_alpha", type=float, default=0.5)
    parser.add_argument("--min_precision", type=float, default=0.0)
    parser.add_argument("--min_recall", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_prediction_rows(args.predictions, default_alpha=args.default_alpha)
    if not rows:
        raise ValueError("No labeled prediction rows found.")

    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    bucket_blocks = {
        bucket: choose_threshold(
            bucket_rows,
            default_threshold=args.default_threshold,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        for bucket, bucket_rows in by_bucket.items()
    }
    thresholds = {
        bucket: block["selected"]["threshold"]
        for bucket, block in bucket_blocks.items()
    }
    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    probs = np.array([float(row["probability"]) for row in rows], dtype=float)
    report = {
        "prediction_inputs": args.predictions,
        "default_threshold": args.default_threshold,
        "default_alpha": args.default_alpha,
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
        "thresholds": thresholds,
        "fallback_threshold": args.default_threshold,
        "bucket_distribution": dict(Counter(row["bucket"] for row in rows)),
        "global_baseline": metrics_for(labels, probs, args.default_threshold),
        "routed_train_eval": apply_thresholds(rows, thresholds, fallback=args.default_threshold),
        "buckets": bucket_blocks,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.output_md))

    print("=" * 70)
    print("Round2 bucket thresholds tuned")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Buckets: {len(bucket_blocks)}")
    print(f"Baseline F1: {report['global_baseline']['f1']:.4f}")
    print(f"Routed F1: {report['routed_train_eval']['f1']:.4f}")
    print(f"JSON: {output_json}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
