import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.predict_neural_model import predict as predict_neural_probs  # noqa: E402
from models.train_stylometry_branch import predict_probs as predict_stylometry_probs  # noqa: E402


DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "round8_ambiguous_selector_train.jsonl"
DEFAULT_DEV = PROJECT_ROOT / "data" / "processed" / "round8_ambiguous_selector_dev.jsonl"
DEFAULT_PROBE = PROJECT_ROOT / "data" / "processed" / "round8_ambiguous_selector_probe.jsonl"
DEFAULT_RESIDUAL_MODEL = PROJECT_ROOT / "outputs" / "models" / "deberta_round8_residual_mix"
DEFAULT_STYLOMETRY_MODEL = PROJECT_ROOT / "outputs" / "models" / "stylometry_round8" / "stylometry_branch.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "predictions" / "round8_ambiguous_selector_score_report.json"


def load_jsonl(path: Path) -> List[Dict]:
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


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prediction_metrics(rows: Sequence[Dict], pred_key: str, prob_key: str) -> Dict:
    if not rows or any(row.get("label") not in [0, 1] for row in rows):
        return {}
    y_true = np.array([int(row["label"]) for row in rows], dtype=int)
    y_pred = np.array([int(row[pred_key]) for row in rows], dtype=int)
    y_prob = np.array([float(row[prob_key]) for row in rows], dtype=float)
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


def split_summary(rows: Sequence[Dict]) -> Dict:
    return {
        "num_rows": len(rows),
        "label_distribution": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "round8_bucket_distribution": dict(sorted(Counter(str(row.get("round8_bucket")) for row in rows).items())),
        "domain_distribution": dict(sorted(Counter(str(row.get("domain")) for row in rows).items())),
        "generator_distribution": dict(sorted(Counter(str(row.get("generator")) for row in rows).items())),
    }


def enrich_rows(rows: Sequence[Dict], residual_probs: Sequence[float], stylometry_probs: Sequence[float], args) -> List[Dict]:
    out = []
    for row, residual_prob, stylometry_prob in zip(rows, residual_probs, stylometry_probs):
        item = dict(row)
        step7_prob = float(item.get("p_step7", item.get("probability", item.get("prob_llm", 0.0))))
        item["p_step7"] = step7_prob
        item["step7_pred"] = int(item.get("step7_pred", item.get("prediction", step7_prob >= args.step7_threshold)))
        item["p_residual_deberta"] = float(residual_prob)
        item["residual_deberta_pred"] = int(residual_prob >= args.residual_threshold)
        item["residual_deberta_threshold"] = float(args.residual_threshold)
        item["p_stylometry"] = float(stylometry_prob)
        item["stylometry_pred"] = int(stylometry_prob >= args.stylometry_threshold)
        item["stylometry_threshold"] = float(args.stylometry_threshold)
        item["branch_disagreement_count"] = len(
            {
                int(item["step7_pred"]),
                int(item["residual_deberta_pred"]),
                int(item["stylometry_pred"]),
            }
        ) - 1
        out.append(item)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Score Round8 ambiguous selector splits with residual and stylometry branches.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev", default=str(DEFAULT_DEV))
    parser.add_argument("--probe", default=str(DEFAULT_PROBE))
    parser.add_argument("--residual_model_dir", default=str(DEFAULT_RESIDUAL_MODEL))
    parser.add_argument("--stylometry_model", default=str(DEFAULT_STYLOMETRY_MODEL))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--step7_threshold", type=float, default=0.55)
    parser.add_argument("--residual_threshold", type=float, default=0.5)
    parser.add_argument("--stylometry_threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    split_paths = {
        "train": Path(args.train),
        "dev": Path(args.dev),
        "probe": Path(args.probe),
    }
    rows_by_split = {name: load_jsonl(path) for name, path in split_paths.items()}
    combined = []
    spans = {}
    for name, rows in rows_by_split.items():
        start = len(combined)
        combined.extend(rows)
        spans[name] = (start, len(combined))
    if not combined:
        raise ValueError("No rows found in ambiguous selector splits.")

    print("=" * 70)
    print("Scoring Round8 ambiguous selector splits")
    print("=" * 70)
    print(f"Rows: {len(combined)}")
    print(f"Residual model: {args.residual_model_dir}")
    print(f"Stylometry model: {args.stylometry_model}")

    residual_probs = predict_neural_probs(
        combined,
        Path(args.residual_model_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    with Path(args.stylometry_model).open("rb") as f:
        stylometry_artifact = pickle.load(f)
    stylometry_probs = predict_stylometry_probs(stylometry_artifact, [row["text"] for row in combined])

    output_dir = Path(args.output_dir)
    report = {
        "inputs": {name: str(path) for name, path in split_paths.items()},
        "models": {
            "residual_deberta": str(Path(args.residual_model_dir)),
            "stylometry": str(Path(args.stylometry_model)),
        },
        "outputs": {},
        "split_summaries": {},
        "metrics": {},
    }
    for name, (start, end) in spans.items():
        scored = enrich_rows(
            rows_by_split[name],
            residual_probs[start:end],
            stylometry_probs[start:end],
            args,
        )
        output_path = output_dir / f"round8_ambiguous_selector_{name}_scored.jsonl"
        save_jsonl(scored, output_path)
        report["outputs"][name] = str(output_path)
        report["split_summaries"][name] = split_summary(scored)
        report["metrics"][name] = {
            "step7": prediction_metrics(scored, "step7_pred", "p_step7"),
            "residual_deberta": prediction_metrics(scored, "residual_deberta_pred", "p_residual_deberta"),
            "stylometry": prediction_metrics(scored, "stylometry_pred", "p_stylometry"),
        }
        print(f"{name}: {len(scored)} -> {output_path}")

    write_json(report, Path(args.report))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
