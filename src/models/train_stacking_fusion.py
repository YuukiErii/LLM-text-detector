import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round2_stacker"


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(paths: Sequence[str]) -> List[Dict]:
    rows = []
    for value in paths:
        path = Path(value)
        split_name = path.stem
        for index, row in enumerate(load_records(path)):
            if row.get("label") not in [0, 1]:
                continue
            item = dict(row)
            item["_source_file"] = str(path)
            item["_row_index"] = index
            item.setdefault("split_name", split_name)
            rows.append(item)
    return rows


def feature_dict(row: Dict) -> Dict:
    text = str(row.get("text", ""))
    p_tfidf = to_float(row.get("p_tfidf", row.get("prob_tfidf")))
    p_deberta = to_float(row.get("p_deberta", row.get("prob_deberta")))
    p_ensemble = to_float(row.get("probability", row.get("prob_llm", row.get("score"))))
    bucket = str(row.get("bucket") or row.get("rough_domain") or (assign_bucket(text) if text else row.get("domain", "general_prose")))

    features = {
        "p_tfidf": p_tfidf,
        "p_deberta_step7": p_deberta,
        "p_roberta": to_float(row.get("p_roberta")),
        "p_ensemble_step7": p_ensemble,
        "abs_tfidf_deberta": abs(p_tfidf - p_deberta),
        "abs_deberta_roberta": abs(p_deberta - to_float(row.get("p_roberta"))),
        "abs_tfidf_roberta": abs(p_tfidf - to_float(row.get("p_roberta"))),
        "max_base_prob": max(p_tfidf, p_deberta),
        "min_base_prob": min(p_tfidf, p_deberta),
        "max_three_prob": max(p_tfidf, p_deberta, to_float(row.get("p_roberta"))),
        "min_three_prob": min(p_tfidf, p_deberta, to_float(row.get("p_roberta"))),
        "bucket": bucket,
    }
    if text:
        text_block = text_features(text)
        text_block.pop("bucket", None)
        features.update(text_block)
    else:
        for key in [
            "length_chars",
            "length_words",
            "num_lines",
            "linebreak_ratio",
            "avg_line_length",
            "punctuation_ratio",
            "quote_count",
            "dash_count",
            "semicolon_count",
            "archaic_word_count",
            "academic_marker_count",
            "poetic_marker_count",
            "type_token_ratio",
            "sentence_length_mean",
            "sentence_length_std",
        ]:
            features[key] = to_float(row.get(key))
    return features


def labels_for(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([int(row["label"]) for row in rows], dtype=int)


def metrics_for(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> Dict:
    labels = labels_for(rows)
    preds = (probs >= threshold).astype(int)
    metrics = {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def prediction_rows(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> List[Dict]:
    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["probability"] = float(prob)
        item["prob_llm"] = float(prob)
        item["prediction"] = int(prob >= threshold)
        item["stacker_threshold"] = threshold
        out.append(item)
    return out


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round2 Stacking Fusion Report",
        "",
        "This is a lightweight Logistic Regression stacker trained on validation and round2 teacher-like development predictions.",
        "",
        "| Split | n | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, block in report["metrics"].items():
        auc = "NA" if block.get("roc_auc") is None else f"{block['roc_auc']:.4f}"
        lines.append(
            f"| {name} | {block['num_samples']} | {block['accuracy']:.4f} | {block['precision']:.4f} | "
            f"{block['recall']:.4f} | {block['f1']:.4f} | {auc} | {block['confusion_matrix']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a round2 nonlinear stacking fusion model.")
    parser.add_argument("--train_predictions", nargs="+", required=True)
    parser.add_argument("--eval_predictions", nargs="*", default=[])
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--C", type=float, default=0.35)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(args.train_predictions)
    if not train_rows:
        raise ValueError("No labeled train prediction rows found.")
    x_train = [feature_dict(row) for row in train_rows]
    y_train = labels_for(train_rows)

    model = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                LogisticRegression(
                    C=args.C,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    model_path = output_dir / "stacking_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    metrics = {}
    train_probs = model.predict_proba(x_train)[:, 1]
    metrics["train_meta"] = metrics_for(train_rows, train_probs, threshold=args.threshold)
    save_jsonl(prediction_rows(train_rows, train_probs, threshold=args.threshold), output_dir / "train_meta_predictions.jsonl")

    for value in args.eval_predictions:
        rows = load_rows([value])
        if not rows:
            continue
        probs = model.predict_proba([feature_dict(row) for row in rows])[:, 1]
        name = Path(value).stem
        metrics[name] = metrics_for(rows, probs, threshold=args.threshold)
        save_jsonl(prediction_rows(rows, probs, threshold=args.threshold), output_dir / f"{name}_stacker_predictions.jsonl")

    report = {
        "model_path": str(model_path),
        "train_predictions": args.train_predictions,
        "eval_predictions": args.eval_predictions,
        "C": args.C,
        "threshold": args.threshold,
        "metrics": metrics,
    }
    (output_dir / "stacker_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, PROJECT_ROOT / "outputs" / "evaluation" / "round2_stacker_report.md")

    print("=" * 70)
    print("Round2 stacker trained")
    print("=" * 70)
    print(f"Train rows: {len(train_rows)}")
    print(f"Model: {model_path}")
    for name, block in metrics.items():
        print(f"{name}: f1={block['f1']:.4f} acc={block['accuracy']:.4f} confusion={block['confusion_matrix']}")


if __name__ == "__main__":
    main()
