import argparse
import json
import pickle
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import text_features


DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_round8_residual_mix.jsonl"
DEFAULT_VALID = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_valid.jsonl"
DEFAULT_INTERNAL_TEST = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_internal_test.jsonl"
DEFAULT_RESIDUAL_DEV = PROJECT_ROOT / "data" / "processed" / "residual_dev_v1.jsonl"
DEFAULT_RESIDUAL_PROBE = PROJECT_ROOT / "data" / "processed" / "residual_probe_v1.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "stylometry_round8"


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "if",
    "in", "is", "it", "its", "not", "of", "on", "or", "she", "that",
    "the", "their", "there", "they", "this", "to", "was", "we", "were",
    "which", "who", "with", "you",
}


STYLE_FEATURE_NAMES = [
    "length_chars",
    "length_words",
    "num_lines",
    "linebreak_ratio",
    "avg_line_length",
    "punctuation_ratio",
    "quote_count",
    "dash_count",
    "semicolon_count",
    "colon_count",
    "comma_count",
    "period_count",
    "question_count",
    "exclamation_count",
    "parenthesis_count",
    "digit_ratio",
    "uppercase_ratio",
    "avg_word_length",
    "word_length_std",
    "type_token_ratio",
    "stopword_ratio",
    "function_word_ratio",
    "sentence_length_mean",
    "sentence_length_std",
    "archaic_word_count",
    "academic_marker_count",
    "poetic_marker_count",
]


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
            if isinstance(item, dict) and item.get("label") in [0, 1] and isinstance(item.get("text"), str):
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


def word_list(text: str) -> List[str]:
    return WORD_RE.findall(str(text or ""))


def style_vector(text: str) -> List[float]:
    base = text_features(text)
    words = word_list(text)
    lower_words = [word.lower() for word in words]
    word_lengths = [len(word) for word in words]
    length_chars = max(1, len(text))
    punctuation = Counter(char for char in text if char in string.punctuation)
    uppercase_chars = sum(1 for char in text if char.isupper())
    digit_chars = sum(1 for char in text if char.isdigit())
    function_words = sum(1 for word in lower_words if word in FUNCTION_WORDS)

    values = {
        **{name: float(base.get(name, 0.0)) for name in STYLE_FEATURE_NAMES},
        "colon_count": float(text.count(":")),
        "comma_count": float(text.count(",")),
        "period_count": float(text.count(".")),
        "question_count": float(text.count("?")),
        "exclamation_count": float(text.count("!")),
        "parenthesis_count": float(text.count("(") + text.count(")")),
        "digit_ratio": digit_chars / length_chars,
        "uppercase_ratio": uppercase_chars / length_chars,
        "avg_word_length": float(np.mean(word_lengths)) if word_lengths else 0.0,
        "word_length_std": float(np.std(word_lengths)) if len(word_lengths) > 1 else 0.0,
        "stopword_ratio": function_words / len(words) if words else 0.0,
        "function_word_ratio": function_words / len(words) if words else 0.0,
        "punctuation_ratio": sum(punctuation.values()) / length_chars,
    }
    return [float(values.get(name, 0.0)) for name in STYLE_FEATURE_NAMES]


def extract_texts_labels(rows: Sequence[Dict]) -> Tuple[List[str], np.ndarray]:
    texts = [str(row["text"]) for row in rows]
    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    return texts, labels


def build_style_matrix(texts: Sequence[str], scaler: StandardScaler = None, fit: bool = False) -> Tuple[csr_matrix, StandardScaler]:
    dense = np.array([style_vector(text) for text in texts], dtype=float)
    if scaler is None:
        scaler = StandardScaler(with_mean=False)
    scaled = scaler.fit_transform(dense) if fit else scaler.transform(dense)
    return csr_matrix(scaled), scaler


def build_features(
    texts: Sequence[str],
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    fit: bool,
) -> Tuple[csr_matrix, TfidfVectorizer, TfidfVectorizer, StandardScaler]:
    if fit:
        x_word = word_vectorizer.fit_transform(texts)
        x_char = char_vectorizer.fit_transform(texts)
        x_style, scaler = build_style_matrix(texts, scaler=scaler, fit=True)
    else:
        x_word = word_vectorizer.transform(texts)
        x_char = char_vectorizer.transform(texts)
        x_style, scaler = build_style_matrix(texts, scaler=scaler, fit=False)
    return hstack([x_word, x_char, x_style]), word_vectorizer, char_vectorizer, scaler


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    preds = (probs >= threshold).astype(int)
    metrics = {
        "num_samples": int(len(y_true)),
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


def predict_probs(artifacts: Dict, texts: Sequence[str]) -> np.ndarray:
    x, _, _, _ = build_features(
        texts,
        word_vectorizer=artifacts["word_vectorizer"],
        char_vectorizer=artifacts["char_vectorizer"],
        scaler=artifacts["style_scaler"],
        fit=False,
    )
    return artifacts["model"].predict_proba(x)[:, 1]


def prediction_rows(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> List[Dict]:
    output = []
    for index, (row, prob) in enumerate(zip(rows, probs)):
        pred = int(prob >= threshold)
        item = {
            "id": str(row.get("id", index)),
            "label": row.get("label"),
            "prediction": pred,
            "probability": float(prob),
            "prob_llm": float(prob),
            "p_stylometry": float(prob),
        }
        for key in [
            "domain",
            "generator",
            "source",
            "pair_id",
            "bucket",
            "round8_bucket",
            "round8_bucket_family",
            "selection_tier",
            "residual_split",
        ]:
            if row.get(key) is not None:
                item[key] = row.get(key)
        output.append(item)
    return output


def summarize_rows(rows: Sequence[Dict]) -> Dict:
    return {
        "num_rows": len(rows),
        "label_distribution": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "domain_distribution": dict(sorted(Counter(str(row.get("domain", "unknown")) for row in rows).items())),
        "generator_distribution": dict(sorted(Counter(str(row.get("generator", "unknown")) for row in rows).items())),
        "round8_bucket_distribution": dict(sorted(Counter(str(row.get("round8_bucket", "base")) for row in rows).items())),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Round8 stylometry / surface feature branch.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--internal_test", default=str(DEFAULT_INTERNAL_TEST))
    parser.add_argument("--residual_dev", default=str(DEFAULT_RESIDUAL_DEV))
    parser.add_argument("--residual_probe", default=str(DEFAULT_RESIDUAL_PROBE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--word_max_features", type=int, default=50000)
    parser.add_argument("--char_max_features", type=int, default=100000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=3000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": Path(args.train),
        "valid": Path(args.valid),
        "internal_test": Path(args.internal_test),
        "residual_dev": Path(args.residual_dev),
        "residual_probe": Path(args.residual_probe),
    }
    rows_by_split = {name: load_jsonl(path) for name, path in split_paths.items()}
    train_texts, y_train = extract_texts_labels(rows_by_split["train"])

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=args.word_max_features,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=args.char_max_features,
    )
    scaler = StandardScaler(with_mean=False)

    print("=" * 70)
    print("Train Round8 stylometry branch")
    print("=" * 70)
    print(f"Train rows: {len(rows_by_split['train'])}")
    print(f"Output dir: {output_dir}")

    x_train, word_vectorizer, char_vectorizer, scaler = build_features(
        train_texts,
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        scaler=scaler,
        fit=True,
    )
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=20260522,
    )
    model.fit(x_train, y_train)

    artifacts = {
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "style_scaler": scaler,
        "style_feature_names": STYLE_FEATURE_NAMES,
        "threshold": args.threshold,
        "config": vars(args),
    }
    with (output_dir / "stylometry_branch.pkl").open("wb") as f:
        pickle.dump(artifacts, f)

    metrics = {
        "config": vars(args),
        "inputs": {name: str(path) for name, path in split_paths.items()},
        "split_summaries": {name: summarize_rows(rows) for name, rows in rows_by_split.items()},
        "metrics": {},
    }
    prediction_dir = output_dir / "predictions"
    for split_name, rows in rows_by_split.items():
        texts, labels = extract_texts_labels(rows)
        probs = predict_probs(artifacts, texts)
        split_metrics = evaluate(labels, probs, args.threshold)
        metrics["metrics"][split_name] = split_metrics
        save_jsonl(
            prediction_rows(rows, probs, args.threshold),
            prediction_dir / f"stylometry_{split_name}_predictions.jsonl",
        )
        print(
            f"{split_name}: f1={split_metrics['f1']:.4f} "
            f"precision={split_metrics['precision']:.4f} recall={split_metrics['recall']:.4f} "
            f"confusion={split_metrics['confusion_matrix']}"
        )

    write_json(metrics, output_dir / "stylometry_branch_report.json")
    print("\nSaved:")
    print(f"  model:  {output_dir / 'stylometry_branch.pkl'}")
    print(f"  report: {output_dir / 'stylometry_branch_report.json'}")


if __name__ == "__main__":
    main()
