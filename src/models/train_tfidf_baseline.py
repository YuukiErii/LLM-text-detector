import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train.jsonl"
DEFAULT_VALID_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_valid.jsonl"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_internal_test.jsonl"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "tfidf_baseline"


def load_jsonl(path: Path) -> List[Dict]:
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse line {line_id} in {path}: {e}")

    return samples


def extract_texts_labels(samples: List[Dict]) -> Tuple[List[str], np.ndarray, List[str]]:
    texts = []
    labels = []
    ids = []

    for sample in samples:
        text = sample.get("text", "")
        label = sample.get("label")
        sample_id = sample.get("id", "")

        if not isinstance(text, str) or not text.strip():
            continue

        if label not in [0, 1]:
            continue

        texts.append(text)
        labels.append(int(label))
        ids.append(sample_id)

    return texts, np.array(labels), ids


def build_features(
    train_texts: List[str],
    valid_texts: List[str],
    test_texts: List[str],
    word_max_features: int,
    char_max_features: int,
):
    """
    Build word-level and character-level TF-IDF features.

    Word TF-IDF:
        captures lexical choices and short phrase patterns.

    Char TF-IDF:
        captures spelling, punctuation, archaic forms, and local style.
    """

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=word_max_features,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=char_max_features,
    )

    print("Fitting word TF-IDF...")
    x_train_word = word_vectorizer.fit_transform(train_texts)
    x_valid_word = word_vectorizer.transform(valid_texts)
    x_test_word = word_vectorizer.transform(test_texts)

    print("Fitting char TF-IDF...")
    x_train_char = char_vectorizer.fit_transform(train_texts)
    x_valid_char = char_vectorizer.transform(valid_texts)
    x_test_char = char_vectorizer.transform(test_texts)

    print("Concatenating word + char features...")
    x_train = hstack([x_train_word, x_train_char])
    x_valid = hstack([x_valid_word, x_valid_char])
    x_test = hstack([x_test_word, x_test_char])

    print("Feature shapes:")
    print("  train:", x_train.shape)
    print("  valid:", x_valid.shape)
    print("  test: ", x_test.shape)

    return x_train, x_valid, x_test, word_vectorizer, char_vectorizer


def evaluate_model(model, x, y_true, split_name: str) -> Dict:
    y_pred = model.predict(x)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x)[:, 1]
    else:
        y_prob = None

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["human", "llm"],
            zero_division=0,
            output_dict=True,
        ),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = None

    print(f"\n===== {split_name.upper()} RESULTS =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    return metrics


def save_predictions(model, x, samples: List[Dict], ids: List[str], y_true: np.ndarray, output_path: Path):
    y_pred = model.predict(x)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x)[:, 1]
    else:
        y_prob = np.zeros_like(y_pred, dtype=float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_by_id = {sample.get("id"): sample for sample in samples}

    with open(output_path, "w", encoding="utf-8") as f:
        for sample_id, true_label, pred_label, prob in zip(ids, y_true, y_pred, y_prob):
            source_sample = sample_by_id.get(sample_id, {})
            item = {
                "id": sample_id,
                "label": int(true_label),
                "prediction": int(pred_label),
                "prob_llm": float(prob),
                "domain": source_sample.get("domain"),
                "generator": source_sample.get("generator"),
                "source": source_sample.get("source"),
                "pair_id": source_sample.get("pair_id"),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_artifacts(
    output_dir: Path,
    model,
    word_vectorizer,
    char_vectorizer,
    metrics: Dict,
    args,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "tfidf_logreg_model.pkl"
    word_vectorizer_path = output_dir / "word_tfidf_vectorizer.pkl"
    char_vectorizer_path = output_dir / "char_tfidf_vectorizer.pkl"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "config.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(word_vectorizer_path, "wb") as f:
        pickle.dump(word_vectorizer, f)

    with open(char_vectorizer_path, "wb") as f:
        pickle.dump(char_vectorizer, f)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    config = vars(args)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nSaved artifacts:")
    print("  model:          ", model_path)
    print("  word vectorizer:", word_vectorizer_path)
    print("  char vectorizer:", char_vectorizer_path)
    print("  metrics:        ", metrics_path)
    print("  config:         ", config_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Word + Char TF-IDF Logistic Regression baseline."
    )

    parser.add_argument(
        "--train",
        type=str,
        default=str(DEFAULT_TRAIN_PATH),
        help="Path to train JSONL.",
    )

    parser.add_argument(
        "--valid",
        type=str,
        default=str(DEFAULT_VALID_PATH),
        help="Path to valid JSONL.",
    )

    parser.add_argument(
        "--test",
        type=str,
        default=str(DEFAULT_TEST_PATH),
        help="Path to internal test JSONL.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save model and metrics.",
    )

    parser.add_argument(
        "--word_max_features",
        type=int,
        default=100000,
        help="Max features for word TF-IDF.",
    )

    parser.add_argument(
        "--char_max_features",
        type=int,
        default=100000,
        help="Max features for char TF-IDF.",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for Logistic Regression.",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=3000,
        help="Max iterations for Logistic Regression.",
    )

    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Use balanced class weight or no class weight.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_path = Path(args.train)
    valid_path = Path(args.valid)
    test_path = Path(args.test)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Train TF-IDF Baseline")
    print("=" * 70)
    print("Train:", train_path)
    print("Valid:", valid_path)
    print("Test: ", test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find train file: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Cannot find valid file: {valid_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Cannot find test file: {test_path}")

    train_samples = load_jsonl(train_path)
    valid_samples = load_jsonl(valid_path)
    test_samples = load_jsonl(test_path)

    train_texts, y_train, train_ids = extract_texts_labels(train_samples)
    valid_texts, y_valid, valid_ids = extract_texts_labels(valid_samples)
    test_texts, y_test, test_ids = extract_texts_labels(test_samples)

    print("\nLoaded data:")
    print("  train samples:", len(train_texts), "label counts:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("  valid samples:", len(valid_texts), "label counts:", dict(zip(*np.unique(y_valid, return_counts=True))))
    print("  test samples: ", len(test_texts), "label counts:", dict(zip(*np.unique(y_test, return_counts=True))))

    x_train, x_valid, x_test, word_vectorizer, char_vectorizer = build_features(
        train_texts=train_texts,
        valid_texts=valid_texts,
        test_texts=test_texts,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
    )

    class_weight = None if args.class_weight == "none" else "balanced"

    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        class_weight=class_weight,
        solver="liblinear",
        random_state=42,
    )

    model.fit(x_train, y_train)

    valid_metrics = evaluate_model(model, x_valid, y_valid, "valid")
    test_metrics = evaluate_model(model, x_test, y_test, "internal_test")

    metrics = {
        "valid": valid_metrics,
        "internal_test": test_metrics,
    }

    save_artifacts(
        output_dir=output_dir,
        model=model,
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        metrics=metrics,
        args=args,
    )

    prediction_dir = output_dir / "predictions"
    save_predictions(
        model=model,
        x=x_valid,
        samples=valid_samples,
        ids=valid_ids,
        y_true=y_valid,
        output_path=prediction_dir / "tfidf_valid_predictions.jsonl",
    )

    save_predictions(
        model=model,
        x=x_test,
        samples=test_samples,
        ids=test_ids,
        y_true=y_test,
        output_path=prediction_dir / "tfidf_internal_test_predictions.jsonl",
    )

    print("\nPrediction files saved to:")
    print(" ", prediction_dir / "tfidf_valid_predictions.jsonl")
    print(" ", prediction_dir / "tfidf_internal_test_predictions.jsonl")


if __name__ == "__main__":
    main()
