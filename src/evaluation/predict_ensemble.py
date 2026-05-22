import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.text_normalization import normalize_text

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "teacher_test.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "predictions" / "teacher_test_ensemble_predictions.jsonl"
DEFAULT_SUBMISSION_PATH = PROJECT_ROOT / "outputs" / "predictions" / "teacher_test_submission.json"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "teacher_test_ensemble_metrics.json"

DEFAULT_TFIDF_DIR = PROJECT_ROOT / "outputs" / "models" / "tfidf_lit_academic_poetry"
DEFAULT_DEBERTA_DIR = PROJECT_ROOT / "outputs" / "models" / "deberta_lit_academic_poetry"
DEFAULT_FUSION_CONFIG = PROJECT_ROOT / "outputs" / "models" / "ensemble_lit_academic_poetry_fine" / "fusion_config.json"


def load_records(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse {path}, line {line_id}: {e}") from e
                records.append(item)
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "samples", "records", "items"]:
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Unsupported input format in {path}. Expected JSON list or JSONL.")


def normalize_records(records: Iterable[Dict]) -> List[Dict]:
    normalized = []
    for idx, item in enumerate(records):
        text = item.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        sample_id = item.get("id")
        if sample_id is None or sample_id == "":
            sample_id = str(idx)

        row = dict(item)
        row["id"] = str(sample_id)
        row["text"] = text
        normalized.append(row)

    return normalized


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_tfidf_artifacts(tfidf_dir: Path):
    model = load_pickle(tfidf_dir / "tfidf_logreg_model.pkl")
    word_vectorizer = load_pickle(tfidf_dir / "word_tfidf_vectorizer.pkl")
    char_vectorizer = load_pickle(tfidf_dir / "char_tfidf_vectorizer.pkl")
    return model, word_vectorizer, char_vectorizer


def load_tfidf_config(tfidf_dir: Path) -> Dict:
    config_path = tfidf_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def maybe_normalize_for_tfidf(texts: List[str], config: Dict) -> List[str]:
    if not config.get("normalize_text"):
        return texts

    return [
        normalize_text(
            text,
            repair_mojibake=not config.get("no_repair_mojibake", False),
            use_ftfy=config.get("use_ftfy", False),
            unicode_form=config.get("unicode_form", "NFKC"),
            normalize_quotes=not config.get("no_normalize_quotes", False),
            normalize_dashes=not config.get("no_normalize_dashes", False),
            normalize_spaces=not config.get("no_normalize_spaces", False),
            preserve_linebreaks=not config.get("collapse_linebreaks", False),
        )
        for text in texts
    ]


def predict_tfidf(samples: List[Dict], tfidf_dir: Path) -> np.ndarray:
    model, word_vectorizer, char_vectorizer = load_tfidf_artifacts(tfidf_dir)
    config = load_tfidf_config(tfidf_dir)
    texts = maybe_normalize_for_tfidf([sample["text"] for sample in samples], config)
    x_word = word_vectorizer.transform(texts)
    x_char = char_vectorizer.transform(texts)
    x = hstack([x_word, x_char])

    if not hasattr(model, "predict_proba"):
        raise ValueError("TF-IDF model does not expose predict_proba; cannot ensemble probabilities.")

    return model.predict_proba(x)[:, 1]


def predict_deberta(samples: List[Dict], deberta_dir: Path, batch_size: int, max_length: int) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = deberta_dir / "best_model"
    tokenizer_dir = deberta_dir / "tokenizer"
    tokenizer_path = tokenizer_dir if tokenizer_dir.exists() else model_dir

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_probs = []
    texts = [sample["text"] for sample in samples]

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    return np.array(all_probs, dtype=float)


def load_fusion_config(path: Path, alpha: Optional[float], threshold: Optional[float]) -> Tuple[float, float]:
    config = {}
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))

    final_alpha = alpha if alpha is not None else float(config.get("alpha", 0.33))
    final_threshold = threshold if threshold is not None else float(config.get("threshold", 0.48))
    return final_alpha, final_threshold


def evaluate_if_labeled(samples: List[Dict], probs: np.ndarray, predictions: np.ndarray) -> Optional[Dict]:
    labels = []
    for sample in samples:
        label = sample.get("label")
        if label not in [0, 1]:
            return None
        labels.append(int(label))

    y_true = np.array(labels)
    metrics = {
        "num_samples": len(samples),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, probs)
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def write_predictions(
    samples: List[Dict],
    tfidf_probs: np.ndarray,
    deberta_probs: np.ndarray,
    ensemble_probs: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    include_text: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for sample, p_tfidf, p_deberta, p_ensemble, pred in zip(
            samples, tfidf_probs, deberta_probs, ensemble_probs, predictions
        ):
            row = {
                "id": sample["id"],
                "prediction": int(pred),
                "label": sample.get("label"),
                "probability": float(p_ensemble),
                "prob_llm": float(p_ensemble),
                "p_tfidf": float(p_tfidf),
                "p_deberta": float(p_deberta),
            }
            for key in [
                "domain",
                "generator",
                "source",
                "pair_id",
                "bucket",
                "round2_tag",
                "round3_tag",
                "round4_bucket",
                "round4_tag",
            ]:
                if sample.get(key) is not None:
                    row[key] = sample.get(key)
            if include_text:
                row["text"] = sample["text"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_submission(
    samples: List[Dict],
    probs: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    minimal: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if minimal:
        submission = [
            {
                "label": int(pred),
                "probability": float(prob),
            }
            for prob, pred in zip(probs, predictions)
        ]
    else:
        submission = [
            {
                "id": sample["id"],
                "label": int(pred),
                "probability": float(prob),
            }
            for sample, prob, pred in zip(samples, probs, predictions)
        ]

    output_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run final TF-IDF + DeBERTa ensemble inference.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--submission", type=str, default=str(DEFAULT_SUBMISSION_PATH))
    parser.add_argument("--metrics", type=str, default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--tfidf_dir", type=str, default=str(DEFAULT_TFIDF_DIR))
    parser.add_argument("--deberta_dir", type=str, default=str(DEFAULT_DEBERTA_DIR))
    parser.add_argument("--fusion_config", type=str, default=str(DEFAULT_FUSION_CONFIG))
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--include_text", action="store_true")
    parser.add_argument("--no_submission", action="store_true")
    parser.add_argument(
        "--minimal_submission",
        action="store_true",
        help="Write submission records as {label, probability} without id.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    submission_path = Path(args.submission)
    metrics_path = Path(args.metrics)
    tfidf_dir = Path(args.tfidf_dir)
    deberta_dir = Path(args.deberta_dir)
    fusion_config_path = Path(args.fusion_config)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")
    if not tfidf_dir.exists():
        raise FileNotFoundError(f"Cannot find TF-IDF model dir: {tfidf_dir}")
    if not deberta_dir.exists():
        raise FileNotFoundError(f"Cannot find DeBERTa model dir: {deberta_dir}")

    alpha, threshold = load_fusion_config(fusion_config_path, args.alpha, args.threshold)

    samples = normalize_records(load_records(input_path))
    if not samples:
        raise ValueError(f"No valid text samples found in {input_path}")

    print("=" * 70)
    print("Run Final Ensemble Inference")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Samples: {len(samples)}")
    print(f"TF-IDF dir: {tfidf_dir}")
    print(f"DeBERTa dir: {deberta_dir}")
    print(f"Alpha: {alpha}")
    print(f"Threshold: {threshold}")

    print("\nPredicting with TF-IDF...")
    tfidf_probs = predict_tfidf(samples, tfidf_dir)

    print("Predicting with DeBERTa...")
    deberta_probs = predict_deberta(samples, deberta_dir, args.batch_size, args.max_length)

    ensemble_probs = alpha * deberta_probs + (1.0 - alpha) * tfidf_probs
    predictions = (ensemble_probs >= threshold).astype(int)

    write_predictions(
        samples=samples,
        tfidf_probs=tfidf_probs,
        deberta_probs=deberta_probs,
        ensemble_probs=ensemble_probs,
        predictions=predictions,
        output_path=output_path,
        include_text=args.include_text,
    )

    if not args.no_submission:
        write_submission(samples, ensemble_probs, predictions, submission_path, minimal=args.minimal_submission)

    metrics = evaluate_if_labeled(samples, ensemble_probs, predictions)
    if metrics is not None:
        metrics["alpha"] = alpha
        metrics["threshold"] = threshold
        metrics["input"] = str(input_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  predictions: {output_path}")
    if not args.no_submission:
        print(f"  submission:  {submission_path}")
    if metrics is not None:
        print(f"  metrics:     {metrics_path}")
        print("\nMetrics:")
        print(f"  accuracy:  {metrics['accuracy']:.4f}")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall:    {metrics['recall']:.4f}")
        print(f"  f1:        {metrics['f1']:.4f}")
        if metrics.get("roc_auc") is not None:
            print(f"  roc_auc:   {metrics['roc_auc']:.4f}")
        print(f"  confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
