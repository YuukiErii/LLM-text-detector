import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "round4_residual_train.jsonl"
DEFAULT_DEV_HARDPOS = PROJECT_ROOT / "data" / "processed" / "round4_residual_dev_hardpos.jsonl"
DEFAULT_DEV_HARDNEG = PROJECT_ROOT / "data" / "processed" / "round4_residual_dev_hardneg.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round4_human_style_guard"


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_round4_residual_row(row: Dict) -> bool:
    metadata = row.get("metadata") or {}
    return bool(metadata.get("round4_residual_repair")) or str(row.get("round4_source_stage", "")) != "step7_base_train"


def load_guard_train_rows(path: Path) -> List[Dict]:
    rows = []
    for row in load_records(path):
        if row.get("label") not in [0, 1]:
            continue
        if not is_round4_residual_row(row):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        item = dict(row)
        item["guard_label"] = 1 if int(item["label"]) == 0 else 0
        rows.append(item)
    return rows


def load_guard_eval_rows(path: Path) -> List[Dict]:
    rows = []
    for row in load_records(path):
        if row.get("label") not in [0, 1]:
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        item = dict(row)
        item["guard_label"] = 1 if int(item["label"]) == 0 else 0
        rows.append(item)
    return rows


def lexical_shape_features(text: str) -> Dict[str, float]:
    text = str(text or "")
    chars = max(1, len(text))
    letters = [char for char in text if char.isalpha()]
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    lower_words = [word.lower() for word in words]
    long_words = [word for word in words if len(word) >= 9]
    return {
        "uppercase_char_ratio": sum(1 for char in letters if char.isupper()) / max(1, len(letters)),
        "digit_char_ratio": sum(1 for char in text if char.isdigit()) / chars,
        "space_char_ratio": sum(1 for char in text if char.isspace()) / chars,
        "comma_count": float(text.count(",")),
        "colon_count": float(text.count(":")),
        "paren_count": float(text.count("(") + text.count(")")),
        "long_word_ratio": len(long_words) / max(1, len(words)),
        "avg_word_len": sum(len(word) for word in words) / max(1, len(words)),
        "first_person_count": float(sum(1 for word in lower_words if word in {"i", "me", "my", "mine", "we", "our"})),
    }


def feature_dict(row: Dict) -> Dict:
    text = str(row.get("text", ""))
    bucket = row.get("bucket") or assign_bucket(text)
    features = text_features(text)
    features.pop("bucket", None)
    features.update(lexical_shape_features(text))
    features["bucket"] = str(bucket)
    features["round4_bucket"] = str(row.get("round4_bucket") or bucket)
    return features


def labels_for(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([int(row["guard_label"]) for row in rows], dtype=int)


def build_features(
    train_rows: Sequence[Dict],
    eval_blocks: Sequence[Sequence[Dict]],
    char_max_features: int,
):
    train_texts = [str(row.get("text", "")) for row in train_rows]
    eval_texts = [[str(row.get("text", "")) for row in rows] for rows in eval_blocks]

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=char_max_features,
    )
    dict_vectorizer = DictVectorizer(sparse=True)

    x_train_char = char_vectorizer.fit_transform(train_texts)
    x_eval_char = [char_vectorizer.transform(texts) for texts in eval_texts]

    x_train_dict = dict_vectorizer.fit_transform([feature_dict(row) for row in train_rows])
    x_eval_dict = [dict_vectorizer.transform([feature_dict(row) for row in rows]) for rows in eval_blocks]

    x_train = hstack([x_train_char, x_train_dict])
    x_eval = [hstack([x_char, x_dict]) for x_char, x_dict in zip(x_eval_char, x_eval_dict)]
    return x_train, x_eval, char_vectorizer, dict_vectorizer


def metrics_for(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> Dict:
    labels = labels_for(rows)
    preds = (probs >= threshold).astype(int)
    metrics = {
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_human_style": float(precision_score(labels, preds, zero_division=0)),
        "recall_human_style": float(recall_score(labels, preds, zero_division=0)),
        "f1_human_style": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "threshold": threshold,
        "veto_rate": float(np.mean(preds == 1)) if len(preds) else 0.0,
        "mean_p_human_style": float(np.mean(probs)) if len(probs) else 0.0,
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def prediction_rows(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> List[Dict]:
    out = []
    for row, prob in zip(rows, probs):
        item = {
            "id": row.get("id"),
            "label": row.get("label"),
            "guard_label": row.get("guard_label"),
            "p_human_style": float(prob),
            "human_style_prediction": int(prob >= threshold),
            "guard_threshold": threshold,
            "bucket": row.get("bucket"),
            "round4_bucket": row.get("round4_bucket"),
            "round4_tag": row.get("round4_tag"),
            "domain": row.get("domain"),
            "pair_id": row.get("pair_id"),
        }
        out.append(item)
    return out


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round4 Human-Style Guard Report",
        "",
        "Positive guard label means the row looks like high-style human text. The guard is a veto signal, not a final LLM detector.",
        "",
        "| Split | n | Accuracy | Human-Style Precision | Human-Style Recall | Human-Style F1 | Veto Rate | Mean P(human-style) | Confusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, block in report["metrics"].items():
        lines.append(
            f"| {name} | {block['num_samples']} | {block['accuracy']:.4f} | "
            f"{block['precision_human_style']:.4f} | {block['recall_human_style']:.4f} | "
            f"{block['f1_human_style']:.4f} | {block['veto_rate']:.4f} | "
            f"{block['mean_p_human_style']:.4f} | {block['confusion_matrix']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Round4 human-style guard branch.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev_hardpos", default=str(DEFAULT_DEV_HARDPOS))
    parser.add_argument("--dev_hardneg", default=str(DEFAULT_DEV_HARDNEG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--C", type=float, default=0.75)
    parser.add_argument("--char_max_features", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_guard_train_rows(Path(args.train))
    dev_hardpos = load_guard_eval_rows(Path(args.dev_hardpos))
    dev_hardneg = load_guard_eval_rows(Path(args.dev_hardneg))
    dev_combined = dev_hardpos + dev_hardneg
    if not train_rows:
        raise ValueError("No Round4 residual train rows found for human-style guard.")

    x_train, x_eval, char_vectorizer, dict_vectorizer = build_features(
        train_rows,
        [train_rows, dev_hardpos, dev_hardneg, dev_combined],
        char_max_features=args.char_max_features,
    )
    y_train = labels_for(train_rows)

    model = LogisticRegression(
        C=args.C,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=args.seed,
    )
    model.fit(x_train, y_train)

    blocks = {
        "train_residual": (train_rows, x_eval[0]),
        "dev_hardpos_should_not_veto": (dev_hardpos, x_eval[1]),
        "dev_hardneg_should_veto": (dev_hardneg, x_eval[2]),
        "dev_combined": (dev_combined, x_eval[3]),
    }
    metrics = {}
    for name, (rows, x_block) in blocks.items():
        probs = model.predict_proba(x_block)[:, 1]
        metrics[name] = metrics_for(rows, probs, threshold=args.threshold)
        save_jsonl(prediction_rows(rows, probs, threshold=args.threshold), output_dir / f"{name}_predictions.jsonl")

    artifact = {
        "model": model,
        "char_vectorizer": char_vectorizer,
        "dict_vectorizer": dict_vectorizer,
        "threshold": args.threshold,
        "feature_version": "round4_human_style_guard_v1",
    }
    model_path = output_dir / "human_style_guard.pkl"
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    report = {
        "model_path": str(model_path),
        "train": args.train,
        "dev_hardpos": args.dev_hardpos,
        "dev_hardneg": args.dev_hardneg,
        "threshold": args.threshold,
        "C": args.C,
        "char_max_features": args.char_max_features,
        "train_rows": len(train_rows),
        "metrics": metrics,
        "interpretation": {
            "positive_guard_label": "high_style_human",
            "use": "veto unsafe human-to-LLM overrides when p_human_style is high",
        },
    }
    (output_dir / "human_style_guard_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(report, PROJECT_ROOT / "outputs" / "evaluation" / "round4_human_style_guard_report.md")

    print("=" * 70)
    print("Round4 human-style guard trained")
    print("=" * 70)
    print(f"Train rows: {len(train_rows)}")
    print(f"Model: {model_path}")
    for name, block in metrics.items():
        print(
            f"{name}: acc={block['accuracy']:.4f} "
            f"human_recall={block['recall_human_style']:.4f} "
            f"veto_rate={block['veto_rate']:.4f} "
            f"confusion={block['confusion_matrix']}"
        )


if __name__ == "__main__":
    main()
