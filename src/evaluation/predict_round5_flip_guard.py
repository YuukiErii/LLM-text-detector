import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from scipy.sparse import hstack


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from evaluation.assign_text_bucket import load_records
from train_round5_flip_guard import feature_dict, metrics_for, target_label


DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "round5_flip_guard"


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_artifact(model_dir: Path) -> Dict:
    path = model_dir / "flip_guard.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find Round5 flip guard artifact: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def load_input_rows(path: Path, split: str = "", override_candidates_only: bool = False) -> List[Dict]:
    rows = []
    for row in load_records(path):
        if split and row.get("split") != split:
            continue
        if override_candidates_only and not row.get("round4_override_candidate"):
            continue
        if not str(row.get("text", "")).strip():
            continue
        item = dict(row)
        label = target_label(item)
        if label in [0, 1]:
            item["flip_guard_label"] = int(label)
        rows.append(item)
    return rows


def output_rows(rows: List[Dict], probs, threshold: float, include_text: bool) -> List[Dict]:
    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["p_unsafe_override"] = float(prob)
        item["flip_guard_veto"] = int(prob >= threshold)
        item["flip_guard_threshold"] = float(threshold)
        label = target_label(row)
        if label in [0, 1]:
            item["flip_guard_label"] = int(label)
        if not include_text:
            item.pop("text", None)
        out.append(item)
    return out


def write_metrics(metrics: Optional[Dict], path: str) -> None:
    if not path or metrics is None:
        return
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Round5 flip-guard unsafe override probabilities.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--split", default="")
    parser.add_argument("--override_candidates_only", action="store_true")
    parser.add_argument("--include_text", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_input_rows(
        Path(args.input),
        split=args.split,
        override_candidates_only=args.override_candidates_only,
    )
    if not rows:
        raise ValueError(f"No usable rows found in {args.input}")

    artifact = load_artifact(Path(args.model_dir))
    model = artifact["model"]
    word_vectorizer = artifact["word_vectorizer"]
    char_vectorizer = artifact["char_vectorizer"]
    dict_vectorizer = artifact["dict_vectorizer"]
    threshold = float(args.threshold if args.threshold is not None else artifact.get("threshold", 0.5))

    texts = [str(row.get("text", "")) for row in rows]
    x_word = word_vectorizer.transform(texts)
    x_char = char_vectorizer.transform(texts)
    x_dict = dict_vectorizer.transform([feature_dict(row) for row in rows])
    x = hstack([x_word, x_char, x_dict])
    probs = model.predict_proba(x)[:, 1]

    predictions = output_rows(rows, probs, threshold, include_text=args.include_text)
    save_jsonl(predictions, Path(args.output))
    metrics = metrics_for(rows, probs, threshold)
    write_metrics(metrics, args.metrics)

    print("=" * 70)
    print("Round5 flip guard predictions written")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.output}")
    print(f"Veto rate: {metrics['veto_rate_all_rows']:.4f}")
    if metrics["num_labeled"]:
        print(f"Unsafe recall/protection: {metrics['unsafe_recall_protection']:.4f}")
        print(f"Safe veto rate: {metrics['safe_veto_rate']:.4f}")
        print(f"Confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
