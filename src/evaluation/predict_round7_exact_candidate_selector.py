import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from scipy.sparse import hstack


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import load_records
from models.train_round7_exact_candidate_selector import feature_dict, selector_label


DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "round7_exact_candidate_selector" / "selector.pkl"


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: Dict, index: int) -> str:
    value = row.get("id")
    return str(value) if value not in [None, ""] else f"row_{index:06d}"


def load_rows(path: Path) -> List[Dict]:
    return [dict(row) for row in load_records(path)]


def merge_text_source(rows: List[Dict], text_source: str = "") -> List[Dict]:
    if not text_source:
        return [row for row in rows if str(row.get("text") or "").strip()]
    source_rows = load_rows(Path(text_source))
    source_by_id = {row_id(row, index): row for index, row in enumerate(source_rows)}
    merged = []
    for index, row in enumerate(rows):
        item = dict(row)
        source = source_by_id.get(row_id(row, index))
        if source:
            for key in [
                "text",
                "bucket",
                "round4_bucket",
                "round4_tag",
                "domain",
                "generator",
                "source",
                "word_count",
                "flip_type",
                "override_safety",
                "round4_override_candidate",
            ]:
                if item.get(key) in [None, ""] and source.get(key) not in [None, ""]:
                    item[key] = source[key]
        if str(item.get("text") or "").strip():
            merged.append(item)
    return merged


def predict_rows(rows: List[Dict], artifact: Dict) -> List[Dict]:
    texts = [str(row.get("text") or "") for row in rows]
    x = hstack(
        [
            artifact["word_vectorizer"].transform(texts),
            artifact["char_vectorizer"].transform(texts),
            artifact["dict_vectorizer"].transform([feature_dict(row) for row in rows]),
        ]
    )
    probs = artifact["model"].predict_proba(x)[:, 1]
    threshold = float(artifact["threshold"])
    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["p_round7_safe_override"] = float(prob)
        item["round7_exact_selector_pass"] = int(prob >= threshold)
        item["round7_exact_selector_prediction"] = int(prob >= threshold)
        item["round7_exact_selector_threshold"] = threshold
        try:
            item["round7_selector_label"] = selector_label(item)
        except ValueError:
            pass
        out.append(item)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Predict p_round7_safe_override with the exact-candidate selector.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text_source", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.model).open("rb") as f:
        artifact = pickle.load(f)
    rows = merge_text_source(load_rows(Path(args.input)), args.text_source)
    out = predict_rows(rows, artifact)
    save_jsonl(out, Path(args.output))
    print("=" * 70)
    print("Round7 exact selector predictions written")
    print("=" * 70)
    print(f"Input rows: {len(rows)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
