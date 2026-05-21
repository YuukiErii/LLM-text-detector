import argparse
import json
import re
import string
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

ARCHAIC_WORDS = {
    "afore",
    "anon",
    "art",
    "aught",
    "doth",
    "dost",
    "ere",
    "hath",
    "hence",
    "hither",
    "mayst",
    "methinks",
    "naught",
    "nay",
    "nigh",
    "oft",
    "shalt",
    "thee",
    "thine",
    "thou",
    "thy",
    "tis",
    "twas",
    "whence",
    "wherefore",
    "whilst",
    "ye",
}

ACADEMIC_MARKERS = {
    "algorithm",
    "analysis",
    "approach",
    "baseline",
    "classifier",
    "corpus",
    "dataset",
    "evaluation",
    "experiment",
    "language",
    "method",
    "model",
    "parameter",
    "precision",
    "recall",
    "representation",
    "result",
    "semantic",
    "statistical",
    "syntax",
    "system",
    "task",
    "training",
}

POETIC_MARKERS = {
    "dew",
    "flame",
    "glen",
    "grace",
    "heart",
    "heaven",
    "moon",
    "muse",
    "poem",
    "poesy",
    "rhyme",
    "sea",
    "sonnet",
    "star",
    "stars",
    "thou",
    "thy",
    "verse",
}


def load_records(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
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
                if isinstance(item, dict):
                    rows.append(item)
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ["data", "samples", "records", "items"]:
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    raise ValueError(f"Unsupported input format: {path}")


def save_records(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_list(text: str) -> List[str]:
    return WORD_RE.findall(str(text or ""))


def text_features(text: str) -> Dict[str, float]:
    text = str(text or "")
    words = word_list(text)
    lower_words = [word.lower() for word in words]
    unique_words = set(lower_words)
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        non_empty_lines = [text]

    sentence_lengths = []
    for part in SENTENCE_SPLIT_RE.split(text):
        count = len(word_list(part))
        if count:
            sentence_lengths.append(count)

    length_chars = len(text)
    length_words = len(words)
    num_lines = max(1, len(non_empty_lines))
    avg_line_length = sum(len(line) for line in non_empty_lines) / num_lines

    quote_count = sum(text.count(mark) for mark in ["'", '"', "`", "\u2018", "\u2019", "\u201c", "\u201d"])
    dash_count = sum(text.count(mark) for mark in ["-", "\u2013", "\u2014"])
    punctuation_count = sum(1 for char in text if char in string.punctuation)

    features = {
        "length_chars": float(length_chars),
        "length_words": float(length_words),
        "num_lines": float(num_lines),
        "linebreak_ratio": 0.0 if length_chars == 0 else text.count("\n") / length_chars,
        "avg_line_length": float(avg_line_length),
        "punctuation_ratio": 0.0 if length_chars == 0 else punctuation_count / length_chars,
        "quote_count": float(quote_count),
        "dash_count": float(dash_count),
        "semicolon_count": float(text.count(";")),
        "archaic_word_count": float(sum(1 for word in lower_words if word in ARCHAIC_WORDS)),
        "academic_marker_count": float(sum(1 for word in lower_words if word in ACADEMIC_MARKERS)),
        "poetic_marker_count": float(sum(1 for word in lower_words if word in POETIC_MARKERS)),
        "type_token_ratio": 0.0 if length_words == 0 else len(unique_words) / length_words,
        "sentence_length_mean": float(mean(sentence_lengths)) if sentence_lengths else 0.0,
        "sentence_length_std": float(pstdev(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0,
    }
    features["bucket"] = assign_bucket_from_features(text, features)
    return features


def assign_bucket_from_features(text: str, features: Dict[str, float]) -> str:
    lower = str(text or "").lower()
    words = int(features["length_words"])
    num_lines = int(features["num_lines"])
    avg_line_length = features["avg_line_length"]
    archaic = int(features["archaic_word_count"])
    academic = int(features["academic_marker_count"])
    poetic = int(features["poetic_marker_count"])

    citation_like = bool(
        re.search(r"\([A-Z][A-Za-z-]+,\s*\d{4}\)|\bet al\.\b|\btable\s+\d+|\bfigure\s+\d+", text)
    )
    old_prose_like = bool(
        re.search(r"\b(whilst|whereupon|thereof|hitherto|nay|alas|ere|hath|doth)\b", lower)
    )

    if num_lines >= 4 and avg_line_length <= 72:
        if archaic >= 1 or poetic >= 2:
            return "poetry_classical"
        return "poetry_freeverse"
    if words <= 55 and (poetic >= 2 or archaic >= 1 or (num_lines >= 2 and avg_line_length <= 90)):
        if archaic >= 1 or "sonnet" in lower or "poesy" in lower:
            return "poetry_classical"
        return "poetry_freeverse"
    if academic >= 4 or citation_like:
        return "academic_formal"
    if archaic >= 2 or old_prose_like:
        return "literary_old_prose"
    if words <= 55:
        return "literary_short_fragment"
    return "general_prose"


def assign_bucket(text: str) -> str:
    return str(text_features(text)["bucket"])


def enrich_row(row: Dict, text_key: str = "text", bucket_key: str = "bucket") -> Dict:
    item = dict(row)
    features = text_features(str(item.get(text_key, "")))
    item[bucket_key] = features.pop("bucket")
    for key, value in features.items():
        item.setdefault(key, value)
    return item


def parse_args():
    parser = argparse.ArgumentParser(description="Assign transparent round2 text buckets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--bucket_key", default="bucket")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = [enrich_row(row, text_key=args.text_key, bucket_key=args.bucket_key) for row in load_records(input_path)]
    save_records(rows, output_path)
    print("=" * 70)
    print("Assigned round2 text buckets")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Rows: {len(rows)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
