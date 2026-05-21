import argparse
import csv
import json
import re
import string
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "predictions" / "teacher_test_step7_ensemble_raw_tfidf_predictions.jsonl"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "teacher_test.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "round2" / "error_ledger_teacher_step7.csv"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "outputs" / "round2" / "error_ledger_teacher_step7.jsonl"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[.!?]+")

ARCHAIC_WORDS = {
    "afore",
    "aforetime",
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
    "moone",
    "naught",
    "nighte",
    "o'er",
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
    "ye",
    "aire",
    "brighte",
    "calme",
    "darknesse",
    "eke",
    "glowe",
    "hower",
    "lighte",
    "plighte",
    "skie",
    "starres",
    "streche",
}

ACADEMIC_MARKERS = {
    "algorithm",
    "analysis",
    "approach",
    "baseline",
    "corpus",
    "data",
    "dataset",
    "evaluation",
    "experiment",
    "method",
    "model",
    "paper",
    "parameter",
    "precision",
    "recall",
    "result",
    "statistical",
    "study",
    "system",
    "table",
    "task",
    "training",
}

POETIC_MARKERS = {
    "art",
    "banner",
    "chief",
    "dew",
    "flame",
    "glen",
    "grace",
    "grass",
    "hail",
    "heart",
    "heaven",
    "line",
    "measure",
    "moon",
    "muse",
    "muses",
    "music",
    "pine",
    "poem",
    "poems",
    "poesy",
    "poetry",
    "rhyme",
    "romance",
    "sand",
    "seaweed",
    "silken",
    "sky",
    "sonnet",
    "stars",
    "temple",
    "triumph",
    "tune",
    "waves",
}

FIELDNAMES = [
    "id",
    "label",
    "prediction",
    "error_type",
    "probability",
    "p_tfidf",
    "p_deberta",
    "text",
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
    "rough_domain",
    "confidence_bucket",
    "notes",
]


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json_records(path: Optional[Path]) -> List[Dict]:
    if path is None:
        return []
    if path.suffix.lower() == ".jsonl":
        records = []
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
    raise ValueError(f"Unsupported JSON format: {path}")


def raw_records_by_id(records: List[Dict]) -> Dict[str, Dict]:
    by_id = {}
    for index, row in enumerate(records):
        by_id[str(index)] = row
        if row.get("id") is not None:
            by_id[str(row["id"])] = row
    return by_id


def words_for(text: str) -> List[str]:
    return WORD_RE.findall(text)


def rough_domain_for(text: str, words: List[str], line_count: int, avg_line_length: float, archaic_count: int, academic_count: int) -> str:
    word_count = len(words)
    lower_text = text.lower()
    lower_words = [word.lower() for word in words]
    poetic_marker_count = sum(1 for word in lower_words if word in POETIC_MARKERS)
    comma_count = text.count(",")
    sentence_mark_count = len(SENTENCE_RE.findall(text))
    multi_space_runs = len(re.findall(r" {2,}", text))
    poetic_contraction_count = len(re.findall(r"[A-Za-z]+[’']d\b", text))
    citation_like = bool(re.search(r"\([A-Z][A-Za-z]+,\s*\d{4}\)|\bet al\.\b|\bfigure\s+\d+|\btable\s+\d+", text))

    if line_count >= 4 and avg_line_length <= 55:
        if archaic_count >= 2:
            return "poetry_classical"
        return "poetry_freeverse"
    if word_count <= 150 and (multi_space_runs >= 2 or archaic_count >= 3 or poetic_contraction_count >= 3 or poetic_marker_count >= 3):
        if archaic_count >= 2 or poetic_contraction_count >= 3 or any(marker in lower_text for marker in ["poesy", "muse", "muses", "sonnet"]):
            return "poetry_classical"
        return "poetry_freeverse"
    if word_count <= 50 and (poetic_marker_count >= 1 or (comma_count >= 2 and sentence_mark_count <= 1)):
        if archaic_count >= 1:
            return "poetry_classical"
        return "poetry_freeverse"
    if line_count >= 2 and word_count <= 45:
        return "poetry_or_fragment"
    if academic_count >= 5 or citation_like:
        return "academic_formal"
    if archaic_count >= 2 or re.search(r"\b(whilst|whereupon|thereof|hitherto|nay|alas)\b", lower_text):
        return "literary_old_prose"
    if word_count <= 45:
        return "literary_short_fragment"
    return "general_prose"


def confidence_bucket(probability: float, threshold: float) -> str:
    margin = abs(probability - threshold)
    if margin <= 0.05:
        return "boundary"
    if margin <= 0.15:
        return "near_boundary"
    if probability >= 0.90:
        return "very_high_llm"
    if probability <= 0.10:
        return "very_high_human"
    if probability >= threshold:
        return "confident_llm"
    return "confident_human"


def error_type(label: int, prediction: int) -> str:
    if label == 0 and prediction == 1:
        return "false_positive"
    if label == 1 and prediction == 0:
        return "false_negative"
    if label == 1 and prediction == 1:
        return "true_positive"
    return "true_negative"


def build_notes(row: Dict, rough_domain: str, error: str, threshold: float) -> str:
    notes = []
    p_tfidf = row.get("p_tfidf")
    p_deberta = row.get("p_deberta")
    if p_tfidf is not None and p_deberta is not None:
        tfidf_pred = int(to_float(p_tfidf) >= threshold)
        deberta_pred = int(to_float(p_deberta) >= threshold)
        if tfidf_pred != deberta_pred:
            notes.append("tfidf_deberta_disagree")
    if error == "false_positive":
        if rough_domain.startswith("poetry"):
            notes.append("human_poetry_or_fragment_fp")
        elif rough_domain == "academic_formal":
            notes.append("formal_academic_human_fp")
        elif rough_domain == "literary_old_prose":
            notes.append("old_or_ornate_human_fp")
        else:
            notes.append("human_style_fp")
    elif error == "false_negative":
        if rough_domain.startswith("poetry"):
            notes.append("poetry_llm_fn")
        elif rough_domain == "academic_formal":
            notes.append("natural_academic_llm_fn")
        elif rough_domain == "literary_old_prose":
            notes.append("old_fiction_style_llm_fn")
        else:
            notes.append("conservative_llm_rewrite_fn")
    return ";".join(notes)


def feature_row(prediction: Dict, raw_by_id: Dict[str, Dict], index: int, threshold: float) -> Dict:
    sample_id = str(prediction.get("id", index))
    raw = raw_by_id.get(sample_id, {})
    text = str(prediction.get("text") or raw.get("text") or "")
    label = int(prediction.get("label", raw.get("label")))
    pred = int(prediction.get("prediction", prediction.get("pred", prediction.get("label_pred"))))
    probability = to_float(prediction.get("prob_llm", prediction.get("probability", prediction.get("score"))))

    lines = text.splitlines() or [text]
    non_empty_lines = [line for line in lines if line.strip()]
    words = words_for(text)
    lower_words = [word.lower() for word in words]
    length_chars = len(text)
    length_words = len(words)
    num_lines = max(1, len(non_empty_lines))
    linebreak_ratio = 0.0 if length_chars == 0 else text.count("\n") / length_chars
    avg_line_length = 0.0 if not non_empty_lines else sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
    punctuation_ratio = 0.0 if length_chars == 0 else sum(1 for char in text if char in string.punctuation) / length_chars
    quote_count = sum(text.count(char) for char in ["'", '"', "`", "’", "‘", "“", "”"])
    dash_count = sum(text.count(char) for char in ["-", "–", "—"])
    semicolon_count = text.count(";")
    archaic_count = sum(1 for word in lower_words if word in ARCHAIC_WORDS)
    academic_count = sum(1 for word in lower_words if word in ACADEMIC_MARKERS)
    rough_domain = rough_domain_for(text, words, num_lines, avg_line_length, archaic_count, academic_count)
    err = error_type(label, pred)

    return {
        "id": sample_id,
        "label": label,
        "prediction": pred,
        "error_type": err,
        "probability": probability,
        "p_tfidf": prediction.get("p_tfidf"),
        "p_deberta": prediction.get("p_deberta"),
        "text": text,
        "length_chars": length_chars,
        "length_words": length_words,
        "num_lines": num_lines,
        "linebreak_ratio": linebreak_ratio,
        "avg_line_length": avg_line_length,
        "punctuation_ratio": punctuation_ratio,
        "quote_count": quote_count,
        "dash_count": dash_count,
        "semicolon_count": semicolon_count,
        "archaic_word_count": archaic_count,
        "academic_marker_count": academic_count,
        "rough_domain": rough_domain,
        "confidence_bucket": confidence_bucket(probability, threshold),
        "notes": build_notes(prediction, rough_domain, err, threshold),
    }


def save_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDNAMES})


def parse_args():
    parser = argparse.ArgumentParser(description="Export a feature-rich Phase 0 error ledger.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Optional raw JSON/JSONL input with text.")
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output_jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--errors_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    prediction_path = Path(args.predictions)
    input_path = Path(args.input) if args.input else None
    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)

    predictions = load_json_records(prediction_path)
    raw_records = load_json_records(input_path) if input_path else []
    raw_by_id = raw_records_by_id(raw_records)

    rows = []
    skipped = 0
    for index, prediction in enumerate(predictions):
        try:
            row = feature_row(prediction, raw_by_id, index=index, threshold=args.threshold)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if args.errors_only and row["error_type"] in {"true_positive", "true_negative"}:
            continue
        rows.append(row)

    save_csv(rows, output_csv)
    save_jsonl(rows, output_jsonl)

    error_count = sum(1 for row in rows if row["error_type"] in {"false_positive", "false_negative"})
    print("=" * 70)
    print("Round2 error ledger exported")
    print("=" * 70)
    print(f"Predictions: {prediction_path}")
    print(f"Rows written: {len(rows)}")
    print(f"Errors in written rows: {error_count}")
    print(f"Skipped malformed rows: {skipped}")
    print(f"CSV: {output_csv}")
    print(f"JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
