import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_hardneg_p50_l200_a150.jsonl"
DEFAULT_LLM_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_chatgpt_hard_positive.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos_report.json"


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
            if isinstance(item, dict):
                rows.append(item)
    return rows


def save_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_llm_positive(row: Dict) -> Dict:
    item = dict(row)
    item["label"] = 1
    item["generator"] = item.get("generator") or item.get("assigned_generator") or "chatgpt"
    item["source"] = item.get("source") or item["generator"]
    item["generation"] = item.get("generation") or "llm_rewrite"
    item["split"] = "train"
    metadata = dict(item.get("metadata") or {})
    metadata["training_augmentation"] = "chatgpt_hard_positive"
    item["metadata"] = metadata
    return item


def passed_quality(row: Dict, allow_low_quality: bool) -> bool:
    if allow_low_quality:
        return True
    quality = row.get("quality")
    if not isinstance(quality, dict):
        return False
    return quality.get("passed_basic_quality_check") is True


def summarize(rows: List[Dict]) -> Dict:
    return {
        "num_samples": len(rows),
        "num_pair_ids": len({row.get("pair_id") for row in rows}),
        "label_distribution": dict(Counter(str(row.get("label")) for row in rows)),
        "domain_distribution": dict(Counter(row.get("domain", "unknown") for row in rows)),
        "generator_distribution": dict(Counter(row.get("generator", "unknown") for row in rows)),
        "prompt_type_distribution": dict(Counter(row.get("prompt_type", "unknown") for row in rows)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Append LLM hard positives to a train split.")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--llm_positives", type=str, default=str(DEFAULT_LLM_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument(
        "--allow_low_quality",
        action="store_true",
        help="Append rows even if quality.passed_basic_quality_check is missing or false.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = Path(args.train)
    llm_path = Path(args.llm_positives)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find train file: {train_path}")
    if not llm_path.exists():
        raise FileNotFoundError(f"Cannot find LLM positives file: {llm_path}")

    train_rows = load_jsonl(train_path)
    llm_rows_raw = load_jsonl(llm_path)

    existing_ids = {row.get("id") for row in train_rows}
    llm_rows = []
    skipped = Counter()

    for row in llm_rows_raw:
        item = normalize_llm_positive(row)
        if item.get("label") != 1:
            skipped["bad_label"] += 1
            continue
        if item.get("id") in existing_ids:
            skipped["duplicate_id"] += 1
            continue
        if not item.get("pair_id"):
            skipped["missing_pair_id"] += 1
            continue
        if not isinstance(item.get("text"), str) or not item.get("text", "").strip():
            skipped["empty_text"] += 1
            continue
        if not passed_quality(item, allow_low_quality=args.allow_low_quality):
            skipped["quality_not_passed"] += 1
            continue
        existing_ids.add(item.get("id"))
        llm_rows.append(item)

    augmented_rows = train_rows + llm_rows
    save_jsonl(augmented_rows, output_path)

    report = {
        "train_input": str(train_path),
        "llm_positives_input": str(llm_path),
        "output": str(output_path),
        "base_train": summarize(train_rows),
        "llm_positives_raw": summarize(llm_rows_raw),
        "llm_positives_kept": summarize(llm_rows),
        "skipped": dict(skipped),
        "augmented_train": summarize(augmented_rows),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Augment Train With LLM Positives")
    print("=" * 70)
    print(f"Base train rows: {len(train_rows)}")
    print(f"LLM positives raw: {len(llm_rows_raw)}")
    print(f"LLM positives kept: {len(llm_rows)}")
    print(f"Augmented train rows: {len(augmented_rows)}")
    print(f"Skipped: {dict(skipped)}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
