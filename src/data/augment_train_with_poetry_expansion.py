import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train.jsonl"
DEFAULT_HUMAN_PATH = PROJECT_ROOT / "data" / "processed" / "poetry_expansion_seed.jsonl"
DEFAULT_LLM_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_poetry_expansion_chatgpt.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_poetry_expansion.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_poetry_expansion_report.json"


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
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


def quality_passed(row: Dict, allow_low_quality: bool) -> bool:
    if allow_low_quality:
        return True
    quality = row.get("quality")
    if not isinstance(quality, dict):
        return False
    return quality.get("passed_basic_quality_check") is True


def normalize_human(row: Dict) -> Dict:
    item = dict(row)
    item["label"] = 0
    item["domain"] = "poetry"
    item["generator"] = "human"
    item["model"] = "human"
    item["generation"] = "human"
    item["prompt_type"] = item.get("prompt_type") or "human_poetry_expansion"
    item["split"] = "train"
    metadata = dict(item.get("metadata") or {})
    metadata["training_augmentation"] = "poetry_expansion_human"
    item["metadata"] = metadata
    return item


def normalize_llm(row: Dict) -> Dict:
    item = dict(row)
    item["label"] = 1
    item["domain"] = "poetry"
    item["generator"] = item.get("generator") or item.get("assigned_generator") or "chatgpt"
    item["source"] = item.get("source") or item["generator"]
    item["generation"] = item.get("generation") or "llm_rewrite"
    item["split"] = "train"
    metadata = dict(item.get("metadata") or {})
    metadata["training_augmentation"] = "poetry_expansion_llm"
    item["metadata"] = metadata
    return item


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
    parser = argparse.ArgumentParser(description="Append poetry expansion human/LLM samples to train.")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--human_poetry", type=str, default=str(DEFAULT_HUMAN_PATH))
    parser.add_argument("--llm_poetry", type=str, default=str(DEFAULT_LLM_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--allow_low_quality", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = Path(args.train)
    human_path = Path(args.human_poetry)
    llm_path = Path(args.llm_poetry)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find train file: {train_path}")
    if not human_path.exists():
        raise FileNotFoundError(f"Cannot find human poetry file: {human_path}")
    if not llm_path.exists():
        raise FileNotFoundError(f"Cannot find LLM poetry file: {llm_path}")

    train_rows = load_jsonl(train_path)
    human_rows_raw = load_jsonl(human_path)
    llm_rows_raw = load_jsonl(llm_path)

    existing_ids = {row.get("id") for row in train_rows}
    kept_human = []
    kept_llm = []
    skipped = Counter()

    for row in human_rows_raw:
        item = normalize_human(row)
        if not item.get("id") or item.get("id") in existing_ids:
            skipped["human_duplicate_or_missing_id"] += 1
            continue
        if not item.get("pair_id"):
            skipped["human_missing_pair_id"] += 1
            continue
        if not isinstance(item.get("text"), str) or not item.get("text", "").strip():
            skipped["human_empty_text"] += 1
            continue
        existing_ids.add(item.get("id"))
        kept_human.append(item)

    for row in llm_rows_raw:
        item = normalize_llm(row)
        if not item.get("id") or item.get("id") in existing_ids:
            skipped["llm_duplicate_or_missing_id"] += 1
            continue
        if not item.get("pair_id"):
            skipped["llm_missing_pair_id"] += 1
            continue
        if not isinstance(item.get("text"), str) or not item.get("text", "").strip():
            skipped["llm_empty_text"] += 1
            continue
        if not quality_passed(item, allow_low_quality=args.allow_low_quality):
            skipped["llm_quality_not_passed"] += 1
            continue
        existing_ids.add(item.get("id"))
        kept_llm.append(item)

    augmented = train_rows + kept_human + kept_llm
    save_jsonl(augmented, output_path)

    report = {
        "train_input": str(train_path),
        "human_poetry_input": str(human_path),
        "llm_poetry_input": str(llm_path),
        "output": str(output_path),
        "base_train": summarize(train_rows),
        "human_poetry_raw": summarize(human_rows_raw),
        "llm_poetry_raw": summarize(llm_rows_raw),
        "human_poetry_kept": summarize(kept_human),
        "llm_poetry_kept": summarize(kept_llm),
        "skipped": dict(skipped),
        "augmented_train": summarize(augmented),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Augment Train With Poetry Expansion")
    print("=" * 70)
    print(f"Base train rows: {len(train_rows)}")
    print(f"Human poetry kept: {len(kept_human)}")
    print(f"LLM poetry kept: {len(kept_llm)}")
    print(f"Augmented train rows: {len(augmented)}")
    print(f"Skipped: {dict(skipped)}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
