import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train.jsonl"
DEFAULT_HARD_NEG_PATH = PROJECT_ROOT / "data" / "processed" / "human_hard_negative_seed.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_hardneg.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_hardneg_report.json"


def parse_domain_limits(value: str) -> Dict[str, int]:
    limits = {}
    value = (value or "").strip()
    if not value:
        return limits

    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            domain, count = part.split("=", 1)
        elif ":" in part:
            domain, count = part.split(":", 1)
        else:
            raise ValueError(f"Domain limit must use domain=count format, got: {part}")

        domain = domain.strip()
        if not domain:
            raise ValueError(f"Missing domain in domain limit: {part}")

        try:
            count_value = int(count)
        except ValueError as exc:
            raise ValueError(f"Invalid count in domain limit: {part}") from exc
        if count_value < 0:
            raise ValueError(f"Domain limit must be non-negative, got: {part}")
        limits[domain] = count_value

    return limits


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


def normalize_hard_negative(sample: Dict) -> Dict:
    item = dict(sample)
    item["label"] = 0
    item["generator"] = "human"
    item["model"] = "human"
    item["generation"] = "human"
    item["prompt_type"] = "human_hard_negative"
    item["split"] = "train"
    metadata = dict(item.get("metadata") or {})
    metadata["training_augmentation"] = "hard_negative_human"
    item["metadata"] = metadata
    return item


def apply_domain_limits(rows: List[Dict], limits: Dict[str, int]) -> List[Dict]:
    if not limits:
        return rows

    kept = []
    used = Counter()
    for row in rows:
        domain = row.get("domain", "unknown")
        limit = limits.get(domain)
        if limit is None or used[domain] >= limit:
            continue
        kept.append(row)
        used[domain] += 1
    return kept


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
    parser = argparse.ArgumentParser(description="Append human hard negatives to the train split only.")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--hard_negatives", type=str, default=str(DEFAULT_HARD_NEG_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument(
        "--domain_limits",
        type=str,
        default="",
        help="Optional comma-separated limits such as poetry=150,literature=200,academic=150. "
        "When set, only listed domains are kept up to their limit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_path = Path(args.train)
    hard_neg_path = Path(args.hard_negatives)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find train file: {train_path}")
    if not hard_neg_path.exists():
        raise FileNotFoundError(f"Cannot find hard-negative file: {hard_neg_path}")

    train_rows = load_jsonl(train_path)
    domain_limits = parse_domain_limits(args.domain_limits)
    hard_neg_rows_all = [normalize_hard_negative(row) for row in load_jsonl(hard_neg_path)]
    hard_neg_rows = apply_domain_limits(hard_neg_rows_all, domain_limits)

    existing_ids = {row.get("id") for row in train_rows}
    existing_pairs = {row.get("pair_id") for row in train_rows}

    kept_hard_negatives = []
    skipped_duplicate_id = 0
    skipped_duplicate_pair = 0
    skipped_bad_label = 0

    for row in hard_neg_rows:
        if row.get("label") != 0:
            skipped_bad_label += 1
            continue
        if row.get("id") in existing_ids:
            skipped_duplicate_id += 1
            continue
        if row.get("pair_id") in existing_pairs:
            skipped_duplicate_pair += 1
            continue
        existing_ids.add(row.get("id"))
        existing_pairs.add(row.get("pair_id"))
        kept_hard_negatives.append(row)

    augmented_rows = train_rows + kept_hard_negatives
    save_jsonl(augmented_rows, output_path)

    report = {
        "train_input": str(train_path),
        "hard_negative_input": str(hard_neg_path),
        "domain_limits": domain_limits,
        "output": str(output_path),
        "base_train": summarize(train_rows),
        "hard_negatives_available": summarize(hard_neg_rows_all),
        "hard_negatives_requested": len(hard_neg_rows),
        "hard_negatives_kept": len(kept_hard_negatives),
        "hard_negatives_skipped": {
            "duplicate_id": skipped_duplicate_id,
            "duplicate_pair_id": skipped_duplicate_pair,
            "bad_label": skipped_bad_label,
        },
        "hard_negative_distribution": summarize(kept_hard_negatives),
        "augmented_train": summarize(augmented_rows),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Augment Train With Hard Negatives")
    print("=" * 70)
    print(f"Base train rows: {len(train_rows)}")
    print(f"Hard negatives kept: {len(kept_hard_negatives)}")
    print(f"Augmented train rows: {len(augmented_rows)}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Augmented label distribution: {report['augmented_train']['label_distribution']}")
    print(f"Augmented domain distribution: {report['augmented_train']['domain_distribution']}")


if __name__ == "__main__":
    main()
