import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUTS = [
    PROJECT_ROOT / "data" / "processed" / "human_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "academic_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "poetry_seed.jsonl",
]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "human_seed_combined.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "human_seed_combined_report.json"


def load_jsonl(path: Path) -> List[Dict]:
    samples = []

    if not path.exists():
        print(f"[Warning] Missing input file, skipped: {path}")
        return samples

    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse {path}, line {line_id}: {e}")

    return samples


def save_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def is_valid_human_sample(sample: Dict) -> bool:
    if sample.get("label") != 0:
        return False
    if not isinstance(sample.get("text"), str) or not sample["text"].strip():
        return False
    if not sample.get("id"):
        return False
    if not sample.get("pair_id"):
        return False
    return True


def normalize_human_sample(sample: Dict) -> Dict:
    item = dict(sample)
    item["label"] = 0
    item["generation"] = "human"
    item.setdefault("domain", "unknown")
    item.setdefault("source", "unknown")
    return item


def build_report(samples: List[Dict], invalid_count: int, duplicate_id_count: int, duplicate_pair_count: int) -> Dict:
    word_counts = [len(sample.get("text", "").split()) for sample in samples]

    return {
        "total_samples": len(samples),
        "invalid_samples_filtered": invalid_count,
        "duplicate_id_count": duplicate_id_count,
        "duplicate_pair_id_count": duplicate_pair_count,
        "domain_distribution": dict(Counter(sample.get("domain", "unknown") for sample in samples)),
        "source_distribution_top50": dict(Counter(sample.get("source", "unknown") for sample in samples).most_common(50)),
        "word_count": {
            "min": min(word_counts) if word_counts else None,
            "max": max(word_counts) if word_counts else None,
            "mean": sum(word_counts) / len(word_counts) if word_counts else None,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple human seed JSONL files.")

    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUTS],
        help="Input human seed JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output combined human seed JSONL.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help="Output report JSON.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_paths = [Path(path) for path in args.inputs]
    output_path = Path(args.output)
    report_path = Path(args.report)

    seen_ids = set()
    seen_pair_ids = set()
    duplicate_id_count = 0
    duplicate_pair_count = 0
    invalid_count = 0
    merged = []

    print("=" * 70)
    print("Merge Human Seeds")
    print("=" * 70)

    for path in input_paths:
        raw_samples = load_jsonl(path)
        valid_count = 0

        for sample in raw_samples:
            if not is_valid_human_sample(sample):
                invalid_count += 1
                continue

            sample_id = sample.get("id")
            pair_id = sample.get("pair_id")

            if sample_id in seen_ids:
                duplicate_id_count += 1
                continue
            if pair_id in seen_pair_ids:
                duplicate_pair_count += 1
                continue

            seen_ids.add(sample_id)
            seen_pair_ids.add(pair_id)
            merged.append(normalize_human_sample(sample))
            valid_count += 1

        print(f"{path}: raw={len(raw_samples)} kept={valid_count}")

    save_jsonl(merged, output_path)

    report = build_report(
        samples=merged,
        invalid_count=invalid_count,
        duplicate_id_count=duplicate_id_count,
        duplicate_pair_count=duplicate_pair_count,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 70)
    print(f"Saved combined human seed: {output_path}")
    print(f"Saved report: {report_path}")
    print(f"Total samples: {report['total_samples']}")
    print(f"Domain distribution: {report['domain_distribution']}")
    print(f"Invalid filtered: {invalid_count}")
    print(f"Duplicate ids skipped: {duplicate_id_count}")
    print(f"Duplicate pair_ids skipped: {duplicate_pair_count}")


if __name__ == "__main__":
    main()
