import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))

from build_poetry_seed_from_txt import build_samples, load_metadata, word_count


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "external_human" / "poetry" / "gutenberg_poetry"
DEFAULT_METADATA_PATH = DEFAULT_INPUT_DIR / "metadata.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "poetry_expansion_seed.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "poetry_expansion_seed_report.json"
DEFAULT_EXISTING_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "human_seed_combined.jsonl",
    PROJECT_ROOT / "data" / "processed" / "human_hard_negative_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_valid.jsonl",
    PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_internal_test.jsonl",
]


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


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


def load_existing_keys(paths: List[Path]) -> Set[str]:
    keys = set()
    for path in paths:
        for row in load_jsonl(path):
            text = row.get("text", "")
            if isinstance(text, str) and text.strip():
                keys.add(normalize_key(text))
    return keys


def rewrite_ids(rows: List[Dict], prefix: str) -> List[Dict]:
    output = []
    for idx, row in enumerate(rows, start=1):
        item = dict(row)
        metadata = dict(item.get("metadata") or {})
        metadata["poetry_expansion"] = True
        metadata["original_builder_id"] = item.get("id")
        metadata["original_builder_pair_id"] = item.get("pair_id")
        item["id"] = f"human_{prefix}_{idx:06d}"
        item["pair_id"] = f"pair_{prefix}_{idx:06d}"
        item["prompt_type"] = "human_poetry_expansion"
        item["generation"] = "human"
        item["metadata"] = metadata
        output.append(item)
    return output


def summarize(rows: List[Dict]) -> Dict:
    word_lengths = [word_count(row.get("text", "")) for row in rows]
    line_lengths = [len([line for line in row.get("text", "").splitlines() if line.strip()]) for row in rows]
    return {
        "num_samples": len(rows),
        "source_distribution": dict(Counter(row.get("source", "unknown") for row in rows)),
        "word_count": {
            "min": min(word_lengths) if word_lengths else None,
            "max": max(word_lengths) if word_lengths else None,
            "mean": sum(word_lengths) / len(word_lengths) if word_lengths else None,
        },
        "line_count": {
            "min": min(line_lengths) if line_lengths else None,
            "max": max(line_lengths) if line_lengths else None,
            "mean": sum(line_lengths) / len(line_lengths) if line_lengths else None,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build a deduplicated train-only poetry expansion seed.")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--existing", type=str, nargs="*", default=[str(path) for path in DEFAULT_EXISTING_PATHS])
    parser.add_argument("--target_count", type=int, default=200)
    parser.add_argument("--candidate_count", type=int, default=2000)
    parser.add_argument("--min_words", type=int, default=24)
    parser.add_argument("--max_words", type=int, default=170)
    parser.add_argument("--min_lines", type=int, default=3)
    parser.add_argument("--max_lines", type=int, default=18)
    parser.add_argument("--max_per_source", type=int, default=400)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--id_prefix", type=str, default="poetry_expansion")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    report_path = Path(args.report)
    existing_paths = [Path(path) for path in args.existing]

    if not input_dir.exists():
        raise FileNotFoundError(f"Cannot find input directory: {input_dir}")

    existing_keys = load_existing_keys(existing_paths)
    metadata = load_metadata(metadata_path)
    candidates = build_samples(
        input_dir=input_dir,
        metadata=metadata,
        min_words=args.min_words,
        max_words=args.max_words,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        max_per_source=args.max_per_source,
        max_samples=args.candidate_count,
        seed=args.seed,
    )

    selected = []
    seen = set()
    skipped_existing = 0
    skipped_duplicate = 0
    for row in candidates:
        key = normalize_key(row.get("text", ""))
        if not key:
            continue
        if key in existing_keys:
            skipped_existing += 1
            continue
        if key in seen:
            skipped_duplicate += 1
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= args.target_count:
            break

    if len(selected) < args.target_count:
        raise ValueError(f"Only selected {len(selected)} new poetry samples, requested {args.target_count}")

    output_rows = rewrite_ids(selected, prefix=args.id_prefix)
    save_jsonl(output_rows, output_path)

    report = {
        "input_dir": str(input_dir),
        "metadata": str(metadata_path),
        "output": str(output_path),
        "target_count": args.target_count,
        "candidate_count": args.candidate_count,
        "num_existing_text_keys": len(existing_keys),
        "existing_paths": [str(path) for path in existing_paths],
        "skipped_existing_text": skipped_existing,
        "skipped_duplicate_candidate": skipped_duplicate,
        "selected": summarize(output_rows),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Build Poetry Expansion Seed")
    print("=" * 70)
    print(f"Candidates scanned: {len(candidates)}")
    print(f"Existing text keys: {len(existing_keys)}")
    print(f"Selected new poetry samples: {len(output_rows)}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Source distribution: {report['selected']['source_distribution']}")


if __name__ == "__main__":
    main()
