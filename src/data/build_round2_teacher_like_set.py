import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


DEFAULT_BASE_TRAIN = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train.jsonl"
DEFAULT_VALID = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_valid.jsonl"
DEFAULT_INTERNAL_TEST = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_internal_test.jsonl"

DEFAULT_HUMAN_SEEDS = [
    PROJECT_ROOT / "data" / "processed" / "human_hard_negative_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "poetry_expansion_seed.jsonl",
]
DEFAULT_LLM_SEEDS = [
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_chatgpt_hard_positive.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_poetry_expansion_chatgpt.jsonl",
]

DEFAULT_HUMAN_OUT = PROJECT_ROOT / "data" / "processed" / "round2_human_hardneg_seed.jsonl"
DEFAULT_LLM_OUT = PROJECT_ROOT / "data" / "processed" / "round2_llm_hardpos_seed.jsonl"
DEFAULT_TRAIN_OUT = PROJECT_ROOT / "data" / "processed" / "round2_teacher_like_train.jsonl"
DEFAULT_DEV_OUT = PROJECT_ROOT / "data" / "processed" / "round2_teacher_like_dev.jsonl"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "data" / "processed" / "round2_teacher_like_report.json"


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_many(paths: Sequence[Path]) -> List[Dict]:
    rows = []
    for path in paths:
        if path.exists():
            for row in load_records(path):
                item = dict(row)
                item["_input_path"] = str(path)
                rows.append(item)
    return rows


def row_key(row: Dict) -> str:
    return " ".join(str(row.get("text", "")).lower().split())


def hard_bucket(row: Dict) -> str:
    text = str(row.get("text", ""))
    bucket = assign_bucket(text)
    label = int(row.get("label", 0))
    prompt_type = str(row.get("prompt_type", ""))
    domain = str(row.get("domain", ""))

    if label == 0:
        if bucket == "poetry_classical":
            return "classical_poetry_human"
        if bucket == "poetry_freeverse":
            return "modern_freeverse_human"
        if bucket == "academic_formal" or domain == "academic":
            return "formal_academic_human"
        if bucket == "literary_short_fragment":
            return "short_fragment_human"
        return "ornate_literary_prose_human"

    if domain == "academic" or "academic" in prompt_type:
        return "chatgpt_natural_academic"
    if domain == "poetry" or "poetry" in prompt_type:
        if "archaic" in prompt_type or bucket == "poetry_classical":
            return "chatgpt_archaic_poetry"
        return "chatgpt_poetry_conservative"
    if "archaic" in prompt_type:
        return "chatgpt_old_fiction"
    return "chatgpt_conservative_literary"


def normalize_candidate(row: Dict, label: int, round2_source: str) -> Dict:
    item = dict(row)
    item["label"] = label
    item["text"] = str(item.get("text", "")).strip()
    item["domain"] = item.get("domain") or "unknown"
    item["generator"] = "human" if label == 0 else item.get("generator") or item.get("source") or "unknown_llm"
    item["source"] = item.get("source") or item["generator"]
    item["model"] = "human" if label == 0 else item.get("model", item["generator"])
    item["generation"] = "human" if label == 0 else item.get("generation", "llm_rewrite")
    item["pair_id"] = item.get("pair_id") or item.get("source_id") or item.get("id")
    item["source_id"] = item.get("source_id") or item.get("id")
    item["round2_source"] = round2_source
    item["round2_tag"] = hard_bucket(item)
    item["subdomain"] = item["round2_tag"]
    item["bucket"] = assign_bucket(item["text"])
    metadata = dict(item.get("metadata") or {})
    metadata["round2_teacher_like"] = True
    metadata["round2_source"] = round2_source
    metadata["round2_bucket"] = item["bucket"]
    item["metadata"] = metadata
    return item


def dedup_candidates(rows: Sequence[Dict], forbidden_ids: set, forbidden_pairs: set, forbidden_texts: set) -> Tuple[List[Dict], Counter]:
    kept = []
    seen_ids = set(forbidden_ids)
    seen_pairs = set(forbidden_pairs)
    seen_texts = set(forbidden_texts)
    skipped = Counter()

    for row in rows:
        if row.get("label") not in [0, 1]:
            skipped["bad_label"] += 1
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            skipped["empty_text"] += 1
            continue
        sample_id = row.get("id")
        pair_id = row.get("pair_id")
        key = row_key(row)
        if sample_id in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        if pair_id in seen_pairs:
            skipped["duplicate_pair_id"] += 1
            continue
        if key in seen_texts:
            skipped["duplicate_or_near_duplicate_text"] += 1
            continue
        seen_ids.add(sample_id)
        seen_pairs.add(pair_id)
        seen_texts.add(key)
        kept.append(row)
    return kept, skipped


def split_by_pair(rows: Sequence[Dict], dev_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("pair_id")].append(row)

    pair_ids = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(pair_ids)
    dev_count = max(1, int(round(len(pair_ids) * dev_ratio))) if pair_ids else 0
    dev_pairs = set(pair_ids[:dev_count])

    train_rows = []
    dev_rows = []
    for pair_id, group in grouped.items():
        target = dev_rows if pair_id in dev_pairs else train_rows
        target.extend(group)
    return train_rows, dev_rows


def summarize(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {
            "num_samples": 0,
            "num_pair_ids": 0,
            "label_distribution": {},
            "domain_distribution": {},
            "round2_tag_distribution": {},
            "bucket_distribution": {},
        }
    lengths = [len(str(row.get("text", "")).split()) for row in rows]
    return {
        "num_samples": len(rows),
        "num_pair_ids": len({row.get("pair_id") for row in rows}),
        "label_distribution": dict(Counter(str(row.get("label")) for row in rows)),
        "domain_distribution": dict(Counter(row.get("domain", "unknown") for row in rows)),
        "generator_distribution": dict(Counter(row.get("generator", "unknown") for row in rows)),
        "round2_tag_distribution": dict(Counter(row.get("round2_tag", "unknown") for row in rows)),
        "bucket_distribution": dict(Counter(row.get("bucket", "unknown") for row in rows)),
        "word_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
    }


def add_split(rows: Sequence[Dict], split: str) -> List[Dict]:
    out = []
    for row in rows:
        item = dict(row)
        item["split"] = split
        out.append(item)
    return out


def acceptance(report: Dict, dev_rows: Sequence[Dict], train_additions: Sequence[Dict]) -> Dict:
    labels = Counter(str(row.get("label")) for row in dev_rows)
    total = len(dev_rows)
    min_class_share = min((count / total for count in labels.values()), default=0.0)
    hard_buckets = set(row.get("round2_tag") for row in dev_rows)
    bucket_counts = Counter(row.get("bucket", "unknown") for row in dev_rows)
    return {
        "hard_buckets_covered": len(hard_buckets),
        "round2_dev_rows": len(dev_rows),
        "round2_train_additions": len(train_additions),
        "min_dev_class_share": min_class_share,
        "has_poetry_representation": bucket_counts["poetry_classical"] + bucket_counts["poetry_freeverse"] > 0,
        "has_academic_representation": bucket_counts["academic_formal"] > 0,
        "meets_plan_minimums": {
            "hard_buckets_at_least_5": len(hard_buckets) >= 5,
            "dev_rows_at_least_800": len(dev_rows) >= 800,
            "train_additions_at_least_2500": len(train_additions) >= 2500,
            "no_dev_class_below_40_percent": min_class_share >= 0.40,
        },
        "notes": report.get("notes", []),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build the round2 teacher-like train/dev data artifacts.")
    parser.add_argument("--base_train", default=str(DEFAULT_BASE_TRAIN))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--internal_test", default=str(DEFAULT_INTERNAL_TEST))
    parser.add_argument("--human_seeds", nargs="+", default=[str(path) for path in DEFAULT_HUMAN_SEEDS])
    parser.add_argument("--llm_seeds", nargs="+", default=[str(path) for path in DEFAULT_LLM_SEEDS])
    parser.add_argument("--human_output", default=str(DEFAULT_HUMAN_OUT))
    parser.add_argument("--llm_output", default=str(DEFAULT_LLM_OUT))
    parser.add_argument("--train_output", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--dev_output", default=str(DEFAULT_DEV_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_OUT))
    parser.add_argument("--dev_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    base_train = load_records(Path(args.base_train))
    valid = load_records(Path(args.valid)) if Path(args.valid).exists() else []
    internal_test = load_records(Path(args.internal_test)) if Path(args.internal_test).exists() else []
    original_rows = base_train + valid + internal_test
    forbidden_ids = {row.get("id") for row in original_rows}
    forbidden_pairs = {row.get("pair_id") for row in original_rows}
    forbidden_texts = {row_key(row) for row in original_rows if row.get("text")}

    human_raw = load_many([Path(path) for path in args.human_seeds])
    llm_raw = load_many([Path(path) for path in args.llm_seeds])
    human_candidates = [
        normalize_candidate(row, label=0, round2_source=Path(str(row.get("_input_path", ""))).name or "human_seed")
        for row in human_raw
    ]
    llm_candidates = [
        normalize_candidate(row, label=1, round2_source=Path(str(row.get("_input_path", ""))).name or "llm_seed")
        for row in llm_raw
    ]

    human_candidates, human_skipped = dedup_candidates(human_candidates, forbidden_ids, forbidden_pairs, forbidden_texts)
    llm_candidates, llm_skipped = dedup_candidates(llm_candidates, forbidden_ids, forbidden_pairs, forbidden_texts)

    combined_candidates = human_candidates + llm_candidates
    combined_train, combined_dev = split_by_pair(combined_candidates, dev_ratio=args.dev_ratio, seed=args.seed)
    train_additions = add_split(combined_train, "train")
    dev_rows = add_split(combined_dev, "round2_dev")
    round2_train = add_split(base_train, "train") + train_additions

    save_jsonl(human_candidates, Path(args.human_output))
    save_jsonl(llm_candidates, Path(args.llm_output))
    save_jsonl(round2_train, Path(args.train_output))
    save_jsonl(dev_rows, Path(args.dev_output))

    notes = []
    if len(llm_candidates) < 1500:
        notes.append("Local LLM hard-positive seed is below the Phase 1 target; prepare/generate more conservative rewrites before final neural retraining.")
    if len(dev_rows) < 800:
        notes.append("Round2 dev is below the suggested 800-row target because strict dedup leaves limited held-out hard-positive rows.")

    report = {
        "base_train": str(Path(args.base_train)),
        "valid_for_leakage_check": str(Path(args.valid)),
        "internal_test_for_leakage_check": str(Path(args.internal_test)),
        "human_seed_inputs": args.human_seeds,
        "llm_seed_inputs": args.llm_seeds,
        "outputs": {
            "human_seed": args.human_output,
            "llm_seed": args.llm_output,
            "train": args.train_output,
            "dev": args.dev_output,
        },
        "dev_ratio": args.dev_ratio,
        "seed": args.seed,
        "human_candidates": summarize(human_candidates),
        "llm_candidates": summarize(llm_candidates),
        "train_additions": summarize(train_additions),
        "round2_train": summarize(round2_train),
        "round2_dev": summarize(dev_rows),
        "dedup_skipped": {
            "human": dict(human_skipped),
            "llm": dict(llm_skipped),
        },
        "notes": notes,
    }
    report["acceptance"] = acceptance(report, dev_rows=dev_rows, train_additions=train_additions)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Round2 teacher-like data built")
    print("=" * 70)
    print(f"Human hard-negative seed rows: {len(human_candidates)}")
    print(f"LLM hard-positive seed rows: {len(llm_candidates)}")
    print(f"Round2 train rows: {len(round2_train)}")
    print(f"Round2 dev rows: {len(dev_rows)}")
    print(f"Report: {report_path}")
    if notes:
        print("Notes:")
        for note in notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
