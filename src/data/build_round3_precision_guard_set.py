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
DEFAULT_ROUND2_TRAIN = PROJECT_ROOT / "data" / "processed" / "round2_teacher_like_train.jsonl"
DEFAULT_ROUND2_DEV = PROJECT_ROOT / "data" / "processed" / "round2_teacher_like_dev.jsonl"

DEFAULT_HUMAN_INPUTS = [
    PROJECT_ROOT / "data" / "processed" / "round2_human_hardneg_source.jsonl",
    PROJECT_ROOT / "data" / "processed" / "human_hard_negative_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "poetry_expansion_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "poetry_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "academic_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "human_seed_combined.jsonl",
]
DEFAULT_LLM_INPUTS = [
    PROJECT_ROOT / "data" / "processed" / "round2_llm_hardpos_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_round2_chatgpt_hard_positive.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_chatgpt_hard_positive.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_poetry_expansion_chatgpt.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_academic_chatgpt.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_academic_deepseek.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_academic_doubao.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_academic_gemini.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_poetry_chatgpt.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_poetry_deepseek.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_poetry_doubao.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_poetry_gemini.jsonl",
]

DEFAULT_HUMAN_OUT = PROJECT_ROOT / "data" / "processed" / "round3_hard_negative_mirror_source.jsonl"
DEFAULT_LLM_OUT = PROJECT_ROOT / "data" / "processed" / "round3_llm_hardpos_multi_generator_seed.jsonl"
DEFAULT_TRAIN_OUT = PROJECT_ROOT / "data" / "processed" / "round3_precision_guard_train.jsonl"
DEFAULT_DEV_OUT = PROJECT_ROOT / "data" / "processed" / "round3_precision_guard_dev.jsonl"
DEFAULT_SPOTCHECK_OUT = PROJECT_ROOT / "data" / "processed" / "round3_precision_guard_spotcheck.jsonl"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "data" / "processed" / "round3_precision_guard_report.json"


HIGH_RISK_HUMAN_BUCKETS = {
    "poetry_classical",
    "poetry_freeverse",
    "literary_old_prose",
    "literary_short_fragment",
    "academic_formal",
    "general_prose",
}


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_many(paths: Sequence[Path]) -> List[Dict]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        for row in load_records(path):
            item = dict(row)
            item["_input_path"] = str(path)
            rows.append(item)
    return rows


def row_key(row: Dict) -> str:
    return " ".join(str(row.get("text", "")).lower().split())


def safe_id(value, fallback: str) -> str:
    value = str(value or "").strip()
    return value if value else fallback


def source_stage(path_name: str) -> str:
    if "round2" in path_name:
        return "round2_reused_pool"
    if "poetry" in path_name:
        return "round3_poetry_mirror_pool"
    if "academic" in path_name:
        return "round3_academic_mirror_pool"
    if "hard_negative" in path_name:
        return "round3_hard_negative_pool"
    return "round3_local_pool"


def human_tag(bucket: str, row: Dict) -> str:
    if bucket == "poetry_classical":
        return "human_poetry_classical_mirror"
    if bucket == "poetry_freeverse":
        return "human_poetry_freeverse_mirror"
    if bucket == "literary_old_prose":
        return "human_literary_old_prose_mirror"
    if bucket == "academic_formal" or row.get("domain") == "academic":
        return "human_formal_academic_mirror"
    if bucket == "literary_short_fragment":
        return "human_short_fragment_mirror"
    return "human_ornate_literary_mirror"


def llm_tag(bucket: str, row: Dict) -> str:
    prompt_type = str(row.get("prompt_type", "")).lower()
    domain = str(row.get("domain", "")).lower()
    if bucket == "poetry_classical" or "archaic" in prompt_type:
        return "llm_archaic_poetry_hardpos"
    if bucket == "poetry_freeverse" or domain == "poetry":
        return "llm_poetry_preserving_hardpos"
    if bucket == "academic_formal" or domain == "academic":
        return "llm_natural_academic_hardpos"
    if bucket == "literary_old_prose" or "old" in prompt_type:
        return "llm_old_fiction_style_hardpos"
    if bucket == "literary_short_fragment":
        return "llm_short_fragment_hardpos"
    return "llm_conservative_literary_hardpos"


def sample_weight(label: int, bucket: str, tag: str) -> float:
    if label == 1:
        return 1.0
    if bucket in {"poetry_classical", "poetry_freeverse"}:
        return 2.0
    if bucket in {"literary_old_prose", "literary_short_fragment"}:
        return 1.7
    if bucket == "academic_formal":
        return 1.4
    if tag == "human_ornate_literary_mirror":
        return 1.5
    return 1.2


def normalize_candidate(row: Dict, label: int, index: int) -> Dict:
    text = str(row.get("text", "")).strip()
    bucket = assign_bucket(text)
    path_name = Path(str(row.get("_input_path", ""))).name
    tag = human_tag(bucket, row) if label == 0 else llm_tag(bucket, row)
    features = text_features(text)

    item = dict(row)
    item.pop("_input_path", None)
    item["id"] = safe_id(item.get("id"), f"round3_{label}_{index:06d}")
    item["text"] = text
    item["label"] = label
    item["domain"] = item.get("domain") or ("academic" if bucket == "academic_formal" else "literature")
    item["generator"] = "human" if label == 0 else item.get("generator") or item.get("source") or "unknown_llm"
    item["source"] = item.get("source") or item["generator"]
    item["model"] = "human" if label == 0 else item.get("model", item["generator"])
    item["generation"] = "human" if label == 0 else item.get("generation", "llm_rewrite")
    item["pair_id"] = safe_id(item.get("pair_id") or item.get("source_id"), f"round3_pair_{label}_{index:06d}")
    item["source_id"] = item.get("source_id") or item["id"]
    item["bucket"] = bucket
    item["round3_tag"] = tag
    item["subdomain"] = tag
    item["round3_source"] = path_name
    item["round3_source_stage"] = source_stage(path_name)
    item["sample_weight"] = sample_weight(label, bucket, tag)

    metadata = dict(item.get("metadata") or {})
    metadata["round3_precision_guard"] = True
    metadata["round3_source"] = path_name
    metadata["round3_bucket"] = bucket
    item["metadata"] = metadata

    for key, value in features.items():
        if key != "bucket":
            item.setdefault(key, value)

    return item


def forbidden_sets(rows: Sequence[Dict]) -> Tuple[set, set, set]:
    ids = {row.get("id") for row in rows if row.get("id") is not None}
    pairs = {row.get("pair_id") for row in rows if row.get("pair_id") is not None}
    texts = {row_key(row) for row in rows if row.get("text")}
    return ids, pairs, texts


def dedup_candidates(
    rows: Sequence[Dict],
    forbidden_ids: set,
    forbidden_pairs: set,
    forbidden_texts: set,
    allow_forbidden_pairs: set,
) -> Tuple[List[Dict], Counter]:
    kept = []
    seen_ids = set()
    seen_texts = set()
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
        if sample_id in forbidden_ids:
            skipped["forbidden_id"] += 1
            continue
        if pair_id in forbidden_pairs and pair_id not in allow_forbidden_pairs:
            skipped["forbidden_pair"] += 1
            continue
        if key in forbidden_texts:
            skipped["forbidden_text"] += 1
            continue
        if sample_id in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        if key in seen_texts:
            skipped["duplicate_or_near_duplicate_text"] += 1
            continue
        seen_ids.add(sample_id)
        seen_texts.add(key)
        kept.append(row)
    return kept, skipped


def group_by_pair(rows: Sequence[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("pair_id"))].append(row)
    return grouped


def split_guard_candidates(
    candidates: Sequence[Dict],
    base_train_pairs: set,
    round2_train_texts: set,
    dev_per_class: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    forced_train = []
    eligible = []
    for row in candidates:
        pair_id = row.get("pair_id")
        if pair_id in base_train_pairs or row_key(row) in round2_train_texts:
            forced_train.append(row)
        else:
            eligible.append(row)

    grouped = group_by_pair(eligible)
    pair_blocks = list(grouped.values())
    rng = random.Random(seed)
    rng.shuffle(pair_blocks)

    dev_rows = []
    train_rows = []
    dev_label_counts = Counter()

    # First pass: build a balanced, label-constrained precision dev set.
    for group in pair_blocks:
        group_label_counts = Counter(str(int(row["label"])) for row in group)
        fits_dev = all(
            dev_label_counts[label] + count <= dev_per_class
            for label, count in group_label_counts.items()
        )
        if fits_dev:
            dev_rows.extend(group)
            dev_label_counts.update(group_label_counts)
        else:
            train_rows.extend(group)

    return forced_train + train_rows, dev_rows


def balance_dev_rows(rows: Sequence[Dict], seed: int) -> Tuple[List[Dict], List[Dict]]:
    by_label = defaultdict(list)
    for row in rows:
        by_label[str(int(row["label"]))].append(row)
    if len(by_label) < 2:
        return list(rows), []

    target = min(len(items) for items in by_label.values())
    rng = random.Random(seed)
    balanced = []
    moved_to_train = []

    for label, items in by_label.items():
        items = list(items)
        if len(items) <= target:
            balanced.extend(items)
            continue

        # Preserve at least one majority-class example from each observed bucket
        # before random downsampling, so the guard set remains diagnostically
        # useful across high-risk text styles.
        by_bucket = defaultdict(list)
        for row in items:
            by_bucket[str(row.get("bucket", "unknown"))].append(row)
        selected_ids = set()
        selected = []
        for bucket_rows in by_bucket.values():
            rng.shuffle(bucket_rows)
            row = bucket_rows[0]
            selected.append(row)
            selected_ids.add(id(row))
        remaining = [row for row in items if id(row) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, target - len(selected))])
        selected_ids = {id(row) for row in selected}
        balanced.extend(selected)
        moved_to_train.extend(row for row in items if id(row) not in selected_ids)

    rng.shuffle(balanced)
    return balanced, moved_to_train


def enrich_existing_train(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        item = dict(row)
        text = str(item.get("text", "")).strip()
        if not text or item.get("label") not in [0, 1]:
            continue
        bucket = item.get("bucket") or assign_bucket(text)
        label = int(item["label"])
        tag = item.get("round3_tag") or item.get("round2_tag") or ("base_human" if label == 0 else "base_llm")
        item["bucket"] = bucket
        item.setdefault("round3_tag", tag)
        item.setdefault("round3_source_stage", "round2_train_or_base")
        item.setdefault("sample_weight", sample_weight(label, bucket, str(tag)))
        out.append(item)
    return out


def with_split(rows: Sequence[Dict], split: str) -> List[Dict]:
    out = []
    for row in rows:
        item = dict(row)
        item["split"] = split
        out.append(item)
    return out


def summarize(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {
            "num_samples": 0,
            "num_pair_ids": 0,
            "label_distribution": {},
            "bucket_distribution": {},
            "round3_tag_distribution": {},
        }
    lengths = [len(str(row.get("text", "")).split()) for row in rows]
    return {
        "num_samples": len(rows),
        "num_pair_ids": len({row.get("pair_id") for row in rows}),
        "label_distribution": dict(Counter(str(row.get("label")) for row in rows)),
        "domain_distribution": dict(Counter(str(row.get("domain", "unknown")) for row in rows)),
        "generator_distribution": dict(Counter(str(row.get("generator", "unknown")) for row in rows)),
        "bucket_distribution": dict(Counter(str(row.get("bucket", "unknown")) for row in rows)),
        "round3_tag_distribution": dict(Counter(str(row.get("round3_tag", "unknown")) for row in rows)),
        "source_stage_distribution": dict(Counter(str(row.get("round3_source_stage", "unknown")) for row in rows)),
        "sample_weight_distribution": dict(Counter(str(row.get("sample_weight", "1.0")) for row in rows)),
        "word_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
    }


def paired_bucket_coverage(rows: Sequence[Dict]) -> Dict:
    coverage = {}
    for bucket in sorted({row.get("bucket") for row in rows}):
        bucket_rows = [row for row in rows if row.get("bucket") == bucket]
        labels = Counter(str(row.get("label")) for row in bucket_rows)
        coverage[str(bucket)] = {
            "human": labels.get("0", 0),
            "llm": labels.get("1", 0),
            "has_both_labels": labels.get("0", 0) > 0 and labels.get("1", 0) > 0,
        }
    return coverage


def acceptance(human_rows: Sequence[Dict], llm_rows: Sequence[Dict], dev_rows: Sequence[Dict]) -> Dict:
    dev_labels = Counter(str(row.get("label")) for row in dev_rows)
    dev_total = len(dev_rows)
    min_class_share = min((count / dev_total for count in dev_labels.values()), default=0.0)
    coverage = paired_bucket_coverage(dev_rows)
    high_risk_with_both = {
        bucket: block["has_both_labels"]
        for bucket, block in coverage.items()
        if bucket in HIGH_RISK_HUMAN_BUCKETS
    }
    return {
        "hard_human_negatives": len(human_rows),
        "hard_llm_positives": len(llm_rows),
        "round3_dev_rows": len(dev_rows),
        "round3_dev_min_class_share": min_class_share,
        "bucket_label_coverage": coverage,
        "high_risk_buckets_with_both_labels": high_risk_with_both,
        "meets_plan_minimums": {
            "hard_human_negatives_at_least_1800": len(human_rows) >= 1800,
            "hard_llm_positives_at_least_1500_if_possible": len(llm_rows) >= 1500,
            "dev_min_class_share_at_least_45_percent": min_class_share >= 0.45,
            "all_observed_high_risk_dev_buckets_have_both_labels": all(high_risk_with_both.values()),
        },
    }


def spotcheck(rows: Sequence[Dict], n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    selected = rows[: min(n, len(rows))]
    out = []
    for row in selected:
        item = {
            "id": row.get("id"),
            "label": row.get("label"),
            "bucket": row.get("bucket"),
            "round3_tag": row.get("round3_tag"),
            "generator": row.get("generator"),
            "source": row.get("source"),
            "pair_id": row.get("pair_id"),
            "text": row.get("text"),
            "manual_check": {
                "quality_ok": None,
                "prompt_leakage": None,
                "teacher_near_duplicate": None,
                "notes": "",
            },
        }
        out.append(item)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Build Round3 precision-guard hard-negative mirror data.")
    parser.add_argument("--base_train", default=str(DEFAULT_BASE_TRAIN))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--internal_test", default=str(DEFAULT_INTERNAL_TEST))
    parser.add_argument("--round2_train", default=str(DEFAULT_ROUND2_TRAIN))
    parser.add_argument("--round2_dev", default=str(DEFAULT_ROUND2_DEV))
    parser.add_argument("--human_inputs", nargs="+", default=[str(path) for path in DEFAULT_HUMAN_INPUTS])
    parser.add_argument("--llm_inputs", nargs="+", default=[str(path) for path in DEFAULT_LLM_INPUTS])
    parser.add_argument("--human_output", default=str(DEFAULT_HUMAN_OUT))
    parser.add_argument("--llm_output", default=str(DEFAULT_LLM_OUT))
    parser.add_argument("--train_output", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--dev_output", default=str(DEFAULT_DEV_OUT))
    parser.add_argument("--spotcheck_output", default=str(DEFAULT_SPOTCHECK_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_OUT))
    parser.add_argument("--dev_per_class", type=int, default=520)
    parser.add_argument("--spotcheck_n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def main():
    args = parse_args()
    base_train = load_records(Path(args.base_train))
    valid = load_records(Path(args.valid)) if Path(args.valid).exists() else []
    internal_test = load_records(Path(args.internal_test)) if Path(args.internal_test).exists() else []
    round2_train = load_records(Path(args.round2_train)) if Path(args.round2_train).exists() else base_train
    round2_dev = load_records(Path(args.round2_dev)) if Path(args.round2_dev).exists() else []

    base_train_ids, base_train_pairs, _base_train_texts = forbidden_sets(base_train)
    valid_ids, valid_pairs, valid_texts = forbidden_sets(valid)
    internal_ids, internal_pairs, internal_texts = forbidden_sets(internal_test)
    round2_train_ids, _round2_train_pairs, round2_train_texts = forbidden_sets(round2_train)

    forbidden_ids = valid_ids | internal_ids
    forbidden_pairs = valid_pairs | internal_pairs
    forbidden_texts = valid_texts | internal_texts
    allow_forbidden_pairs = set(base_train_pairs)

    raw_human = load_many([Path(path) for path in args.human_inputs])
    raw_llm = load_many([Path(path) for path in args.llm_inputs])

    human_candidates = [normalize_candidate(row, label=0, index=i) for i, row in enumerate(raw_human)]
    llm_candidates = [normalize_candidate(row, label=1, index=i) for i, row in enumerate(raw_llm)]

    human_candidates, human_skipped = dedup_candidates(
        human_candidates,
        forbidden_ids=forbidden_ids,
        forbidden_pairs=forbidden_pairs,
        forbidden_texts=forbidden_texts,
        allow_forbidden_pairs=allow_forbidden_pairs,
    )
    llm_candidates, llm_skipped = dedup_candidates(
        llm_candidates,
        forbidden_ids=forbidden_ids,
        forbidden_pairs=forbidden_pairs,
        forbidden_texts=forbidden_texts,
        allow_forbidden_pairs=allow_forbidden_pairs,
    )

    guard_train_additions, guard_dev = split_guard_candidates(
        candidates=human_candidates + llm_candidates,
        base_train_pairs=base_train_pairs,
        round2_train_texts=round2_train_texts,
        dev_per_class=args.dev_per_class,
        seed=args.seed,
    )
    guard_dev, dev_moved_to_train = balance_dev_rows(guard_dev, seed=args.seed)
    guard_train_additions = guard_train_additions + dev_moved_to_train

    # Do not add exact duplicates already present in the Round2 training file.
    deduped_train_additions = []
    seen_train_texts = set(round2_train_texts)
    for row in guard_train_additions:
        key = row_key(row)
        if key in seen_train_texts or row.get("id") in round2_train_ids:
            continue
        seen_train_texts.add(key)
        deduped_train_additions.append(row)

    # Keep precision dev disjoint from the final Round3 training output.
    train_texts = set(seen_train_texts)
    clean_guard_dev = [row for row in guard_dev if row_key(row) not in train_texts]

    round3_train = with_split(enrich_existing_train(round2_train), "train") + with_split(deduped_train_additions, "train")
    round3_dev = with_split(clean_guard_dev, "round3_precision_guard_dev")
    spotcheck_rows = spotcheck(round3_dev, n=args.spotcheck_n, seed=args.seed)

    save_jsonl(human_candidates, Path(args.human_output))
    save_jsonl(llm_candidates, Path(args.llm_output))
    save_jsonl(round3_train, Path(args.train_output))
    save_jsonl(round3_dev, Path(args.dev_output))
    save_jsonl(spotcheck_rows, Path(args.spotcheck_output))

    notes = []
    if len(llm_candidates) < 1500:
        notes.append("Local LLM hard-positive pool is below the 1500-row target; generate more multi-model rewrites before final neural retraining.")
    old_prose_human = sum(1 for row in human_candidates if row.get("bucket") == "literary_old_prose")
    if old_prose_human < 400:
        notes.append("Old-prose human mirror coverage is below the ideal 400-600 target; current local sources are limited for this bucket.")
    if any(not block["has_both_labels"] for block in paired_bucket_coverage(round3_dev).values()):
        notes.append("Some observed dev buckets do not have both labels; use this report before interpreting per-bucket FP/FN.")

    report = {
        "inputs": {
            "base_train": args.base_train,
            "valid": args.valid,
            "internal_test": args.internal_test,
            "round2_train": args.round2_train,
            "round2_dev": args.round2_dev,
            "human_inputs": args.human_inputs,
            "llm_inputs": args.llm_inputs,
        },
        "outputs": {
            "human_source": args.human_output,
            "llm_source": args.llm_output,
            "train": args.train_output,
            "dev": args.dev_output,
            "spotcheck": args.spotcheck_output,
            "report": args.report,
        },
        "seed": args.seed,
        "dev_per_class": args.dev_per_class,
        "human_candidates": summarize(human_candidates),
        "llm_candidates": summarize(llm_candidates),
        "guard_train_additions": summarize(deduped_train_additions),
        "round3_train": summarize(round3_train),
        "round3_dev": summarize(round3_dev),
        "round2_dev_reference": summarize(round2_dev),
        "dedup_skipped": {
            "human": dict(human_skipped),
            "llm": dict(llm_skipped),
        },
        "acceptance": acceptance(human_candidates, llm_candidates, round3_dev),
        "notes": notes,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Round3 precision-guard data built")
    print("=" * 70)
    print(f"Human hard-negative mirror rows: {len(human_candidates)}")
    print(f"LLM hard-positive rows: {len(llm_candidates)}")
    print(f"Round3 train rows: {len(round3_train)}")
    print(f"Round3 precision-dev rows: {len(round3_dev)}")
    print(f"Spotcheck rows: {len(spotcheck_rows)}")
    print(f"Report: {report_path}")
    if notes:
        print("Notes:")
        for note in notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
