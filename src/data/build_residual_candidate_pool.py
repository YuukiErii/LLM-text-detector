import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


DEFAULT_HUMAN_INPUTS = [
    PROJECT_ROOT / "data" / "processed" / "round4_hard_human_mirror_seed.jsonl",
]
DEFAULT_LLM_INPUTS = [
    PROJECT_ROOT / "data" / "processed" / "round4_hard_llm_positive_seed.jsonl",
]
DEFAULT_VALID = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_valid.jsonl"
DEFAULT_INTERNAL_TEST = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_internal_test.jsonl"
DEFAULT_TEACHER_TEST = PROJECT_ROOT / "data" / "raw" / "teacher_test.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "residual_candidate_pool_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "residual_candidate_pool_v1_report.json"


PROMPT_LEAKAGE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bas an ai\b",
        r"\bi cannot\b",
        r"\bi can't\b",
        r"\bsure[,! ]+here",
        r"\bhere is (the )?(rewrite|paraphrase)",
        r"\brewritten (version|text)",
        r"\breturn only\b",
    ]
]

TAXONOMY_BUCKETS = [
    "human_free_verse",
    "human_classical_poetry",
    "human_archaic_prose",
    "human_polished_literary_prose",
    "human_formal_academic",
    "human_structured_explanatory",
    "human_literary_short_fragment",
    "human_old_fiction_dialogue",
    "llm_conservative_chatgpt_paraphrase",
    "llm_old_fiction_style_rewrite",
    "llm_archaic_poetry_rewrite",
    "llm_academic_term_preserving_paraphrase",
    "llm_low_temperature_minimal_rewrite",
    "llm_style_preserving_dialogue_rewrite",
    "llm_high_jaccard_rewrite",
    "llm_human_like_formal_paraphrase",
]


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_many(paths: Sequence[Path]) -> Tuple[List[Dict], Dict[str, int]]:
    rows = []
    counts = {}
    for path in paths:
        if not path.exists():
            counts[str(path)] = 0
            continue
        loaded = load_records(path)
        counts[str(path)] = len(loaded)
        for row in loaded:
            item = dict(row)
            item["_source_file"] = path.name
            item["_source_path"] = str(path)
            rows.append(item)
    return rows, counts


def text_key(row: Dict) -> str:
    return " ".join(str(row.get("text") or "").lower().split())


def word_count(text: str) -> int:
    return len(str(text or "").split())


def has_prompt_leakage(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in PROMPT_LEAKAGE_PATTERNS)


def quality_jaccard(row: Dict) -> float:
    quality = row.get("quality")
    if isinstance(quality, dict):
        value = quality.get("lexical_jaccard")
        if isinstance(value, (int, float)):
            return float(value)
    return -1.0


def quality_flags(row: Dict, label: int, min_words: int, max_words: int, allow_short_poetry_words: int) -> Tuple[bool, Dict]:
    text = str(row.get("text") or "").strip()
    features = text_features(text) if text else {}
    words = word_count(text)
    bucket = str(row.get("bucket") or row.get("round4_bucket") or features.get("bucket") or "unknown")
    is_poetry = bucket.startswith("poetry") or str(row.get("domain") or "").lower() == "poetry"

    flags = {
        "empty_text": not bool(text),
        "too_short": words < min_words and not (is_poetry and words >= allow_short_poetry_words),
        "too_long": words > max_words,
        "prompt_leakage": label == 1 and has_prompt_leakage(text),
        "failed_basic_quality": False,
        "finish_reason_length": False,
    }

    quality = row.get("quality")
    if isinstance(quality, dict):
        flags["failed_basic_quality"] = quality.get("passed_basic_quality_check") is False
    flags["finish_reason_length"] = row.get("finish_reason") == "length"
    keep = not any(flags.values())
    return keep, flags


def first_value(row: Dict, keys: Sequence[str], default: str = "") -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in keys:
        value = row.get(key)
        if value not in [None, ""]:
            return str(value)
        value = metadata.get(key)
        if value not in [None, ""]:
            return str(value)
    return default


def source_doc_id(row: Dict) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ["paper_id", "gutenberg_id", "source_file", "title"]:
        value = metadata.get(key)
        if value not in [None, ""]:
            return str(value)
    for key in ["source_pair_id", "pair_id", "source_id", "original_id", "id"]:
        value = row.get(key)
        if value not in [None, ""]:
            return str(value)
    return "unknown_source_doc"


def split_group(row: Dict) -> str:
    for key in ["source_pair_id", "pair_id", "source_id", "original_id", "source_doc_id", "id"]:
        value = row.get(key)
        if value not in [None, ""]:
            return str(value)
    return text_key(row)


def text_signal(row: Dict) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    values = [
        row.get("id"),
        row.get("domain"),
        row.get("source"),
        row.get("generator"),
        row.get("prompt_type"),
        row.get("generation"),
        row.get("bucket"),
        row.get("round4_bucket"),
        row.get("round4_tag"),
        row.get("subdomain"),
        metadata.get("title"),
        metadata.get("section"),
        metadata.get("hard_negative_reason"),
    ]
    return " ".join(str(value or "").lower() for value in values)


def human_taxonomy_bucket(row: Dict, features: Dict) -> str:
    bucket = str(row.get("round4_bucket") or row.get("bucket") or features.get("bucket") or assign_bucket(row.get("text", "")))
    signal = text_signal(row)
    text = str(row.get("text") or "")

    if bucket == "poetry_classical":
        return "human_classical_poetry"
    if bucket == "poetry_freeverse":
        return "human_free_verse"
    if "dialogue" in signal or text.count('"') + text.count("“") + text.count("”") >= 4:
        return "human_old_fiction_dialogue" if bucket == "literary_old_prose" else "human_polished_literary_prose"
    if bucket == "literary_old_prose":
        return "human_archaic_prose"
    if bucket == "academic_formal" or str(row.get("domain") or "").lower() == "academic":
        return "human_formal_academic"
    if bucket == "literary_short_fragment":
        return "human_literary_short_fragment"
    if features.get("academic_marker_count", 0.0) >= 2 and word_count(text) >= 80:
        return "human_structured_explanatory"
    return "human_polished_literary_prose"


def llm_taxonomy_bucket(row: Dict, features: Dict) -> str:
    bucket = str(row.get("round4_bucket") or row.get("bucket") or features.get("bucket") or assign_bucket(row.get("text", "")))
    generator = str(row.get("generator") or row.get("source") or "").lower()
    prompt = str(row.get("prompt_type") or "").lower()
    signal = text_signal(row)
    jaccard = quality_jaccard(row)
    text = str(row.get("text") or "")

    if bucket == "academic_formal" or str(row.get("domain") or "").lower() == "academic":
        return "llm_academic_term_preserving_paraphrase"
    if bucket in {"poetry_classical", "poetry_freeverse"} or str(row.get("domain") or "").lower() == "poetry":
        return "llm_archaic_poetry_rewrite"
    if "old" in prompt or "archaic" in prompt or bucket == "literary_old_prose" or "old_prose" in signal:
        return "llm_old_fiction_style_rewrite"
    if "dialogue" in prompt or text.count('"') + text.count("“") + text.count("”") >= 4:
        return "llm_style_preserving_dialogue_rewrite"
    if jaccard >= 0.68:
        return "llm_high_jaccard_rewrite"
    if "minimal" in prompt or "conservative" in prompt or "preserving" in prompt:
        return "llm_low_temperature_minimal_rewrite"
    if generator == "chatgpt":
        return "llm_conservative_chatgpt_paraphrase"
    return "llm_human_like_formal_paraphrase"


def taxonomy_bucket(row: Dict, label: int, features: Dict) -> str:
    if label == 0:
        return human_taxonomy_bucket(row, features)
    return llm_taxonomy_bucket(row, features)


def origin_for(row: Dict) -> str:
    for key in ["round4_source_stage", "round3_source_stage", "round2_source", "generation"]:
        value = row.get(key)
        if value not in [None, ""]:
            return str(value)
    return "unknown_non_teacher_source"


def normalize_candidate(row: Dict, label: int, index: int) -> Dict:
    text = str(row.get("text") or "").strip()
    features = text_features(text)
    bucket = str(row.get("bucket") or features["bucket"])
    round4_bucket = str(row.get("round4_bucket") or bucket)
    round8_bucket = taxonomy_bucket(row, label, features)
    source_file = str(row.get("_source_file") or row.get("round4_source") or "unknown")
    original_id = first_value(row, ["original_id", "id"], default=f"source_{label}_{index:06d}")
    source_pair = first_value(row, ["source_pair_id", "pair_id", "source_id"], default=f"source_pair_{label}_{index:06d}")
    doc_id = source_doc_id(row)

    item = dict(row)
    item.pop("_source_file", None)
    item.pop("_source_path", None)
    item["id"] = f"round8_candidate_{label}_{index:06d}"
    item["source_record_id"] = original_id
    item["text"] = text
    item["label"] = label
    item["domain"] = item.get("domain") or ("academic" if round4_bucket == "academic_formal" else "literature")
    item["generator"] = "human" if label == 0 else item.get("generator") or item.get("source") or "unknown_llm"
    item["origin"] = origin_for(row)
    item["source_file"] = source_file
    item["source_doc_id"] = doc_id
    item["source_pair_id"] = source_pair
    item["pair_id"] = f"round8_{source_pair}"
    item["split_group"] = split_group(
        {
            **item,
            "source_doc_id": doc_id,
            "source_pair_id": source_pair,
            "original_id": original_id,
        }
    )
    item["bucket"] = bucket
    item["round4_bucket"] = round4_bucket
    item["round8_bucket"] = round8_bucket
    item["round8_bucket_family"] = "human_hard_negative" if label == 0 else "llm_hard_positive"
    item["round8_source_stage"] = "residual_candidate_pool_v1"
    item["taxonomy_version"] = "RESIDUAL_ERROR_TAXONOMY.md:2026-05-22"
    item["quality_flags"] = {
        "lexical_jaccard": quality_jaccard(row),
        "source_quality_passed": (row.get("quality") or {}).get("passed_basic_quality_check")
        if isinstance(row.get("quality"), dict)
        else None,
    }

    for key, value in features.items():
        if key != "bucket":
            item.setdefault(key, value)

    return item


def deduplicate_and_filter(
    rows: Sequence[Dict],
    label: int,
    forbidden_texts: set,
    forbidden_pairs: set,
    min_words: int,
    max_words: int,
    allow_short_poetry_words: int,
) -> Tuple[List[Dict], Counter]:
    kept = []
    seen_texts = set()
    seen_source_ids = set()
    skipped = Counter()

    for index, row in enumerate(rows):
        normalized = normalize_candidate(row, label=label, index=index)
        keep, flags = quality_flags(
            normalized,
            label=label,
            min_words=min_words,
            max_words=max_words,
            allow_short_poetry_words=allow_short_poetry_words,
        )
        if not keep:
            for key, value in flags.items():
                if value:
                    skipped[key] += 1
            continue
        key = text_key(normalized)
        source_id = str(normalized.get("source_record_id") or "")
        if key in forbidden_texts:
            skipped["forbidden_text"] += 1
            continue
        if normalized.get("source_pair_id") in forbidden_pairs:
            skipped["forbidden_pair"] += 1
            continue
        if key in seen_texts:
            skipped["duplicate_text"] += 1
            continue
        if source_id and source_id in seen_source_ids:
            skipped["duplicate_source_record_id"] += 1
            continue
        seen_texts.add(key)
        seen_source_ids.add(source_id)
        kept.append(normalized)
    return kept, skipped


def forbidden_from(paths: Sequence[Path]) -> Tuple[set, set]:
    texts = set()
    pairs = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_records(path):
            key = text_key(row)
            if key:
                texts.add(key)
            for field in ["pair_id", "source_pair_id"]:
                value = row.get(field)
                if value not in [None, ""]:
                    pairs.add(str(value))
    return texts, pairs


def cap_by_family(rows: Sequence[Dict], max_total: int) -> List[Dict]:
    if len(rows) <= max_total:
        return list(rows)
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["round8_bucket_family"]].append(row)
    per_family = max_total // max(1, len(by_family))
    capped = []
    for family, family_rows in sorted(by_family.items()):
        family_rows = sorted(family_rows, key=lambda row: (row["round8_bucket"], row["source_record_id"]))
        capped.extend(family_rows[:per_family])
    if len(capped) < max_total:
        used = {row["id"] for row in capped}
        for row in sorted(rows, key=lambda row: (row["round8_bucket"], row["source_record_id"])):
            if row["id"] not in used:
                capped.append(row)
                used.add(row["id"])
            if len(capped) >= max_total:
                break
    return capped


def summarize(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {
            "num_rows": 0,
            "label_distribution": {},
            "round8_bucket_distribution": {},
        }
    lengths = [word_count(row.get("text", "")) for row in rows]
    return {
        "num_rows": len(rows),
        "num_split_groups": len({row.get("split_group") for row in rows}),
        "label_distribution": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "family_distribution": dict(sorted(Counter(str(row.get("round8_bucket_family")) for row in rows).items())),
        "round8_bucket_distribution": dict(sorted(Counter(str(row.get("round8_bucket")) for row in rows).items())),
        "round4_bucket_distribution": dict(sorted(Counter(str(row.get("round4_bucket")) for row in rows).items())),
        "domain_distribution": dict(sorted(Counter(str(row.get("domain")) for row in rows).items())),
        "generator_distribution": dict(sorted(Counter(str(row.get("generator")) for row in rows).items())),
        "source_file_distribution": dict(Counter(str(row.get("source_file")) for row in rows).most_common(25)),
        "word_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
    }


def acceptance(rows: Sequence[Dict], teacher_texts: set, min_total: int, max_total: int) -> Dict:
    labels = Counter(int(row.get("label")) for row in rows)
    bucket_counts = Counter(row.get("round8_bucket") for row in rows)
    missing_buckets = [bucket for bucket in TAXONOMY_BUCKETS if bucket_counts.get(bucket, 0) == 0]
    total = len(rows)
    min_label_share = min(labels.values()) / total if labels and total else 0.0
    teacher_dupes = sum(1 for row in rows if text_key(row) in teacher_texts)
    checks = {
        "total_within_target_range": min_total <= total <= max_total,
        "both_labels_present": labels.get(0, 0) > 0 and labels.get(1, 0) > 0,
        "minimum_label_share_at_least_40pct": min_label_share >= 0.40,
        "teacher_exact_duplicates_zero": teacher_dupes == 0,
        "all_taxonomy_families_present": labels.get(0, 0) > 0 and labels.get(1, 0) > 0,
        "at_least_12_taxonomy_buckets_present": len(TAXONOMY_BUCKETS) - len(missing_buckets) >= 12,
    }
    return {
        "checks": checks,
        "candidate_pool_ready_for_step7_scoring": all(checks.values()),
        "label_counts": dict(sorted(labels.items())),
        "min_label_share": min_label_share,
        "teacher_exact_duplicates": teacher_dupes,
        "taxonomy_buckets_present": len(TAXONOMY_BUCKETS) - len(missing_buckets),
        "taxonomy_buckets_missing": missing_buckets,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build Round8-OneShot residual candidate pool.")
    parser.add_argument("--human_inputs", nargs="+", default=[str(path) for path in DEFAULT_HUMAN_INPUTS])
    parser.add_argument("--llm_inputs", nargs="+", default=[str(path) for path in DEFAULT_LLM_INPUTS])
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--internal_test", default=str(DEFAULT_INTERNAL_TEST))
    parser.add_argument("--teacher_test", default=str(DEFAULT_TEACHER_TEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--min_words", type=int, default=20)
    parser.add_argument("--allow_short_poetry_words", type=int, default=12)
    parser.add_argument("--max_words", type=int, default=300)
    parser.add_argument("--target_min", type=int, default=5000)
    parser.add_argument("--target_max", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    human_paths = [Path(path) for path in args.human_inputs]
    llm_paths = [Path(path) for path in args.llm_inputs]
    valid_path = Path(args.valid)
    internal_test_path = Path(args.internal_test)
    teacher_path = Path(args.teacher_test)

    forbidden_eval_texts, forbidden_eval_pairs = forbidden_from([valid_path, internal_test_path])
    teacher_texts, _teacher_pairs = forbidden_from([teacher_path])
    forbidden_texts = forbidden_eval_texts | teacher_texts

    raw_human, human_input_counts = load_many(human_paths)
    raw_llm, llm_input_counts = load_many(llm_paths)

    human_rows, human_skipped = deduplicate_and_filter(
        raw_human,
        label=0,
        forbidden_texts=forbidden_texts,
        forbidden_pairs=forbidden_eval_pairs,
        min_words=args.min_words,
        max_words=args.max_words,
        allow_short_poetry_words=args.allow_short_poetry_words,
    )
    llm_rows, llm_skipped = deduplicate_and_filter(
        raw_llm,
        label=1,
        forbidden_texts=forbidden_texts,
        forbidden_pairs=forbidden_eval_pairs,
        min_words=args.min_words,
        max_words=args.max_words,
        allow_short_poetry_words=args.allow_short_poetry_words,
    )

    rows = cap_by_family(human_rows + llm_rows, max_total=args.target_max)
    rows = sorted(rows, key=lambda row: (row["round8_bucket_family"], row["round8_bucket"], row["source_record_id"]))
    save_jsonl(rows, Path(args.output))

    report = {
        "inputs": {
            "human_inputs": [str(path) for path in human_paths],
            "llm_inputs": [str(path) for path in llm_paths],
            "valid_text_pair_exclusion": str(valid_path),
            "internal_test_text_pair_exclusion": str(internal_test_path),
            "teacher_test_exact_text_exclusion_only": str(teacher_path),
        },
        "outputs": {
            "candidate_pool": str(Path(args.output)),
            "report": str(Path(args.report)),
        },
        "filters": {
            "min_words": args.min_words,
            "allow_short_poetry_words": args.allow_short_poetry_words,
            "max_words": args.max_words,
            "target_min": args.target_min,
            "target_max": args.target_max,
        },
        "raw_counts": {
            "human_input_counts": human_input_counts,
            "llm_input_counts": llm_input_counts,
            "human_rows_loaded": len(raw_human),
            "llm_rows_loaded": len(raw_llm),
        },
        "skipped": {
            "human": dict(sorted(human_skipped.items())),
            "llm": dict(sorted(llm_skipped.items())),
        },
        "summaries": {
            "human_candidates": summarize(human_rows),
            "llm_candidates": summarize(llm_rows),
            "candidate_pool": summarize(rows),
        },
        "acceptance": acceptance(rows, teacher_texts, min_total=args.target_min, max_total=args.target_max),
        "notes": [
            "Teacher-test text is used only for exact duplicate exclusion; teacher labels are not loaded or used.",
            "The default source is the previously validated Round4 residual seed pool, remapped to Round8 taxonomy buckets.",
            "This file is a candidate pool only. Step7 scoring and hard residual split must happen in later phases.",
        ],
    }
    write_json(report, Path(args.report))

    print("=" * 70)
    print("Built Round8 residual candidate pool")
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    print(f"Ready for Step7 scoring: {report['acceptance']['candidate_pool_ready_for_step7_scoring']}")


if __name__ == "__main__":
    main()
