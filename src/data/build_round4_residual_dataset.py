import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


DEFAULT_BASE_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos_poetry_expansion.jsonl"
)
DEFAULT_VALID = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_valid.jsonl"
DEFAULT_INTERNAL_TEST = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_internal_test.jsonl"
DEFAULT_TEACHER_TEST = PROJECT_ROOT / "data" / "raw" / "teacher_test.json"

DEFAULT_HUMAN_INPUTS = [
    PROJECT_ROOT / "data" / "processed" / "round4_old_prose_human_mirror_candidates.jsonl",
    PROJECT_ROOT / "data" / "processed" / "round3_hard_negative_mirror_source.jsonl",
    PROJECT_ROOT / "data" / "processed" / "round2_human_hardneg_source.jsonl",
    PROJECT_ROOT / "data" / "processed" / "round2_human_hardneg_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "human_hard_negative_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "poetry_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "poetry_expansion_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "academic_seed.jsonl",
    PROJECT_ROOT / "data" / "processed" / "human_seed_combined_with_hardneg.jsonl",
    PROJECT_ROOT / "data" / "processed" / "human_seed_combined.jsonl",
]

DEFAULT_HUMAN_OUT = PROJECT_ROOT / "data" / "processed" / "round4_hard_human_mirror_seed.jsonl"
DEFAULT_LLM_OUT = PROJECT_ROOT / "data" / "processed" / "round4_hard_llm_positive_seed.jsonl"
DEFAULT_TRAIN_OUT = PROJECT_ROOT / "data" / "processed" / "round4_residual_train.jsonl"
DEFAULT_DEV_HARDPOS_OUT = PROJECT_ROOT / "data" / "processed" / "round4_residual_dev_hardpos.jsonl"
DEFAULT_DEV_HARDNEG_OUT = PROJECT_ROOT / "data" / "processed" / "round4_residual_dev_hardneg.jsonl"
DEFAULT_SPOTCHECK_OUT = PROJECT_ROOT / "data" / "processed" / "round4_residual_spotcheck.jsonl"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "data" / "processed" / "round4_residual_report.json"


HUMAN_BUCKET_TARGETS = {
    "literary_old_prose": 800,
    "poetry_classical": 500,
    "poetry_freeverse": 1000,
    "academic_formal": 1000,
    "literary_short_fragment": 700,
    "general_prose": 1000,
}

LLM_BUCKET_TARGETS = {
    "literary_old_prose": 300,
    "poetry_classical": 500,
    "poetry_freeverse": 900,
    "academic_formal": 900,
    "literary_short_fragment": 900,
    "general_prose": 800,
}

HIGH_RISK_HUMAN_BUCKETS = {
    "poetry_classical",
    "poetry_freeverse",
    "literary_old_prose",
    "literary_short_fragment",
    "academic_formal",
}

PROMPT_LEAKAGE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bas an ai\b",
        r"\bi cannot\b",
        r"\bi can't\b",
        r"\bsure[,! ]+here",
        r"\bhere is (the )?(rewrite|paraphrase)",
        r"\brewritten (version|text)",
    ]
]


def default_llm_inputs() -> List[Path]:
    processed = PROJECT_ROOT / "data" / "processed"
    paths = [
        processed / "round3_llm_hardpos_multi_generator_seed.jsonl",
        processed / "round2_llm_hardpos_seed.jsonl",
    ]
    for path in sorted(processed.glob("llm_rewrite*.jsonl")):
        name = path.name.lower()
        if "failed" in name:
            continue
        paths.append(path)
    seen = set()
    out = []
    for path in paths:
        if path.exists() and path not in seen:
            out.append(path)
            seen.add(path)
    return out


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


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().lower())
    return value.strip("_") or "unknown"


def word_count(text: str) -> int:
    return len(str(text or "").split())


def has_prompt_leakage(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in PROMPT_LEAKAGE_PATTERNS)


def source_stage(path_name: str) -> str:
    name = path_name.lower()
    if name.startswith("round3_"):
        return "round3_reused_pool"
    if name.startswith("round2_") or "round2_" in name:
        return "round2_reused_pool"
    if "hard_positive" in name or "hardpos" in name:
        return "hard_positive_pool"
    if "poetry" in name:
        return "poetry_pool"
    if "academic" in name:
        return "academic_pool"
    if "hard_negative" in name:
        return "hard_negative_pool"
    return "generic_rewrite_pool"


def infer_round4_bucket(label: int, bucket: str, row: Dict, path_name: str) -> str:
    metadata = row.get("metadata") or {}
    signal = " ".join(
        str(value or "").lower()
        for value in [
            path_name,
            row.get("round2_tag"),
            row.get("round3_tag"),
            row.get("subdomain"),
            row.get("prompt_type"),
            row.get("domain"),
            metadata.get("hard_negative_reason"),
        ]
    )
    if label == 0 and (
        "round4_old_prose_human_mirror_candidates" in signal
        or "short_polished_or_archaic_literary_prose" in signal
    ):
        return "literary_old_prose"
    if label == 1 and (
        "old_fiction" in signal
        or "old_prose" in signal
        or "old-prose" in signal
        or "older literary fiction" in signal
    ):
        return "literary_old_prose"
    return bucket


def human_tag(round4_bucket: str, row: Dict) -> str:
    if round4_bucket == "poetry_classical":
        return "human_poetry_classical_mirror"
    if round4_bucket == "poetry_freeverse":
        return "human_poetry_freeverse_mirror"
    if round4_bucket == "literary_old_prose":
        return "human_literary_old_prose_mirror"
    if round4_bucket == "academic_formal" or row.get("domain") == "academic":
        return "human_formal_academic_mirror"
    if round4_bucket == "literary_short_fragment":
        return "human_short_fragment_mirror"
    return "human_polished_general_mirror"


def llm_tag(round4_bucket: str, row: Dict) -> str:
    prompt_type = str(row.get("prompt_type", "")).lower()
    domain = str(row.get("domain", "")).lower()
    if round4_bucket == "poetry_classical" or "archaic" in prompt_type:
        return "llm_archaic_poetry_hardpos"
    if round4_bucket == "poetry_freeverse" or domain == "poetry":
        return "llm_freeverse_poetry_hardpos"
    if round4_bucket == "academic_formal" or domain == "academic":
        return "llm_natural_academic_hardpos"
    if round4_bucket == "literary_old_prose" or "old" in prompt_type:
        return "llm_old_prose_hardpos"
    if round4_bucket == "literary_short_fragment":
        return "llm_short_fragment_hardpos"
    return "llm_polished_general_hardpos"


def sample_weight(label: int, bucket: str, tag: str) -> float:
    if label == 1:
        return 1.0
    if bucket in {"poetry_classical", "poetry_freeverse"}:
        return 2.0
    if bucket == "literary_old_prose":
        return 2.0
    if bucket == "literary_short_fragment":
        return 1.7
    if bucket == "academic_formal":
        return 1.4
    if tag == "human_polished_general_mirror":
        return 1.5
    return 1.2


def source_priority(row: Dict) -> int:
    name = str(row.get("round4_source", "")).lower()
    if name.startswith("round3_"):
        return 0
    if name.startswith("round2_"):
        return 1
    if "hard_positive" in name or "hardpos" in name:
        return 2
    if "poetry" in name or "academic" in name:
        return 3
    return 4


def normalize_candidate(row: Dict, label: int, index: int) -> Dict:
    text = str(row.get("text", "")).strip()
    bucket = row.get("bucket") or assign_bucket(text)
    path_name = Path(str(row.get("_input_path", ""))).name
    round4_bucket = infer_round4_bucket(label, bucket, row, path_name)
    path_slug = slugify(Path(path_name).stem)
    original_id = safe_id(row.get("id"), f"candidate_{label}_{index:06d}")
    original_pair_id = safe_id(row.get("pair_id") or row.get("source_id"), f"pair_{label}_{index:06d}")
    tag = human_tag(round4_bucket, row) if label == 0 else llm_tag(round4_bucket, row)
    features = text_features(text)

    item = dict(row)
    item.pop("_input_path", None)
    item["original_id"] = original_id
    item["source_pair_id"] = original_pair_id
    item["id"] = f"round4_{path_slug}_{label}_{index:06d}"
    item["text"] = text
    item["label"] = label
    item["domain"] = item.get("domain") or ("academic" if bucket == "academic_formal" else "literature")
    item["generator"] = "human" if label == 0 else item.get("generator") or item.get("source") or "unknown_llm"
    item["source"] = item.get("source") or item["generator"]
    item["model"] = "human" if label == 0 else item.get("model", item["generator"])
    item["generation"] = "human" if label == 0 else item.get("generation", "llm_rewrite")
    item["pair_id"] = f"round4_{path_slug}_{original_pair_id}"
    item["source_id"] = item.get("source_id") or original_id
    item["bucket"] = bucket
    item["round4_bucket"] = round4_bucket
    item["round4_tag"] = tag
    item["subdomain"] = tag
    item["round4_source"] = path_name
    item["round4_source_stage"] = source_stage(path_name)
    item["sample_weight"] = sample_weight(label, round4_bucket, tag)

    metadata = dict(item.get("metadata") or {})
    metadata["round4_residual_repair"] = True
    metadata["round4_source"] = path_name
    metadata["round4_text_bucket"] = bucket
    metadata["round4_bucket"] = round4_bucket
    item["metadata"] = metadata

    for key, value in features.items():
        if key != "bucket":
            item.setdefault(key, value)

    return item


def quality_ok(row: Dict, label: int) -> Tuple[bool, str]:
    text = str(row.get("text", "")).strip()
    if not text:
        return False, "empty_text"
    if word_count(text) < 15:
        return False, "too_short"
    if label == 1:
        quality = row.get("quality") or {}
        if quality.get("passed_basic_quality_check") is False:
            return False, "failed_basic_quality"
        if has_prompt_leakage(text):
            return False, "prompt_leakage"
    return True, "ok"


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
    label: int,
) -> Tuple[List[Dict], Counter]:
    kept = []
    seen_ids = set()
    seen_texts = set()
    skipped = Counter()

    for row in rows:
        is_ok, reason = quality_ok(row, label=label)
        if not is_ok:
            skipped[reason] += 1
            continue
        sample_id = row.get("id")
        pair_id = row.get("pair_id")
        source_pair_id = row.get("source_pair_id")
        key = row_key(row)
        if sample_id in forbidden_ids:
            skipped["forbidden_id"] += 1
            continue
        if pair_id in forbidden_pairs or source_pair_id in forbidden_pairs:
            skipped["forbidden_pair"] += 1
            continue
        if key in forbidden_texts:
            skipped["forbidden_text"] += 1
            continue
        if sample_id in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        if key in seen_texts:
            skipped["duplicate_text"] += 1
            continue
        seen_ids.add(sample_id)
        seen_texts.add(key)
        kept.append(row)
    return kept, skipped


def select_by_bucket_targets(
    rows: Sequence[Dict],
    targets: Dict[str, int],
    min_total: int,
    seed: int,
) -> Tuple[List[Dict], Dict]:
    rng = random.Random(seed)
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[str(row.get("round4_bucket", row.get("bucket", "unknown")))].append(row)

    selected = []
    selected_ids = set()
    bucket_report = {}

    for bucket, target in targets.items():
        candidates = list(by_bucket.get(bucket, []))
        candidates.sort(key=lambda row: (source_priority(row), row_key(row)))
        if len(candidates) > target:
            preferred = candidates[: max(target * 2, target)]
            rng.shuffle(preferred)
            candidates = preferred[:target]
        for row in candidates[:target]:
            selected.append(row)
            selected_ids.add(id(row))
        bucket_report[bucket] = {
            "available": len(by_bucket.get(bucket, [])),
            "target": target,
            "selected": min(len(by_bucket.get(bucket, [])), target),
            "shortfall": max(0, target - len(by_bucket.get(bucket, []))),
        }

    if len(selected) < min_total:
        remaining = [row for row in rows if id(row) not in selected_ids]
        remaining.sort(key=lambda row: (source_priority(row), row_key(row)))
        need = min_total - len(selected)
        selected.extend(remaining[:need])
        selected_ids.update(id(row) for row in remaining[:need])

    selected.sort(key=lambda row: (int(row["label"]), str(row.get("bucket", "")), row_key(row)))
    return selected, {
        "min_total": min_total,
        "selected_total": len(selected),
        "bucket_targets": bucket_report,
    }


def split_dev_by_pair(
    selected_rows: Sequence[Dict],
    base_train_pairs: set,
    base_train_texts: set,
    dev_per_label: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    forced_train = []
    eligible = []
    for row in selected_rows:
        if (
            row.get("pair_id") in base_train_pairs
            or row.get("source_pair_id") in base_train_pairs
            or row_key(row) in base_train_texts
        ):
            forced_train.append(row)
        else:
            eligible.append(row)

    grouped = defaultdict(list)
    for row in eligible:
        grouped[str(row.get("pair_id"))].append(row)

    groups = list(grouped.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    dev_rows = []
    train_rows = []
    dev_counts = Counter()
    for group in groups:
        labels = Counter(int(row["label"]) for row in group)
        fits = all(dev_counts[label] + count <= dev_per_label for label, count in labels.items())
        if fits and any(dev_counts[label] < dev_per_label for label in labels):
            dev_rows.extend(group)
            dev_counts.update(labels)
        else:
            train_rows.extend(group)

    train_rows = forced_train + train_rows
    dev_hardpos = [row for row in dev_rows if int(row["label"]) == 1]
    dev_hardneg = [row for row in dev_rows if int(row["label"]) == 0]
    return train_rows, dev_hardpos, dev_hardneg


def enrich_base_train(rows: Sequence[Dict]) -> List[Dict]:
    enriched = []
    for i, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        if not text or row.get("label") not in [0, 1]:
            continue
        item = dict(row)
        item["id"] = safe_id(item.get("id"), f"round4_base_{i:06d}")
        bucket = item.get("bucket") or assign_bucket(text)
        item["bucket"] = bucket
        item.setdefault("round4_tag", "base_human" if int(item["label"]) == 0 else "base_llm")
        item.setdefault("round4_source_stage", "step7_base_train")
        item.setdefault("sample_weight", 1.0)
        enriched.append(item)
    return enriched


def remove_train_duplicates(rows: Sequence[Dict]) -> Tuple[List[Dict], Counter]:
    kept = []
    seen_ids = set()
    seen_texts = set()
    skipped = Counter()
    for row in rows:
        sample_id = row.get("id")
        key = row_key(row)
        if sample_id in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        if key in seen_texts:
            skipped["duplicate_text"] += 1
            continue
        seen_ids.add(sample_id)
        seen_texts.add(key)
        kept.append(row)
    return kept, skipped


def summarize(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {
            "num_samples": 0,
            "label_distribution": {},
            "bucket_distribution": {},
            "round4_tag_distribution": {},
        }
    lengths = [word_count(str(row.get("text", ""))) for row in rows]
    return {
        "num_samples": len(rows),
        "num_pair_ids": len({row.get("pair_id") for row in rows}),
        "label_distribution": dict(Counter(str(row.get("label")) for row in rows)),
        "domain_distribution": dict(Counter(str(row.get("domain", "unknown")) for row in rows)),
        "generator_distribution": dict(Counter(str(row.get("generator", "unknown")) for row in rows)),
        "bucket_distribution": dict(Counter(str(row.get("bucket", "unknown")) for row in rows)),
        "round4_bucket_distribution": dict(
            Counter(str(row.get("round4_bucket", row.get("bucket", "unknown"))) for row in rows)
        ),
        "round4_tag_distribution": dict(Counter(str(row.get("round4_tag", "unknown")) for row in rows)),
        "source_stage_distribution": dict(Counter(str(row.get("round4_source_stage", "unknown")) for row in rows)),
        "sample_weight_distribution": dict(Counter(str(row.get("sample_weight", "1.0")) for row in rows)),
        "word_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
    }


def spotcheck(rows: Sequence[Dict], n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    out = []
    for row in rows[: min(n, len(rows))]:
        out.append(
            {
                "id": row.get("id"),
                "label": row.get("label"),
                "bucket": row.get("bucket"),
                "round4_bucket": row.get("round4_bucket"),
                "round4_tag": row.get("round4_tag"),
                "generator": row.get("generator"),
                "source": row.get("source"),
                "round4_source": row.get("round4_source"),
                "pair_id": row.get("pair_id"),
                "text": row.get("text"),
                "manual_check": {
                    "quality_ok": None,
                    "prompt_leakage": None,
                    "teacher_near_duplicate": None,
                    "notes": "",
                },
            }
        )
    return out


def acceptance(
    human_rows: Sequence[Dict],
    llm_rows: Sequence[Dict],
    dev_hardpos: Sequence[Dict],
    dev_hardneg: Sequence[Dict],
    spotcheck_rows: Sequence[Dict],
    teacher_texts: set,
) -> Dict:
    human_buckets = Counter(str(row.get("round4_bucket", row.get("bucket"))) for row in human_rows)
    llm_buckets = Counter(str(row.get("round4_bucket", row.get("bucket"))) for row in llm_rows)
    teacher_duplicates = sum(1 for row in list(human_rows) + list(llm_rows) if row_key(row) in teacher_texts)
    hard_human = len(human_rows)
    hard_llm = len(llm_rows)
    poetry_freeverse_mirrors = human_buckets.get("poetry_classical", 0) + human_buckets.get("poetry_freeverse", 0)
    checks = {
        "hard_human_negatives_at_least_3000": hard_human >= 3000,
        "hard_llm_positives_at_least_3000": hard_llm >= 3000,
        "old_prose_human_mirrors_at_least_800": human_buckets.get("literary_old_prose", 0) >= 800,
        "poetry_freeverse_human_mirrors_at_least_1000": poetry_freeverse_mirrors >= 1000,
        "manual_spotcheck_at_least_100": len(spotcheck_rows) >= 100,
        "no_teacher_test_exact_text_duplicates": teacher_duplicates == 0,
        "human_to_llm_ratio_at_least_1": hard_human >= hard_llm,
    }
    return {
        "checks": checks,
        "meets_all_acceptance": all(checks.values()),
        "hard_human_negatives": hard_human,
        "hard_llm_positives": hard_llm,
        "hard_human_to_llm_ratio": None if hard_llm == 0 else hard_human / hard_llm,
        "old_prose_human_mirrors": human_buckets.get("literary_old_prose", 0),
        "poetry_freeverse_human_mirrors": poetry_freeverse_mirrors,
        "natural_academic_human_mirrors": human_buckets.get("academic_formal", 0),
        "old_prose_llm_positives": llm_buckets.get("literary_old_prose", 0),
        "dev_hardpos_rows": len(dev_hardpos),
        "dev_hardneg_rows": len(dev_hardneg),
        "teacher_test_exact_text_duplicates": teacher_duplicates,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build Round4 paired residual repair datasets.")
    parser.add_argument("--base_train", default=str(DEFAULT_BASE_TRAIN))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--internal_test", default=str(DEFAULT_INTERNAL_TEST))
    parser.add_argument("--teacher_test", default=str(DEFAULT_TEACHER_TEST))
    parser.add_argument("--human_inputs", nargs="+", default=[str(path) for path in DEFAULT_HUMAN_INPUTS])
    parser.add_argument("--llm_inputs", nargs="+", default=[str(path) for path in default_llm_inputs()])
    parser.add_argument("--human_output", default=str(DEFAULT_HUMAN_OUT))
    parser.add_argument("--llm_output", default=str(DEFAULT_LLM_OUT))
    parser.add_argument("--train_output", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--dev_hardpos_output", default=str(DEFAULT_DEV_HARDPOS_OUT))
    parser.add_argument("--dev_hardneg_output", default=str(DEFAULT_DEV_HARDNEG_OUT))
    parser.add_argument("--spotcheck_output", default=str(DEFAULT_SPOTCHECK_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_OUT))
    parser.add_argument("--dev_per_label", type=int, default=500)
    parser.add_argument("--spotcheck_n", type=int, default=100)
    parser.add_argument("--min_human", type=int, default=3000)
    parser.add_argument("--min_llm", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def main():
    args = parse_args()
    base_train = load_records(Path(args.base_train))
    valid = load_records(Path(args.valid)) if Path(args.valid).exists() else []
    internal_test = load_records(Path(args.internal_test)) if Path(args.internal_test).exists() else []
    teacher_test = load_records(Path(args.teacher_test)) if Path(args.teacher_test).exists() else []

    base_train_ids, base_train_pairs, base_train_texts = forbidden_sets(base_train)
    valid_ids, valid_pairs, valid_texts = forbidden_sets(valid)
    internal_ids, internal_pairs, internal_texts = forbidden_sets(internal_test)
    _teacher_ids, _teacher_pairs, teacher_texts = forbidden_sets(teacher_test)

    forbidden_ids = valid_ids | internal_ids
    forbidden_pairs = valid_pairs | internal_pairs
    forbidden_texts = valid_texts | internal_texts | teacher_texts

    raw_human = load_many([Path(path) for path in args.human_inputs])
    raw_llm = load_many([Path(path) for path in args.llm_inputs])

    human_candidates = [normalize_candidate(row, label=0, index=i) for i, row in enumerate(raw_human)]
    llm_candidates = [normalize_candidate(row, label=1, index=i) for i, row in enumerate(raw_llm)]

    human_candidates, human_skipped = dedup_candidates(
        human_candidates,
        forbidden_ids=forbidden_ids,
        forbidden_pairs=forbidden_pairs,
        forbidden_texts=forbidden_texts,
        label=0,
    )
    llm_candidates, llm_skipped = dedup_candidates(
        llm_candidates,
        forbidden_ids=forbidden_ids,
        forbidden_pairs=forbidden_pairs,
        forbidden_texts=forbidden_texts,
        label=1,
    )

    selected_human, human_selection_report = select_by_bucket_targets(
        human_candidates,
        targets=HUMAN_BUCKET_TARGETS,
        min_total=args.min_human,
        seed=args.seed,
    )
    selected_llm, llm_selection_report = select_by_bucket_targets(
        llm_candidates,
        targets=LLM_BUCKET_TARGETS,
        min_total=args.min_llm,
        seed=args.seed + 1,
    )

    train_additions, dev_hardpos, dev_hardneg = split_dev_by_pair(
        selected_human + selected_llm,
        base_train_pairs=base_train_pairs,
        base_train_texts=base_train_texts,
        dev_per_label=args.dev_per_label,
        seed=args.seed,
    )
    base_train_enriched = enrich_base_train(base_train)
    round4_train, train_skipped = remove_train_duplicates(base_train_enriched + train_additions)
    spotcheck_rows = spotcheck(selected_human + selected_llm, n=args.spotcheck_n, seed=args.seed)

    save_jsonl(selected_human, Path(args.human_output))
    save_jsonl(selected_llm, Path(args.llm_output))
    save_jsonl(round4_train, Path(args.train_output))
    save_jsonl(dev_hardpos, Path(args.dev_hardpos_output))
    save_jsonl(dev_hardneg, Path(args.dev_hardneg_output))
    save_jsonl(spotcheck_rows, Path(args.spotcheck_output))

    report = {
        "inputs": {
            "base_train": str(Path(args.base_train)),
            "valid": str(Path(args.valid)),
            "internal_test": str(Path(args.internal_test)),
            "teacher_test_text_dedup_only": str(Path(args.teacher_test)),
            "human_inputs": [str(path) for path in args.human_inputs],
            "llm_inputs": [str(path) for path in args.llm_inputs],
        },
        "outputs": {
            "human_seed": str(Path(args.human_output)),
            "llm_seed": str(Path(args.llm_output)),
            "train": str(Path(args.train_output)),
            "dev_hardpos": str(Path(args.dev_hardpos_output)),
            "dev_hardneg": str(Path(args.dev_hardneg_output)),
            "spotcheck": str(Path(args.spotcheck_output)),
            "report": str(Path(args.report)),
        },
        "seed": args.seed,
        "raw_counts": {
            "human_rows_loaded": len(raw_human),
            "llm_rows_loaded": len(raw_llm),
            "base_train_rows": len(base_train),
            "valid_rows": len(valid),
            "internal_test_rows": len(internal_test),
            "teacher_test_rows": len(teacher_test),
        },
        "skipped": {
            "human_candidates": dict(human_skipped),
            "llm_candidates": dict(llm_skipped),
            "round4_train": dict(train_skipped),
        },
        "selection": {
            "human": human_selection_report,
            "llm": llm_selection_report,
        },
        "summaries": {
            "human_candidates": summarize(human_candidates),
            "llm_candidates": summarize(llm_candidates),
            "selected_human_seed": summarize(selected_human),
            "selected_llm_seed": summarize(selected_llm),
            "dev_hardpos": summarize(dev_hardpos),
            "dev_hardneg": summarize(dev_hardneg),
            "round4_train": summarize(round4_train),
        },
        "acceptance": acceptance(
            selected_human,
            selected_llm,
            dev_hardpos,
            dev_hardneg,
            spotcheck_rows,
            teacher_texts,
        ),
        "notes": [
            "Teacher-test text is used only as an exact-text leakage exclusion set; labels are not used.",
            "Round4 selected seeds intentionally prefer high-risk buckets and prior hard-positive sources.",
            "If old-prose mirror acceptance fails, collect more public-domain old-prose human passages before neural retraining.",
        ],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Built Round4 residual repair datasets")
    print("=" * 70)
    print(f"Hard human mirrors: {len(selected_human)} -> {args.human_output}")
    print(f"Hard LLM positives: {len(selected_llm)} -> {args.llm_output}")
    print(f"Round4 train rows: {len(round4_train)} -> {args.train_output}")
    print(f"Dev hardpos rows: {len(dev_hardpos)} -> {args.dev_hardpos_output}")
    print(f"Dev hardneg rows: {len(dev_hardneg)} -> {args.dev_hardneg_output}")
    print(f"Spotcheck rows: {len(spotcheck_rows)} -> {args.spotcheck_output}")
    print(f"Report: {args.report}")
    print(f"Meets all acceptance: {report['acceptance']['meets_all_acceptance']}")


if __name__ == "__main__":
    main()
