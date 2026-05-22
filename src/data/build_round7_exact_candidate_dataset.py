import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records


DEFAULT_LEDGER = PROJECT_ROOT / "outputs" / "evaluation" / "round5_flip_ledger.jsonl"
DEFAULT_PROBE = PROJECT_ROOT / "data" / "processed" / "round6_override_probe_mixed.jsonl"
DEFAULT_TEACHER = PROJECT_ROOT / "data" / "raw" / "teacher_test.json"

DEFAULT_TRAIN_OUT = PROJECT_ROOT / "data" / "processed" / "round7_exact_candidate_train.jsonl"
DEFAULT_DEV_OUT = PROJECT_ROOT / "data" / "processed" / "round7_exact_candidate_dev.jsonl"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "data" / "processed" / "round7_exact_candidate_dataset_report.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "data" / "processed" / "round7_exact_candidate_dataset_report.md"

SAFE_LABEL = 1
UNSAFE_LABEL = 0
SAFE_FLIP = "fixed_fn_candidate"
UNSAFE_FLIP = "induced_fp"
EXACT_FLIPS = {SAFE_FLIP, UNSAFE_FLIP}
WATCHLIST_BUCKETS = [
    "general_prose",
    "literary_short_fragment",
    "literary_old_prose",
    "academic_formal",
]


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def row_id(row: Dict, index: int) -> str:
    value = row.get("id")
    return str(value) if value not in [None, ""] else f"row_{index:06d}"


def text_key(row: Dict) -> str:
    return " ".join(str(row.get("text") or "").lower().split())


def group_key(row: Dict, index: int) -> str:
    for key in ["pair_id", "source_pair_id", "source_id", "original_id", "id"]:
        value = row.get(key)
        if value not in [None, ""]:
            return str(value)
    return row_id(row, index)


def int_value(value, default: int = -1) -> int:
    try:
        if value in [None, ""]:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def exact_candidate(row: Dict) -> bool:
    return (
        row.get("flip_type") in EXACT_FLIPS
        and int_value(row.get("step7_pred"), 99) == 0
        and int_value(row.get("round4_pred"), 99) == 1
    )


def safety_for(row: Dict) -> str:
    return "safe_override" if row.get("flip_type") == SAFE_FLIP else "unsafe_override"


def selector_label_for(row: Dict) -> int:
    return SAFE_LABEL if safety_for(row) == "safe_override" else UNSAFE_LABEL


def normalize_candidate(row: Dict, index: int) -> Dict:
    item = dict(row)
    bucket = str(item.get("bucket") or assign_bucket(str(item.get("text") or "")))
    round4_bucket = str(item.get("round4_bucket") or bucket)
    detection_label = int_value(item.get("label"))
    selector_label = selector_label_for(item)
    item["id"] = row_id(item, index)
    item["label"] = selector_label
    item["target"] = selector_label
    item["safe_selector_label"] = selector_label
    item["round7_selector_label"] = selector_label
    item["positive_label_meaning"] = "safe_override"
    item["original_detection_label"] = detection_label
    item["override_safety"] = safety_for(item)
    item["round7_source_kind"] = "round5_exact_candidate"
    item["round7_origin_split"] = str(item.get("split") or "unknown")
    item["round7_group_key"] = group_key(item, index)
    item["round7_text_key"] = text_key(item)
    item["is_exact_override_candidate"] = True
    item["bucket"] = bucket
    item["round4_bucket"] = round4_bucket
    return item


def count_by_safety(rows: Sequence[Dict]) -> Counter:
    return Counter(str(row.get("override_safety") or "unknown") for row in rows)


def forbidden_texts(paths: Sequence[Path]) -> set:
    keys = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_records(path):
            key = text_key(row)
            if key:
                keys.add(key)
    return keys


def ledger_pool(rows: Sequence[Dict], source_splits: Sequence[str]) -> List[Dict]:
    allowed = set(source_splits)
    out = []
    for index, row in enumerate(rows):
        if str(row.get("split") or "") not in allowed:
            continue
        if exact_candidate(row):
            out.append(normalize_candidate(row, index))
    return out


def extra_exact_pool(paths: Sequence[Path]) -> Tuple[List[Dict], Dict[str, int]]:
    out = []
    source_counts = Counter()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Cannot find extra exact pool: {path}")
        rows = load_records(path)
        source_counts[str(path)] += len(rows)
        for index, row in enumerate(rows):
            if exact_candidate(row):
                item = normalize_candidate(row, index)
                item["round7_source_kind"] = "extra_exact_candidate"
                item["round7_extra_source_path"] = str(path)
                out.append(item)
    return out, dict(source_counts)


def deduplicate(rows: Sequence[Dict], forbidden_keys: set) -> Tuple[List[Dict], Dict[str, int]]:
    kept = []
    skipped = Counter()
    seen_ids = set()
    seen_texts = set()
    for row in rows:
        if not str(row.get("text") or "").strip():
            skipped["empty_text"] += 1
            continue
        if row["round7_text_key"] in forbidden_keys:
            skipped["forbidden_text"] += 1
            continue
        if row["id"] in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        if row["round7_text_key"] in seen_texts:
            skipped["duplicate_text"] += 1
            continue
        seen_ids.add(row["id"])
        seen_texts.add(row["round7_text_key"])
        kept.append(row)
    return kept, dict(skipped)


def provisional_dev_target(available: int, min_train: int, min_dev: int, fallback_fraction: float) -> int:
    if available >= min_train + min_dev:
        return min_dev
    if available <= 1:
        return available
    return max(1, min(available - 1, round(available * fallback_fraction)))


def group_rows(rows: Sequence[Dict]) -> List[List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["round7_group_key"])].append(row)
    return list(grouped.values())


def group_counts(rows: Sequence[Dict]) -> Counter:
    return count_by_safety(rows)


def split_pool(rows: Sequence[Dict], dev_targets: Dict[str, int], seed: int) -> Tuple[List[Dict], List[Dict]]:
    groups = group_rows(rows)
    rng = random.Random(seed)
    groups.sort(key=lambda group: (str(group[0].get("round7_origin_split") or ""), str(group[0]["round7_group_key"])))
    rng.shuffle(groups)

    train = []
    dev = []
    dev_counts = Counter()
    for group in groups:
        counts = group_counts(group)
        residual_need = sum(max(0, dev_targets.get(name, 0) - dev_counts.get(name, 0)) for name in counts)
        if residual_need > 0:
            dev.extend(group)
            dev_counts.update(counts)
        else:
            train.extend(group)
    return train, dev


def summarize(rows: Sequence[Dict]) -> Dict:
    return {
        "num_rows": len(rows),
        "override_safety_distribution": dict(sorted(count_by_safety(rows).items())),
        "origin_split_distribution": dict(sorted(Counter(str(row.get("round7_origin_split") or "unknown") for row in rows).items())),
        "round4_bucket_distribution": dict(sorted(Counter(str(row.get("round4_bucket") or "unknown") for row in rows).items())),
        "group_count": len({row["round7_group_key"] for row in rows}),
        "exact_candidate_rows": sum(1 for row in rows if row.get("is_exact_override_candidate")),
    }


def bucket_watchlist(rows: Sequence[Dict]) -> Dict[str, Dict[str, int]]:
    counts = defaultdict(Counter)
    for row in rows:
        bucket = str(row.get("round4_bucket") or "unknown")
        if bucket in WATCHLIST_BUCKETS:
            counts[str(row.get("override_safety") or "unknown")][bucket] += 1
    return {
        safety: {bucket: counts[safety].get(bucket, 0) for bucket in WATCHLIST_BUCKETS}
        for safety in ["safe_override", "unsafe_override"]
    }


def leakage_report(train: Sequence[Dict], dev: Sequence[Dict], probe_rows: Sequence[Dict], teacher_keys: set) -> Dict:
    train_groups = {row["round7_group_key"] for row in train}
    dev_groups = {row["round7_group_key"] for row in dev}
    train_texts = {row["round7_text_key"] for row in train}
    dev_texts = {row["round7_text_key"] for row in dev}
    probe_texts = {text_key(row) for row in probe_rows if text_key(row)}
    all_rows = list(train) + list(dev)
    return {
        "train_dev_group_overlap": len(train_groups & dev_groups),
        "train_dev_text_overlap": len(train_texts & dev_texts),
        "teacher_exact_text_duplicates": sum(1 for row in all_rows if row["round7_text_key"] in teacher_keys),
        "probe_in_train_text_overlap": len(probe_texts & train_texts),
        "probe_in_dev_text_overlap": len(probe_texts & dev_texts),
    }


def target_status(train: Sequence[Dict], dev: Sequence[Dict], args) -> Dict:
    train_counts = count_by_safety(train)
    dev_counts = count_by_safety(dev)
    status = {
        "train_safe": {
            "observed": train_counts.get("safe_override", 0),
            "target": args.min_train_safe,
        },
        "train_unsafe": {
            "observed": train_counts.get("unsafe_override", 0),
            "target": args.min_train_unsafe,
        },
        "dev_safe": {
            "observed": dev_counts.get("safe_override", 0),
            "target": args.min_dev_safe,
        },
        "dev_unsafe": {
            "observed": dev_counts.get("unsafe_override", 0),
            "target": args.min_dev_unsafe,
        },
    }
    for block in status.values():
        block["shortfall"] = max(0, block["target"] - block["observed"])
        block["pass"] = block["shortfall"] == 0
    return status


def acceptance(targets: Dict, leakage: Dict) -> Dict:
    checks = {
        "train_safe_exact_target_met": targets["train_safe"]["pass"],
        "train_unsafe_exact_target_met": targets["train_unsafe"]["pass"],
        "dev_safe_exact_target_met": targets["dev_safe"]["pass"],
        "dev_unsafe_exact_target_met": targets["dev_unsafe"]["pass"],
        "train_dev_group_overlap_zero": leakage["train_dev_group_overlap"] == 0,
        "train_dev_text_overlap_zero": leakage["train_dev_text_overlap"] == 0,
        "teacher_exact_duplicate_zero": leakage["teacher_exact_text_duplicates"] == 0,
        "probe_train_overlap_zero": leakage["probe_in_train_text_overlap"] == 0,
        "probe_dev_overlap_zero": leakage["probe_in_dev_text_overlap"] == 0,
    }
    return {
        "checks": checks,
        "exact_selector_training_ready": all(checks.values()),
    }


def write_target_table(lines: List[str], targets: Dict) -> None:
    lines.extend(["", "## Exact-Candidate Targets", "", "| Target | Observed | Required | Shortfall | Pass |", "| --- | ---: | ---: | ---: | --- |"])
    for name, block in targets.items():
        lines.append(f"| {name} | {block['observed']} | {block['target']} | {block['shortfall']} | {block['pass']} |")


def write_summary_table(lines: List[str], summaries: Dict[str, Dict]) -> None:
    lines.extend(
        [
            "",
            "## Row Counts",
            "",
            "| Split | Rows | Safe | Unsafe | Groups | Exact rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, block in summaries.items():
        safety = block["override_safety_distribution"]
        lines.append(
            f"| {name} | {block['num_rows']} | {safety.get('safe_override', 0)} | "
            f"{safety.get('unsafe_override', 0)} | {block['group_count']} | {block['exact_candidate_rows']} |"
        )


def write_bucket_table(lines: List[str], title: str, counts: Dict[str, Dict[str, int]]) -> None:
    lines.extend(["", f"## {title}", "", "| Safety | Round4 bucket | Rows |", "| --- | --- | ---: |"])
    for safety, buckets in counts.items():
        for bucket, count in buckets.items():
            lines.append(f"| {safety} | {bucket} | {count} |")


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round7 Exact Candidate Dataset Report",
        "",
        "This build uses non-teacher Round5 exact disagreement candidates from `hardpos` and `hardneg` only.",
        "The Round6 internal exact probe is read for overlap checks and is not written into Round7 train/dev.",
        "",
        "## Label Definition",
        "",
        "```text",
        "label = 1 means safe_override",
        "label = 0 means unsafe_override",
        "original_detection_label preserves the detector label",
        "```",
    ]
    write_target_table(lines, report["target_status"])
    write_summary_table(lines, report["summaries"])
    write_bucket_table(lines, "Watchlist Bucket Coverage In Candidate Pool", report["bucket_watchlist"]["pool"])
    write_bucket_table(lines, "Watchlist Bucket Coverage In Train", report["bucket_watchlist"]["train"])
    write_bucket_table(lines, "Watchlist Bucket Coverage In Dev", report["bucket_watchlist"]["dev"])
    lines.extend(["", "## Leakage", "", "| Item | Count |", "| --- | ---: |"])
    for key, value in report["leakage"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Acceptance", "", "| Check | Pass |", "| --- | --- |"])
    for key, value in report["acceptance"]["checks"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Decision", "", "```text", report["decision"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_source_splits(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Build the Round7 exact-candidate dataset from non-teacher disagreements.")
    parser.add_argument("--round5_ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument(
        "--extra_exact_pool",
        action="append",
        default=[],
        help="Additional non-teacher exact-like candidate JSONL. Rows must carry Step7/Round4 disagreement fields.",
    )
    parser.add_argument("--probe_mixed", default=str(DEFAULT_PROBE))
    parser.add_argument("--teacher_test", default=str(DEFAULT_TEACHER))
    parser.add_argument("--source_splits", default="hardpos,hardneg")
    parser.add_argument("--train_output", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--dev_output", default=str(DEFAULT_DEV_OUT))
    parser.add_argument("--report_json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--min_train_safe", type=int, default=250)
    parser.add_argument("--min_train_unsafe", type=int, default=350)
    parser.add_argument("--min_dev_safe", type=int, default=80)
    parser.add_argument("--min_dev_unsafe", type=int, default=120)
    parser.add_argument("--fallback_dev_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def main():
    args = parse_args()
    source_splits = parse_source_splits(args.source_splits)
    ledger_rows = load_records(Path(args.round5_ledger))
    probe_rows = load_records(Path(args.probe_mixed))
    teacher_keys = forbidden_texts([Path(args.teacher_test)])
    probe_keys = forbidden_texts([Path(args.probe_mixed)])

    ledger_exact_pool = ledger_pool(ledger_rows, source_splits)
    extra_pool, extra_source_counts = extra_exact_pool([Path(value) for value in args.extra_exact_pool])
    raw_pool = ledger_exact_pool + extra_pool
    pool, skipped = deduplicate(raw_pool, teacher_keys | probe_keys)
    pool_counts = count_by_safety(pool)
    dev_targets = {
        "safe_override": provisional_dev_target(
            pool_counts.get("safe_override", 0),
            args.min_train_safe,
            args.min_dev_safe,
            args.fallback_dev_fraction,
        ),
        "unsafe_override": provisional_dev_target(
            pool_counts.get("unsafe_override", 0),
            args.min_train_unsafe,
            args.min_dev_unsafe,
            args.fallback_dev_fraction,
        ),
    }
    train, dev = split_pool(pool, dev_targets, args.seed)

    save_jsonl(train, Path(args.train_output))
    save_jsonl(dev, Path(args.dev_output))
    leakage = leakage_report(train, dev, probe_rows, teacher_keys)
    targets = target_status(train, dev, args)
    ready = acceptance(targets, leakage)
    report = {
        "inputs": {
            "round5_ledger": str(Path(args.round5_ledger)),
            "source_splits": source_splits,
            "extra_exact_pool": [str(Path(value)) for value in args.extra_exact_pool],
            "probe_mixed_read_only": str(Path(args.probe_mixed)),
            "teacher_test_text_exclusion_only": str(Path(args.teacher_test)),
        },
        "outputs": {
            "train": str(Path(args.train_output)),
            "dev": str(Path(args.dev_output)),
            "report_json": str(Path(args.report_json)),
            "report_md": str(Path(args.report_md)),
        },
        "label_definition": {
            "label_1": "safe_override",
            "label_0": "unsafe_override",
            "original_detection_label": "original detector label, 1=LLM and 0=human",
        },
        "raw_counts": {
            "round5_ledger_rows": len(ledger_rows),
            "ledger_exact_rows_from_source_splits": len(ledger_exact_pool),
            "extra_exact_rows": len(extra_pool),
            "extra_exact_source_rows": extra_source_counts,
            "pool_after_dedup": len(pool),
            "pool_safety_distribution": dict(sorted(pool_counts.items())),
            "dedup_skipped": skipped,
            "read_only_probe_rows": len(probe_rows),
        },
        "dev_allocation_targets_for_short_pool": dev_targets,
        "summaries": {
            "pool": summarize(pool),
            "train": summarize(train),
            "dev": summarize(dev),
        },
        "bucket_watchlist": {
            "pool": bucket_watchlist(pool),
            "train": bucket_watchlist(train),
            "dev": bucket_watchlist(dev),
        },
        "leakage": leakage,
        "target_status": targets,
        "acceptance": ready,
    }
    report["decision"] = (
        "PROMOTE_TO_ROUND7_SELECTOR_TRAINING = yes"
        if ready["exact_selector_training_ready"]
        else "PROMOTE_TO_ROUND7_SELECTOR_TRAINING = no; existing exact candidates are short of Round7 train/dev targets, so add train-side exact-like safe/unsafe data before selector training."
    )
    write_json(report, Path(args.report_json))
    write_markdown(report, Path(args.report_md))

    print("=" * 70)
    print("Round7 exact-candidate dataset built")
    print("=" * 70)
    print(f"Candidate pool rows: {len(pool)} safety={dict(sorted(pool_counts.items()))}")
    print(f"Train rows: {len(train)}")
    print(f"Dev rows: {len(dev)}")
    print(f"Exact selector training ready: {ready['exact_selector_training_ready']}")
    print(f"Report: {args.report_json}")


if __name__ == "__main__":
    main()
