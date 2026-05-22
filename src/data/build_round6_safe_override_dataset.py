import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


DEFAULT_LEDGER = PROJECT_ROOT / "outputs" / "evaluation" / "round5_flip_ledger.jsonl"
DEFAULT_ROUND4_TRAIN = PROJECT_ROOT / "data" / "processed" / "round4_residual_train.jsonl"
DEFAULT_ROUND4_HARDPOS = PROJECT_ROOT / "data" / "processed" / "round4_residual_dev_hardpos.jsonl"
DEFAULT_ROUND4_HARDNEG = PROJECT_ROOT / "data" / "processed" / "round4_residual_dev_hardneg.jsonl"
DEFAULT_VALID = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_valid.jsonl"
DEFAULT_INTERNAL = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_internal_test.jsonl"
DEFAULT_TEACHER = PROJECT_ROOT / "data" / "raw" / "teacher_test.json"

DEFAULT_TRAIN_OUT = PROJECT_ROOT / "data" / "processed" / "round6_override_train.jsonl"
DEFAULT_DEV_SAFE_OUT = PROJECT_ROOT / "data" / "processed" / "round6_override_dev_safe.jsonl"
DEFAULT_DEV_UNSAFE_OUT = PROJECT_ROOT / "data" / "processed" / "round6_override_dev_unsafe.jsonl"
DEFAULT_PROBE_OUT = PROJECT_ROOT / "data" / "processed" / "round6_override_probe_mixed.jsonl"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "data" / "processed" / "round6_override_dataset_report.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "data" / "processed" / "round6_override_dataset_report.md"


SAFE_LABEL = 1
UNSAFE_LABEL = 0


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def text_key(row: Dict) -> str:
    return " ".join(str(row.get("text", "")).lower().split())


def sample_id(row: Dict, index: int = 0) -> str:
    value = row.get("id")
    return str(value) if value not in [None, ""] else f"row_{index:06d}"


def group_key(row: Dict, index: int = 0) -> str:
    for key in ["pair_id", "source_pair_id", "source_id", "original_id", "id"]:
        value = row.get(key)
        if value not in [None, ""]:
            return str(value)
    return sample_id(row, index)


def word_count(text: str) -> int:
    return len(str(text or "").split())


def is_residual_proxy(row: Dict) -> bool:
    tag = str(row.get("round4_tag", ""))
    stage = str(row.get("round4_source_stage", ""))
    if tag.startswith("base_"):
        return False
    return stage not in ["", "step7_base_train"]


def selector_label_for_safety(safety: str) -> int:
    if safety == "safe_override":
        return SAFE_LABEL
    if safety == "unsafe_override":
        return UNSAFE_LABEL
    raise ValueError(f"Unknown safety label: {safety}")


def safety_for_detection_label(label: int) -> str:
    return "safe_override" if int(label) == 1 else "unsafe_override"


def label_for_detection_label(label: int) -> int:
    return SAFE_LABEL if int(label) == 1 else UNSAFE_LABEL


def numeric_or_none(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_candidate(
    row: Dict,
    selector_label: int,
    safety: str,
    source_kind: str,
    origin_split: str,
    index: int,
    exact_row: Optional[Dict] = None,
) -> Dict:
    base = dict(exact_row or row)
    text = str(base.get("text") or row.get("text") or "").strip()
    bucket = str(base.get("bucket") or row.get("bucket") or assign_bucket(text))
    round4_bucket = str(base.get("round4_bucket") or row.get("round4_bucket") or bucket)
    original_detection_label = int((exact_row or row).get("label", row.get("label", -1)))
    features = text_features(text)

    item = dict(row)
    item.update(base)
    item["id"] = sample_id(base or row, index)
    item["text"] = text
    item["label"] = int(selector_label)
    item["target"] = int(selector_label)
    item["safe_selector_label"] = int(selector_label)
    item["positive_label_meaning"] = "safe_override"
    item["original_detection_label"] = original_detection_label
    item["override_safety"] = safety
    item["round6_source_kind"] = source_kind
    item["round6_origin_split"] = origin_split
    item["round6_group_key"] = group_key(base or row, index)
    item["round6_text_key"] = text_key({"text": text})
    item["is_exact_override_candidate"] = bool(source_kind == "exact_candidate" or exact_row is not None)
    item["bucket"] = bucket
    item["round4_bucket"] = round4_bucket
    item["word_count"] = int(base.get("word_count") or word_count(text))
    item["step7_pred"] = int(base.get("step7_pred", 0) or 0)
    item["round4_pred"] = int(base.get("round4_pred", 1) or 1)
    item["step7_prob"] = numeric_or_none(base.get("step7_prob"))
    item["round4_prob"] = numeric_or_none(base.get("round4_prob"))
    item["prob_delta"] = numeric_or_none(base.get("prob_delta"))
    item["guard_p_human_style"] = numeric_or_none(base.get("guard_p_human_style"))
    item["guard_human_style_veto"] = int(base.get("guard_human_style_veto", 0) or 0)
    item["source_stage"] = str(row.get("round4_source_stage") or base.get("round4_source_stage") or "unknown")
    item["source_tag"] = str(row.get("round4_tag") or base.get("round4_tag") or "unknown")

    for key, value in features.items():
        if key != "bucket":
            item.setdefault(key, value)
    return item


def exact_candidate_rows(ledger_rows: Sequence[Dict], split: Optional[str] = None) -> List[Dict]:
    out = []
    for row in ledger_rows:
        if split and row.get("split") != split:
            continue
        if row.get("flip_type") not in ["fixed_fn_candidate", "induced_fp"]:
            continue
        safety = "safe_override" if row["flip_type"] == "fixed_fn_candidate" else "unsafe_override"
        out.append(
            normalize_candidate(
                row,
                selector_label_for_safety(safety),
                safety=safety,
                source_kind="exact_candidate",
                origin_split=str(row.get("split") or "unknown"),
                index=len(out),
                exact_row=row,
            )
        )
    return out


def forbidden_texts(paths: Sequence[Path]) -> set:
    out = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_records(path):
            if row.get("text"):
                out.add(text_key(row))
    return out


def make_proxy_pool(
    train_rows: Sequence[Dict],
    hardpos_rows: Sequence[Dict],
    hardneg_rows: Sequence[Dict],
    exact_by_id: Dict[str, Dict],
) -> List[Dict]:
    pool = []
    for idx, row in enumerate(train_rows):
        if row.get("label") not in [0, 1]:
            continue
        if not is_residual_proxy(row):
            continue
        safety = safety_for_detection_label(int(row["label"]))
        exact = exact_by_id.get(str(row.get("id")))
        pool.append(
            normalize_candidate(
                row,
                label_for_detection_label(int(row["label"])),
                safety=safety,
                source_kind="train_proxy",
                origin_split="round4_residual_train",
                index=idx,
                exact_row=exact,
            )
        )

    for idx, row in enumerate(hardpos_rows):
        exact = exact_by_id.get(str(row.get("id")))
        pool.append(
            normalize_candidate(
                row,
                SAFE_LABEL,
                safety="safe_override",
                source_kind="hardpos_proxy",
                origin_split="round4_residual_dev_hardpos",
                index=idx,
                exact_row=exact,
            )
        )

    for idx, row in enumerate(hardneg_rows):
        exact = exact_by_id.get(str(row.get("id")))
        pool.append(
            normalize_candidate(
                row,
                UNSAFE_LABEL,
                safety="unsafe_override",
                source_kind="hardneg_proxy",
                origin_split="round4_residual_dev_hardneg",
                index=idx,
                exact_row=exact,
            )
        )
    return pool


def dedup_pool(rows: Sequence[Dict], forbidden_text_set: set) -> Tuple[List[Dict], Counter]:
    kept = []
    seen_ids = set()
    seen_texts = set()
    skipped = Counter()
    for row in rows:
        if not str(row.get("text", "")).strip():
            skipped["empty_text"] += 1
            continue
        if row["round6_text_key"] in forbidden_text_set:
            skipped["forbidden_text"] += 1
            continue
        if row["id"] in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        if row["round6_text_key"] in seen_texts:
            skipped["duplicate_text"] += 1
            continue
        seen_ids.add(row["id"])
        seen_texts.add(row["round6_text_key"])
        kept.append(row)
    return kept, skipped


def select_rows_by_bucket(
    pool: Sequence[Dict],
    label: int,
    target: int,
    bucket_minimums: Dict[str, int],
    exact_dev_limit: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    candidates = [row for row in pool if int(row["label"]) == label]
    exact = [row for row in candidates if row.get("is_exact_override_candidate")]
    proxy = [row for row in candidates if not row.get("is_exact_override_candidate")]

    selected = []
    used_groups = set()
    used_texts = set()

    def add(row: Dict) -> bool:
        group = row["round6_group_key"]
        text = row["round6_text_key"]
        if group in used_groups or text in used_texts:
            return False
        selected.append(row)
        used_groups.add(group)
        used_texts.add(text)
        return True

    exact_added = 0
    for row in sorted(exact, key=lambda item: (item.get("round4_bucket", ""), item.get("id", ""))):
        if exact_dev_limit >= 0 and exact_added >= exact_dev_limit:
            break
        add(row)
        exact_added += 1

    by_bucket = defaultdict(list)
    for row in proxy:
        by_bucket[str(row.get("round4_bucket") or row.get("bucket") or "unknown")].append(row)

    for bucket, minimum in bucket_minimums.items():
        rows = list(by_bucket.get(bucket, []))
        rng.shuffle(rows)
        for row in rows:
            if sum(1 for item in selected if item.get("round4_bucket") == bucket) >= minimum:
                break
            add(row)

    remaining = [row for row in proxy if row["round6_group_key"] not in used_groups and row["round6_text_key"] not in used_texts]
    remaining.sort(
        key=lambda row: (
            0 if row.get("round6_origin_split", "").startswith("round4_residual_dev") else 1,
            str(row.get("round4_bucket", "")),
            str(row.get("id", "")),
        )
    )
    for row in remaining:
        if len(selected) >= target:
            break
        add(row)
    return selected


def split_train_dev(
    pool: Sequence[Dict],
    safe_dev_target: int,
    unsafe_dev_target: int,
    safe_dev_bucket_minimums: Dict[str, int],
    safe_exact_dev_limit: int,
    unsafe_exact_dev_limit: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    dev_safe = select_rows_by_bucket(
        pool,
        label=SAFE_LABEL,
        target=safe_dev_target,
        bucket_minimums=safe_dev_bucket_minimums,
        exact_dev_limit=safe_exact_dev_limit,
        seed=seed,
    )
    dev_groups = {row["round6_group_key"] for row in dev_safe}
    dev_texts = {row["round6_text_key"] for row in dev_safe}
    unsafe_pool = [
        row for row in pool if row["round6_group_key"] not in dev_groups and row["round6_text_key"] not in dev_texts
    ]
    dev_unsafe = select_rows_by_bucket(
        unsafe_pool,
        label=UNSAFE_LABEL,
        target=unsafe_dev_target,
        bucket_minimums={},
        exact_dev_limit=unsafe_exact_dev_limit,
        seed=seed + 1,
    )
    dev_groups.update(row["round6_group_key"] for row in dev_unsafe)
    dev_texts.update(row["round6_text_key"] for row in dev_unsafe)

    train = [
        row
        for row in pool
        if row["round6_group_key"] not in dev_groups and row["round6_text_key"] not in dev_texts
    ]
    return train, dev_safe, dev_unsafe


def summarize(rows: Sequence[Dict]) -> Dict:
    return {
        "num_rows": len(rows),
        "selector_label_distribution": dict(Counter(str(row.get("label")) for row in rows)),
        "override_safety_distribution": dict(Counter(str(row.get("override_safety")) for row in rows)),
        "round6_source_kind_distribution": dict(Counter(str(row.get("round6_source_kind")) for row in rows)),
        "origin_split_distribution": dict(Counter(str(row.get("round6_origin_split")) for row in rows)),
        "round4_bucket_distribution": dict(Counter(str(row.get("round4_bucket")) for row in rows)),
        "exact_override_candidates": sum(1 for row in rows if row.get("is_exact_override_candidate")),
    }


def leakage_report(train: Sequence[Dict], dev_safe: Sequence[Dict], dev_unsafe: Sequence[Dict], probe: Sequence[Dict], teacher_texts: set) -> Dict:
    dev = list(dev_safe) + list(dev_unsafe)
    train_groups = {row["round6_group_key"] for row in train}
    dev_groups = {row["round6_group_key"] for row in dev}
    train_texts = {row["round6_text_key"] for row in train}
    dev_texts = {row["round6_text_key"] for row in dev}
    all_rows = list(train) + list(dev) + list(probe)
    return {
        "train_dev_group_overlap": len(train_groups & dev_groups),
        "train_dev_text_overlap": len(train_texts & dev_texts),
        "teacher_exact_text_duplicates": sum(1 for row in all_rows if row["round6_text_key"] in teacher_texts),
        "probe_in_train_text_overlap": len({row["round6_text_key"] for row in probe} & train_texts),
        "probe_in_dev_text_overlap": len({row["round6_text_key"] for row in probe} & dev_texts),
    }


def acceptance(report: Dict, args) -> Dict:
    train_counts = Counter(row.get("override_safety") for row in report["_train_rows"])
    dev_safe = report["_dev_safe_rows"]
    dev_unsafe = report["_dev_unsafe_rows"]
    safe_dev_buckets = Counter(str(row.get("round4_bucket")) for row in dev_safe)
    leakage = report["leakage"]
    checks = {
        "safe_train_at_least_target": train_counts.get("safe_override", 0) >= args.min_safe_train,
        "unsafe_train_at_least_target": train_counts.get("unsafe_override", 0) >= args.min_unsafe_train,
        "safe_dev_at_least_target": len(dev_safe) >= args.min_safe_dev,
        "unsafe_dev_at_least_target": len(dev_unsafe) >= args.min_unsafe_dev,
        "general_prose_safe_dev_at_least_target": safe_dev_buckets.get("general_prose", 0) >= args.min_general_safe_dev,
        "short_fragment_safe_dev_at_least_target": safe_dev_buckets.get("literary_short_fragment", 0) >= args.min_short_safe_dev,
        "train_dev_group_leakage_zero": leakage["train_dev_group_overlap"] == 0,
        "train_dev_text_leakage_zero": leakage["train_dev_text_overlap"] == 0,
        "teacher_exact_duplicate_zero": leakage["teacher_exact_text_duplicates"] == 0,
    }
    return {
        "checks": checks,
        "selector_training_ready": all(checks.values()),
        "train_safe_rows": train_counts.get("safe_override", 0),
        "train_unsafe_rows": train_counts.get("unsafe_override", 0),
        "safe_dev_rows": len(dev_safe),
        "unsafe_dev_rows": len(dev_unsafe),
        "safe_dev_general_prose_rows": safe_dev_buckets.get("general_prose", 0),
        "safe_dev_literary_short_fragment_rows": safe_dev_buckets.get("literary_short_fragment", 0),
    }


def write_markdown(report: Dict, path: Path) -> None:
    acc = report["acceptance"]
    lines = [
        "# Round6 Override Dataset Report",
        "",
        "This dataset is non-teacher only. Teacher-test text is used only for exact-duplicate leakage checks.",
        "",
        "## Acceptance",
        "",
        f"selector_training_ready: `{acc['selector_training_ready']}`",
        "",
        "| Check | Pass |",
        "| --- | --- |",
    ]
    for key, value in acc["checks"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Row Counts",
            "",
            "| Split | Rows | Safe | Unsafe | Exact candidates |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ["train", "dev_safe", "dev_unsafe", "probe_mixed"]:
        block = report["summaries"][name]
        safety = block["override_safety_distribution"]
        lines.append(
            f"| {name} | {block['num_rows']} | {safety.get('safe_override', 0)} | "
            f"{safety.get('unsafe_override', 0)} | {block['exact_override_candidates']} |"
        )

    lines.extend(["", "## Safe Dev Bucket Distribution", ""])
    lines.append("| Round4 bucket | Rows |")
    lines.append("| --- | ---: |")
    for bucket, count in sorted(report["summaries"]["dev_safe"]["round4_bucket_distribution"].items()):
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Leakage",
            "",
            "| Item | Count |",
            "| --- | ---: |",
        ]
    )
    for key, value in report["leakage"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "```text",
            report["decision"],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Round6 non-teacher safe/unsafe override dataset.")
    parser.add_argument("--round5_ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--round4_train", default=str(DEFAULT_ROUND4_TRAIN))
    parser.add_argument("--round4_hardpos", default=str(DEFAULT_ROUND4_HARDPOS))
    parser.add_argument("--round4_hardneg", default=str(DEFAULT_ROUND4_HARDNEG))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--internal_test", default=str(DEFAULT_INTERNAL))
    parser.add_argument("--teacher_test", default=str(DEFAULT_TEACHER))
    parser.add_argument("--train_output", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--dev_safe_output", default=str(DEFAULT_DEV_SAFE_OUT))
    parser.add_argument("--dev_unsafe_output", default=str(DEFAULT_DEV_UNSAFE_OUT))
    parser.add_argument("--probe_output", default=str(DEFAULT_PROBE_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_OUT))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--safe_dev_target", type=int, default=160)
    parser.add_argument("--unsafe_dev_target", type=int, default=240)
    parser.add_argument("--safe_exact_dev_limit", type=int, default=44)
    parser.add_argument("--unsafe_exact_dev_limit", type=int, default=16)
    parser.add_argument("--min_safe_train", type=int, default=300)
    parser.add_argument("--min_unsafe_train", type=int, default=600)
    parser.add_argument("--min_safe_dev", type=int, default=100)
    parser.add_argument("--min_unsafe_dev", type=int, default=200)
    parser.add_argument("--min_general_safe_dev", type=int, default=40)
    parser.add_argument("--min_short_safe_dev", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def main():
    args = parse_args()
    ledger_rows = load_records(Path(args.round5_ledger))
    round4_train = load_records(Path(args.round4_train))
    hardpos = load_records(Path(args.round4_hardpos))
    hardneg = load_records(Path(args.round4_hardneg))
    teacher_texts = forbidden_texts([Path(args.teacher_test)])
    heldout_texts = forbidden_texts([Path(args.valid), Path(args.internal_test), Path(args.teacher_test)])

    exact_candidates = exact_candidate_rows(ledger_rows)
    exact_by_id = {row["id"]: row for row in exact_candidates}
    probe_mixed = [row for row in exact_candidates if row.get("round6_origin_split") == "internal_test"]

    pool = make_proxy_pool(round4_train, hardpos, hardneg, exact_by_id)
    probe_texts = {row["round6_text_key"] for row in probe_mixed}
    pool, skipped = dedup_pool(pool, heldout_texts | probe_texts)
    train, dev_safe, dev_unsafe = split_train_dev(
        pool,
        safe_dev_target=args.safe_dev_target,
        unsafe_dev_target=args.unsafe_dev_target,
        safe_dev_bucket_minimums={
            "general_prose": args.min_general_safe_dev,
            "literary_short_fragment": args.min_short_safe_dev,
        },
        safe_exact_dev_limit=args.safe_exact_dev_limit,
        unsafe_exact_dev_limit=args.unsafe_exact_dev_limit,
        seed=args.seed,
    )

    save_jsonl(train, Path(args.train_output))
    save_jsonl(dev_safe, Path(args.dev_safe_output))
    save_jsonl(dev_unsafe, Path(args.dev_unsafe_output))
    save_jsonl(probe_mixed, Path(args.probe_output))

    report = {
        "inputs": {
            "round5_ledger": str(Path(args.round5_ledger)),
            "round4_train": str(Path(args.round4_train)),
            "round4_hardpos": str(Path(args.round4_hardpos)),
            "round4_hardneg": str(Path(args.round4_hardneg)),
            "valid_text_exclusion": str(Path(args.valid)),
            "internal_text_exclusion": str(Path(args.internal_test)),
            "teacher_text_exclusion_only": str(Path(args.teacher_test)),
        },
        "outputs": {
            "train": str(Path(args.train_output)),
            "dev_safe": str(Path(args.dev_safe_output)),
            "dev_unsafe": str(Path(args.dev_unsafe_output)),
            "probe_mixed": str(Path(args.probe_output)),
            "report": str(Path(args.report)),
            "report_md": str(Path(args.report_md)),
        },
        "label_definition": {
            "label_1": "safe_override",
            "label_0": "unsafe_override",
            "original_detection_label": "original detector label, 1=LLM and 0=human",
        },
        "raw_counts": {
            "ledger_rows": len(ledger_rows),
            "exact_override_candidates": len(exact_candidates),
            "round4_train_rows": len(round4_train),
            "round4_hardpos_rows": len(hardpos),
            "round4_hardneg_rows": len(hardneg),
            "candidate_pool_after_dedup": len(pool),
            "skipped": dict(skipped),
        },
        "summaries": {
            "train": summarize(train),
            "dev_safe": summarize(dev_safe),
            "dev_unsafe": summarize(dev_unsafe),
            "probe_mixed": summarize(probe_mixed),
        },
        "_train_rows": train,
        "_dev_safe_rows": dev_safe,
        "_dev_unsafe_rows": dev_unsafe,
        "leakage": leakage_report(train, dev_safe, dev_unsafe, probe_mixed, teacher_texts),
    }
    report["acceptance"] = acceptance(report, args)
    report["decision"] = (
        "PROMOTE_TO_SELECTOR_TRAINING = yes"
        if report["acceptance"]["selector_training_ready"]
        else "PROMOTE_TO_SELECTOR_TRAINING = no; fill the failed data buckets before training."
    )
    del report["_train_rows"]
    del report["_dev_safe_rows"]
    del report["_dev_unsafe_rows"]

    write_json(report, Path(args.report))
    write_markdown(report, Path(args.report_md))

    print("=" * 70)
    print("Round6 safe/unsafe override dataset built")
    print("=" * 70)
    print(f"Train rows: {report['summaries']['train']['num_rows']}")
    print(f"Dev safe rows: {report['summaries']['dev_safe']['num_rows']}")
    print(f"Dev unsafe rows: {report['summaries']['dev_unsafe']['num_rows']}")
    print(f"Probe mixed rows: {report['summaries']['probe_mixed']['num_rows']}")
    print(f"Selector training ready: {report['acceptance']['selector_training_ready']}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
