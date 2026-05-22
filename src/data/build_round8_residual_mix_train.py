import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos_poetry_expansion.jsonl"
)
DEFAULT_RESIDUAL_TRAIN = PROJECT_ROOT / "data" / "processed" / "residual_train_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_round8_residual_mix.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_round8_residual_mix_report.json"


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


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def text_key(row: Dict) -> str:
    return " ".join(str(row.get("text") or "").lower().split())


def valid_detection_row(row: Dict) -> bool:
    return row.get("label") in [0, 1] and isinstance(row.get("text"), str) and bool(row.get("text").strip())


def normalize_base(row: Dict, index: int) -> Dict:
    item = dict(row)
    item["id"] = str(item.get("id") or f"round8_base_{index:06d}")
    item["label"] = int(item["label"])
    item["round8_mix_source"] = "step7_base_train"
    item["round8_mix_role"] = "base_anchor"
    item.setdefault("sample_weight", 1.0)
    return item


def normalize_residual(row: Dict, index: int) -> Dict:
    item = dict(row)
    item["id"] = f"round8_residual_{index:06d}_{item.get('id', 'row')}"
    item["label"] = int(item["label"])
    item["round8_mix_source"] = "residual_train_v1"
    item["round8_mix_role"] = "residual_hard_or_support"
    item.setdefault("sample_weight", 1.0)
    return item


def stratified_sample(rows: Sequence[Dict], n: int, seed: int) -> List[Dict]:
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[int(row["label"])].append(row)
    label_counts = Counter(int(row["label"]) for row in rows)
    sampled = []
    for label in [0, 1]:
        target = round(n * label_counts[label] / len(rows)) if rows else 0
        bucket = list(by_label[label])
        rng.shuffle(bucket)
        sampled.extend(bucket[:target])
    if len(sampled) < n:
        used = {id(row) for row in sampled}
        remaining = [row for row in rows if id(row) not in used]
        rng.shuffle(remaining)
        sampled.extend(remaining[: n - len(sampled)])
    if len(sampled) > n:
        rng.shuffle(sampled)
        sampled = sampled[:n]
    return sampled


def deduplicate_mix(base_rows: Sequence[Dict], residual_rows: Sequence[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    mixed = []
    seen_texts = set()
    skipped = Counter()

    for row in residual_rows:
        key = text_key(row)
        if key in seen_texts:
            skipped["residual_duplicate_text"] += 1
            continue
        seen_texts.add(key)
        mixed.append(row)

    for row in base_rows:
        key = text_key(row)
        if key in seen_texts:
            skipped["base_duplicate_text_with_residual"] += 1
            continue
        seen_texts.add(key)
        mixed.append(row)

    return mixed, dict(sorted(skipped.items()))


def backfill_base_rows(
    mixed: List[Dict],
    all_base_rows: Sequence[Dict],
    target_base_count: int,
    seed: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    current_base_count = sum(1 for row in mixed if row.get("round8_mix_source") == "step7_base_train")
    if current_base_count >= target_base_count:
        return mixed, {"base_backfilled": 0}

    seen_texts = {text_key(row) for row in mixed}
    used_ids = {row.get("id") for row in mixed}
    remaining = [
        row
        for row in all_base_rows
        if row.get("id") not in used_ids and text_key(row) not in seen_texts
    ]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    added = 0
    for row in remaining:
        if current_base_count >= target_base_count:
            break
        mixed.append(row)
        seen_texts.add(text_key(row))
        used_ids.add(row.get("id"))
        current_base_count += 1
        added += 1

    return mixed, {
        "base_backfilled": added,
        "base_backfill_shortfall": max(0, target_base_count - current_base_count),
    }


def summarize(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {
            "num_rows": 0,
            "label_distribution": {},
        }
    return {
        "num_rows": len(rows),
        "label_distribution": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "mix_source_distribution": dict(sorted(Counter(str(row.get("round8_mix_source", "unknown")) for row in rows).items())),
        "round8_bucket_distribution": dict(sorted(Counter(str(row.get("round8_bucket", "base")) for row in rows).items())),
        "domain_distribution": dict(sorted(Counter(str(row.get("domain", "unknown")) for row in rows).items())),
        "generator_distribution": dict(sorted(Counter(str(row.get("generator", "unknown")) for row in rows).items())),
        "selection_tier_distribution": dict(sorted(Counter(str(row.get("selection_tier", "base")) for row in rows).items())),
    }


def acceptance(mixed: Sequence[Dict], base_sample: Sequence[Dict], residual_rows: Sequence[Dict], target_ratio: float) -> Dict:
    total = len(mixed)
    residual_count = sum(1 for row in mixed if row.get("round8_mix_source") == "residual_train_v1")
    observed_ratio = residual_count / total if total else 0.0
    labels = Counter(int(row["label"]) for row in mixed)
    min_label_share = min(labels.values()) / total if labels and total else 0.0
    checks = {
        "residual_ratio_within_2pct": abs(observed_ratio - target_ratio) <= 0.02,
        "base_rows_present": len(base_sample) > 0,
        "residual_rows_present": len(residual_rows) > 0,
        "min_label_share_at_least_40pct": min_label_share >= 0.40,
        "total_rows_positive": total > 0,
    }
    return {
        "checks": checks,
        "ready_for_residual_deberta_training": all(checks.values()),
        "observed_residual_ratio": observed_ratio,
        "target_residual_ratio": target_ratio,
        "min_label_share": min_label_share,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build Round8 70/30 base/residual training mix.")
    parser.add_argument("--base_train", default=str(DEFAULT_BASE_TRAIN))
    parser.add_argument("--residual_train", default=str(DEFAULT_RESIDUAL_TRAIN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--residual_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = Path(args.base_train)
    residual_path = Path(args.residual_train)
    if not base_path.exists():
        raise FileNotFoundError(f"Cannot find base train: {base_path}")
    if not residual_path.exists():
        raise FileNotFoundError(f"Cannot find residual train: {residual_path}")

    raw_base = [row for row in load_jsonl(base_path) if valid_detection_row(row)]
    raw_residual = [row for row in load_jsonl(residual_path) if valid_detection_row(row)]

    residual_rows = [normalize_residual(row, index) for index, row in enumerate(raw_residual)]
    base_all = [normalize_base(row, index) for index, row in enumerate(raw_base)]
    base_needed = round(len(residual_rows) * (1.0 - args.residual_ratio) / args.residual_ratio)
    base_sample = stratified_sample(base_all, base_needed, args.seed)
    mixed, skipped = deduplicate_mix(base_sample, residual_rows)
    mixed, backfill_report = backfill_base_rows(mixed, base_all, base_needed, args.seed + 1)

    rng = random.Random(args.seed)
    rng.shuffle(mixed)
    save_jsonl(mixed, Path(args.output))

    report = {
        "inputs": {
            "base_train": str(base_path),
            "residual_train": str(residual_path),
        },
        "outputs": {
            "train_mix": str(Path(args.output)),
            "report": str(Path(args.report)),
        },
        "config": {
            "residual_ratio": args.residual_ratio,
            "seed": args.seed,
            "base_needed_before_dedup": base_needed,
        },
        "raw_counts": {
            "base_rows": len(raw_base),
            "residual_rows": len(raw_residual),
        },
        "skipped": {
            **skipped,
            **backfill_report,
        },
        "summaries": {
            "base_sample": summarize(base_sample),
            "residual_train": summarize(residual_rows),
            "mixed_train": summarize(mixed),
        },
    }
    report["acceptance"] = acceptance(mixed, base_sample, residual_rows, args.residual_ratio)
    write_json(report, Path(args.report))

    print("=" * 70)
    print("Built Round8 residual mix train")
    print("=" * 70)
    print(f"Rows: {len(mixed)} -> {args.output}")
    print(f"Residual ratio: {report['acceptance']['observed_residual_ratio']:.4f}")
    print(f"Report: {args.report}")
    print(f"Ready for DeBERTa training: {report['acceptance']['ready_for_residual_deberta_training']}")


if __name__ == "__main__":
    main()
