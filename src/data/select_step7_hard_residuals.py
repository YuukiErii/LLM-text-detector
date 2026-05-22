import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "predictions" / "residual_candidate_pool_v1_step7_predictions.jsonl"
DEFAULT_SELECTED = PROJECT_ROOT / "data" / "processed" / "residual_selected_v1.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "residual_train_v1.jsonl"
DEFAULT_DEV = PROJECT_ROOT / "data" / "processed" / "residual_dev_v1.jsonl"
DEFAULT_PROBE = PROJECT_ROOT / "data" / "processed" / "residual_probe_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "residual_split_v1_report.json"
DEFAULT_TEACHER = PROJECT_ROOT / "data" / "raw" / "teacher_test.json"


def load_records(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
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
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ["data", "samples", "records", "items"]:
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    raise ValueError(f"Unsupported input format: {path}")


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


def group_key(row: Dict, index: int) -> str:
    for key in ["split_group", "source_pair_id", "pair_id", "source_id", "source_record_id", "id"]:
        value = row.get(key)
        if value not in [None, ""]:
            return str(value)
    return f"group_{index:06d}"


def p_step7(row: Dict) -> float:
    return float(row.get("p_step7", row.get("probability", row.get("prob_llm", 0.0))))


def selection_tier(row: Dict) -> str:
    label = int(row.get("label"))
    prob = p_step7(row)
    ambiguous = 0.35 <= prob <= 0.65

    if label == 0:
        if prob >= 0.65:
            return "very_hard_human_fp"
        if prob >= 0.55:
            return "hard_human_fp"
        if ambiguous:
            return "ambiguous_human_support"
        return "human_support"

    if prob <= 0.35:
        return "very_hard_llm_fn"
    if prob <= 0.45:
        return "hard_llm_fn"
    if ambiguous:
        return "ambiguous_llm_support"
    return "llm_support"


def hardness_score(row: Dict) -> float:
    label = int(row.get("label"))
    prob = p_step7(row)
    if label == 0:
        return prob
    return 1.0 - prob


def row_priority(row: Dict) -> Tuple[int, float]:
    tier = selection_tier(row)
    tier_rank = {
        "very_hard_human_fp": 0,
        "very_hard_llm_fn": 0,
        "hard_human_fp": 1,
        "hard_llm_fn": 1,
        "ambiguous_human_support": 2,
        "ambiguous_llm_support": 2,
        "human_support": 3,
        "llm_support": 3,
    }.get(tier, 9)
    return tier_rank, -hardness_score(row)


def enrich_rows(rows: Sequence[Dict]) -> List[Dict]:
    enriched = []
    for index, row in enumerate(rows):
        if row.get("label") not in [0, 1]:
            continue
        if "p_step7" not in row and "probability" not in row:
            continue
        item = dict(row)
        item["selection_tier"] = selection_tier(item)
        item["hardness_score"] = hardness_score(item)
        item["hard_selection_core"] = item["selection_tier"] in {
            "very_hard_human_fp",
            "hard_human_fp",
            "very_hard_llm_fn",
            "hard_llm_fn",
        }
        item["hard_selection_support"] = not item["hard_selection_core"]
        item["residual_group"] = group_key(item, index)
        enriched.append(item)
    return enriched


def target_counts(train_total: int, dev_total: int, probe_total: int, llm_train_share: float) -> Dict[str, Dict[int, int]]:
    train_llm = round(train_total * llm_train_share)
    train_human = train_total - train_llm
    return {
        "train": {0: train_human, 1: train_llm},
        "dev": {0: dev_total // 2, 1: dev_total - dev_total // 2},
        "probe": {0: probe_total // 2, 1: probe_total - probe_total // 2},
    }


def select_groups(rows: Sequence[Dict], targets: Dict[str, Dict[int, int]]) -> Tuple[List[List[Dict]], Dict]:
    total_targets = Counter()
    for split_targets in targets.values():
        for label, count in split_targets.items():
            total_targets[label] += count

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["residual_group"]].append(row)

    group_list = list(grouped.values())
    group_list.sort(
        key=lambda group: (
            min(row_priority(row)[0] for row in group),
            min(row_priority(row)[1] for row in group),
            str(group[0][ "residual_group"]),
        )
    )

    selected = []
    counts = Counter()
    for group in group_list:
        labels = Counter(int(row["label"]) for row in group)
        helps = any(counts[label] < total_targets[label] for label in labels)
        if not helps:
            continue
        selected.append(group)
        counts.update(labels)
        if all(counts[label] >= total_targets[label] for label in total_targets):
            break

    return selected, {
        "target_counts": {str(label): count for label, count in sorted(total_targets.items())},
        "selected_counts": {str(label): count for label, count in sorted(counts.items())},
        "selected_groups": len(selected),
    }


def split_groups(groups: Sequence[List[Dict]], targets: Dict[str, Dict[int, int]], seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    sortable = list(groups)
    sortable.sort(key=lambda group: (min(row_priority(row)[0] for row in group), rng.random()))

    split_rows = {"train": [], "dev": [], "probe": []}
    split_counts = {"train": Counter(), "dev": Counter(), "probe": Counter()}

    def deficits(split_name: str) -> Counter:
        return Counter(
            {
                label: max(0, target - split_counts[split_name].get(label, 0))
                for label, target in targets[split_name].items()
            }
        )

    for group in sortable:
        labels = Counter(int(row["label"]) for row in group)
        best_split = None
        best_gain = -1
        for split_name in ["probe", "dev", "train"]:
            split_deficits = deficits(split_name)
            gain = sum(min(labels[label], split_deficits.get(label, 0)) for label in labels)
            if gain > best_gain:
                best_gain = gain
                best_split = split_name
        if best_split is None:
            best_split = "train"
        for row in group:
            item = dict(row)
            item["residual_split"] = best_split
            split_rows[best_split].append(item)
        split_counts[best_split].update(labels)

    return split_rows


def summarize(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {
            "num_rows": 0,
            "label_distribution": {},
        }
    return {
        "num_rows": len(rows),
        "num_groups": len({row.get("residual_group") for row in rows}),
        "label_distribution": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "selection_tier_distribution": dict(sorted(Counter(str(row.get("selection_tier")) for row in rows).items())),
        "round8_bucket_distribution": dict(sorted(Counter(str(row.get("round8_bucket")) for row in rows).items())),
        "round8_family_distribution": dict(sorted(Counter(str(row.get("round8_bucket_family")) for row in rows).items())),
        "domain_distribution": dict(sorted(Counter(str(row.get("domain")) for row in rows).items())),
        "generator_distribution": dict(sorted(Counter(str(row.get("generator")) for row in rows).items())),
        "step7_error_rows": sum(1 for row in rows if row.get("step7_correct") is False),
        "ambiguous_zone_rows": sum(1 for row in rows if row.get("ambiguous_zone")),
        "core_hard_rows": sum(1 for row in rows if row.get("hard_selection_core")),
    }


def leakage_report(splits: Dict[str, List[Dict]], teacher_texts: set) -> Dict:
    split_names = ["train", "dev", "probe"]
    groups = {name: {row.get("residual_group") for row in splits[name]} for name in split_names}
    texts = {name: {text_key(row) for row in splits[name] if text_key(row)} for name in split_names}
    overlaps = {}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlaps[f"{left}_{right}_group_overlap"] = len(groups[left] & groups[right])
            overlaps[f"{left}_{right}_text_overlap"] = len(texts[left] & texts[right])
    all_rows = splits["train"] + splits["dev"] + splits["probe"]
    overlaps["teacher_exact_text_duplicates"] = sum(1 for row in all_rows if text_key(row) in teacher_texts)
    return overlaps


def acceptance(splits: Dict[str, List[Dict]], leakage: Dict, args) -> Dict:
    train = splits["train"]
    dev = splits["dev"]
    probe = splits["probe"]

    def min_label_share(rows: Sequence[Dict]) -> float:
        counts = Counter(row.get("label") for row in rows)
        if not rows or not counts:
            return 0.0
        return min(counts.values()) / len(rows)

    checks = {
        "train_size_in_recommended_range": args.min_train_size <= len(train) <= args.max_train_size,
        "dev_size_in_recommended_range": args.min_dev_size <= len(dev) <= args.max_dev_size,
        "probe_size_in_recommended_range": args.min_probe_size <= len(probe) <= args.max_probe_size,
        "train_min_label_share_at_least_35pct": min_label_share(train) >= 0.35,
        "dev_min_label_share_at_least_45pct": min_label_share(dev) >= 0.45,
        "probe_min_label_share_at_least_45pct": min_label_share(probe) >= 0.45,
        "no_split_group_overlap": all(value == 0 for key, value in leakage.items() if key.endswith("_group_overlap")),
        "no_split_text_overlap": all(value == 0 for key, value in leakage.items() if key.endswith("_text_overlap")),
        "teacher_exact_duplicate_zero": leakage.get("teacher_exact_text_duplicates", 1) == 0,
    }
    return {
        "checks": checks,
        "ready_for_residual_mix_train_build": all(checks.values()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Select Step7-hard residuals and split them by group.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--selected_output", default=str(DEFAULT_SELECTED))
    parser.add_argument("--train_output", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev_output", default=str(DEFAULT_DEV))
    parser.add_argument("--probe_output", default=str(DEFAULT_PROBE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--teacher_test", default=str(DEFAULT_TEACHER))
    parser.add_argument("--train_size", type=int, default=3000)
    parser.add_argument("--dev_size", type=int, default=800)
    parser.add_argument("--probe_size", type=int, default=400)
    parser.add_argument("--train_llm_share", type=float, default=0.60)
    parser.add_argument("--min_train_size", type=int, default=3000)
    parser.add_argument("--max_train_size", type=int, default=6000)
    parser.add_argument("--min_dev_size", type=int, default=600)
    parser.add_argument("--max_dev_size", type=int, default=1000)
    parser.add_argument("--min_probe_size", type=int, default=300)
    parser.add_argument("--max_probe_size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = enrich_rows(load_records(Path(args.input)))
    teacher_rows = load_records(Path(args.teacher_test)) if Path(args.teacher_test).exists() else []
    teacher_texts = {text_key(row) for row in teacher_rows if text_key(row)}

    targets = target_counts(args.train_size, args.dev_size, args.probe_size, args.train_llm_share)
    selected_groups, selection_report = select_groups(rows, targets)
    splits = split_groups(selected_groups, targets, args.seed)

    selected_rows = []
    for split_name in ["train", "dev", "probe"]:
        selected_rows.extend(splits[split_name])

    save_jsonl(selected_rows, Path(args.selected_output))
    save_jsonl(splits["train"], Path(args.train_output))
    save_jsonl(splits["dev"], Path(args.dev_output))
    save_jsonl(splits["probe"], Path(args.probe_output))

    leakage = leakage_report(splits, teacher_texts)
    report = {
        "inputs": {
            "step7_predictions": str(Path(args.input)),
            "teacher_test_exact_text_exclusion_only": str(Path(args.teacher_test)),
        },
        "outputs": {
            "selected": str(Path(args.selected_output)),
            "train": str(Path(args.train_output)),
            "dev": str(Path(args.dev_output)),
            "probe": str(Path(args.probe_output)),
            "report": str(Path(args.report)),
        },
        "threshold_policy": {
            "hard_human": "label=0 and p_step7>=0.55",
            "very_hard_human": "label=0 and p_step7>=0.65",
            "hard_llm": "label=1 and p_step7<=0.45",
            "very_hard_llm": "label=1 and p_step7<=0.35",
            "ambiguous": "0.35<=p_step7<=0.65",
            "support_policy": "fill remaining target counts by hardness score from the same non-teacher candidate pool",
        },
        "target_counts": {
            split: {str(label): count for label, count in split_targets.items()}
            for split, split_targets in targets.items()
        },
        "selection": selection_report,
        "summaries": {
            "all_scored_candidates": summarize(rows),
            "selected": summarize(selected_rows),
            "train": summarize(splits["train"]),
            "dev": summarize(splits["dev"]),
            "probe": summarize(splits["probe"]),
        },
        "leakage": leakage,
    }
    report["acceptance"] = acceptance(splits, leakage, args)
    write_json(report, Path(args.report))

    print("=" * 70)
    print("Selected and split Round8 residual data")
    print("=" * 70)
    print(f"Selected rows: {len(selected_rows)} -> {args.selected_output}")
    print(f"Train rows: {len(splits['train'])} -> {args.train_output}")
    print(f"Dev rows: {len(splits['dev'])} -> {args.dev_output}")
    print(f"Probe rows: {len(splits['probe'])} -> {args.probe_output}")
    print(f"Report: {args.report}")
    print(f"Ready for residual mix train build: {report['acceptance']['ready_for_residual_mix_train_build']}")


if __name__ == "__main__":
    main()
