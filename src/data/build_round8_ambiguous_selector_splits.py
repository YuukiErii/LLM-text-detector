import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "predictions" / "residual_candidate_pool_v1_step7_predictions.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "round8_ambiguous_selector_train.jsonl"
DEFAULT_DEV = PROJECT_ROOT / "data" / "processed" / "round8_ambiguous_selector_dev.jsonl"
DEFAULT_PROBE = PROJECT_ROOT / "data" / "processed" / "round8_ambiguous_selector_probe.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "round8_ambiguous_selector_split_report.json"
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
    return f"ambiguous_group_{index:06d}"


def probability(row: Dict) -> float:
    for key in ["p_step7", "probability", "prob_llm"]:
        value = row.get(key)
        if value not in [None, ""]:
            return float(value)
    raise ValueError(f"Missing Step7 probability for row {row.get('id')}")


def target_counts(total_counts: Counter, train_share: float, dev_share: float) -> Dict[str, Counter]:
    targets = {"train": Counter(), "dev": Counter(), "probe": Counter()}
    for label, count in sorted(total_counts.items()):
        train_count = round(count * train_share)
        dev_count = round(count * dev_share)
        probe_count = count - train_count - dev_count
        targets["train"][label] = train_count
        targets["dev"][label] = dev_count
        targets["probe"][label] = probe_count
    return targets


def prepare_rows(rows: Sequence[Dict], low: float, high: float) -> List[Dict]:
    out = []
    for index, row in enumerate(rows):
        if row.get("label") not in [0, 1]:
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        p_step7 = probability(row)
        if not (low <= p_step7 <= high):
            continue
        item = dict(row)
        item["p_step7"] = float(p_step7)
        item["step7_pred"] = int(item.get("step7_pred", item.get("prediction", p_step7 >= 0.55)))
        item["ambiguous_zone"] = True
        item["ambiguous_selector_group"] = group_key(item, index)
        item["ambiguous_selector_label"] = int(item["label"])
        item["ambiguous_selector_source"] = "round8_candidate_pool_step7_ambiguous"
        out.append(item)
    return out


def split_grouped_rows(rows: Sequence[Dict], targets: Dict[str, Counter], seed: int) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["ambiguous_selector_group"])].append(row)

    rng = random.Random(seed)
    groups = list(grouped.values())
    groups.sort(
        key=lambda group: (
            # Place mixed-label groups first so paired evidence stays intact.
            -len({int(row["label"]) for row in group}),
            -len(group),
            rng.random(),
            str(group[0]["ambiguous_selector_group"]),
        )
    )

    split_rows = {"train": [], "dev": [], "probe": []}
    split_counts = {"train": Counter(), "dev": Counter(), "probe": Counter()}

    def score_split(split_name: str, labels: Counter) -> tuple:
        deficits = Counter(
            {
                label: max(0, target - split_counts[split_name].get(label, 0))
                for label, target in targets[split_name].items()
            }
        )
        useful = sum(min(labels[label], deficits.get(label, 0)) for label in labels)
        overflow = sum(max(0, split_counts[split_name].get(label, 0) + labels[label] - targets[split_name][label]) for label in labels)
        current_rows = len(split_rows[split_name])
        target_rows = sum(targets[split_name].values())
        row_deficit = max(0, target_rows - current_rows)
        priority = {"dev": 2, "probe": 1, "train": 0}[split_name]
        return useful, row_deficit, -overflow, priority

    for group in groups:
        labels = Counter(int(row["label"]) for row in group)
        split_name = max(["train", "dev", "probe"], key=lambda name: score_split(name, labels))
        for row in group:
            item = dict(row)
            item["ambiguous_selector_split"] = split_name
            split_rows[split_name].append(item)
        split_counts[split_name].update(labels)

    return split_rows


def summarize(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {"num_rows": 0, "label_distribution": {}}
    return {
        "num_rows": len(rows),
        "num_groups": len({row.get("ambiguous_selector_group") for row in rows}),
        "label_distribution": dict(sorted(Counter(str(row.get("label")) for row in rows).items())),
        "step7_prediction_distribution": dict(sorted(Counter(str(row.get("step7_pred")) for row in rows).items())),
        "selection_tier_distribution": dict(sorted(Counter(str(row.get("selection_tier")) for row in rows).items())),
        "round8_bucket_distribution": dict(sorted(Counter(str(row.get("round8_bucket")) for row in rows).items())),
        "round8_family_distribution": dict(sorted(Counter(str(row.get("round8_bucket_family")) for row in rows).items())),
        "domain_distribution": dict(sorted(Counter(str(row.get("domain")) for row in rows).items())),
        "generator_distribution": dict(sorted(Counter(str(row.get("generator")) for row in rows).items())),
        "step7_error_rows": sum(1 for row in rows if row.get("step7_correct") is False),
        "mean_p_step7": sum(float(row["p_step7"]) for row in rows) / len(rows),
    }


def leakage_report(splits: Dict[str, List[Dict]], teacher_texts: set) -> Dict:
    names = ["train", "dev", "probe"]
    groups = {name: {row.get("ambiguous_selector_group") for row in splits[name]} for name in names}
    texts = {name: {text_key(row) for row in splits[name] if text_key(row)} for name in names}
    out = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            out[f"{left}_{right}_group_overlap"] = len(groups[left] & groups[right])
            out[f"{left}_{right}_text_overlap"] = len(texts[left] & texts[right])
    all_rows = splits["train"] + splits["dev"] + splits["probe"]
    out["teacher_exact_text_duplicates"] = sum(1 for row in all_rows if text_key(row) in teacher_texts)
    return out


def acceptance(splits: Dict[str, List[Dict]], leakage: Dict, low: float, high: float) -> Dict:
    def min_label_count(rows: Sequence[Dict]) -> int:
        counts = Counter(int(row["label"]) for row in rows)
        return min(counts.values()) if len(counts) == 2 else 0

    def all_ambiguous(rows: Sequence[Dict]) -> bool:
        return all(low <= float(row["p_step7"]) <= high for row in rows)

    checks = {
        "all_rows_are_step7_ambiguous": all(all_ambiguous(splits[name]) for name in splits),
        "train_has_both_labels": min_label_count(splits["train"]) > 0,
        "dev_min_label_count_at_least_15": min_label_count(splits["dev"]) >= 15,
        "probe_min_label_count_at_least_15": min_label_count(splits["probe"]) >= 15,
        "no_split_group_overlap": all(value == 0 for key, value in leakage.items() if key.endswith("_group_overlap")),
        "no_split_text_overlap": all(value == 0 for key, value in leakage.items() if key.endswith("_text_overlap")),
        "teacher_exact_duplicate_zero": leakage.get("teacher_exact_text_duplicates", 1) == 0,
    }
    return {
        "checks": checks,
        "ready_for_branch_scoring": all(checks.values()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build dedicated Round8 ambiguous selector train/dev/probe splits.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--train_output", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev_output", default=str(DEFAULT_DEV))
    parser.add_argument("--probe_output", default=str(DEFAULT_PROBE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--teacher_test", default=str(DEFAULT_TEACHER))
    parser.add_argument("--ambiguous_low", type=float, default=0.35)
    parser.add_argument("--ambiguous_high", type=float, default=0.65)
    parser.add_argument("--train_share", type=float, default=0.60)
    parser.add_argument("--dev_share", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    all_rows = load_records(input_path)
    rows = prepare_rows(all_rows, low=args.ambiguous_low, high=args.ambiguous_high)
    if not rows:
        raise ValueError(f"No ambiguous selector rows found in {input_path}")

    label_counts = Counter(int(row["label"]) for row in rows)
    targets = target_counts(label_counts, train_share=args.train_share, dev_share=args.dev_share)
    splits = split_grouped_rows(rows, targets=targets, seed=args.seed)

    teacher_rows = load_records(Path(args.teacher_test)) if Path(args.teacher_test).exists() else []
    teacher_texts = {text_key(row) for row in teacher_rows if text_key(row)}
    leakage = leakage_report(splits, teacher_texts)

    save_jsonl(splits["train"], Path(args.train_output))
    save_jsonl(splits["dev"], Path(args.dev_output))
    save_jsonl(splits["probe"], Path(args.probe_output))

    report = {
        "inputs": {
            "step7_predictions": str(input_path),
            "teacher_test_exact_text_exclusion_only": str(Path(args.teacher_test)),
        },
        "outputs": {
            "train": str(Path(args.train_output)),
            "dev": str(Path(args.dev_output)),
            "probe": str(Path(args.probe_output)),
            "report": str(Path(args.report)),
        },
        "policy": {
            "source": "all non-teacher candidate-pool rows with Step7 probability inside the ambiguous zone",
            "ambiguous_low": args.ambiguous_low,
            "ambiguous_high": args.ambiguous_high,
            "train_share": args.train_share,
            "dev_share": args.dev_share,
            "probe_share": 1.0 - args.train_share - args.dev_share,
            "seed": args.seed,
        },
        "target_counts": {
            split: {str(label): count for label, count in sorted(counter.items())}
            for split, counter in targets.items()
        },
        "summaries": {
            "all_ambiguous": summarize(rows),
            "train": summarize(splits["train"]),
            "dev": summarize(splits["dev"]),
            "probe": summarize(splits["probe"]),
        },
        "leakage": leakage,
    }
    report["acceptance"] = acceptance(splits, leakage, low=args.ambiguous_low, high=args.ambiguous_high)
    write_json(report, Path(args.report))

    print("=" * 70)
    print("Built Round8 ambiguous selector splits")
    print("=" * 70)
    print(f"Ambiguous rows: {len(rows)} labels={dict(sorted(label_counts.items()))}")
    for name in ["train", "dev", "probe"]:
        print(f"{name}: {len(splits[name])} -> {getattr(args, name + '_output')}")
    print(f"Report: {args.report}")
    print(f"Ready for branch scoring: {report['acceptance']['ready_for_branch_scoring']}")


if __name__ == "__main__":
    main()
