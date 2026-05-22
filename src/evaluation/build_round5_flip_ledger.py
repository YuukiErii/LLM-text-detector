import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records


DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


SPLIT_DEFAULTS = {
    "internal_test": {
        "source": DEFAULT_PROCESSED_DIR / "lit_academic_poetry_internal_test.jsonl",
        "step7": DEFAULT_PREDICTION_DIR / "round4_step7_internal_test_predictions.jsonl",
        "round4": DEFAULT_PREDICTION_DIR / "round4_deberta_internal_test_predictions.jsonl",
        "guard": DEFAULT_PREDICTION_DIR / "round4_human_style_guard_internal_test_predictions.jsonl",
    },
    "hardpos": {
        "source": DEFAULT_PROCESSED_DIR / "round4_residual_dev_hardpos.jsonl",
        "step7": DEFAULT_PREDICTION_DIR / "round4_step7_hardpos_predictions.jsonl",
        "round4": DEFAULT_PREDICTION_DIR / "round4_deberta_hardpos_predictions.jsonl",
        "guard": DEFAULT_PREDICTION_DIR / "round4_human_style_guard_hardpos_predictions.jsonl",
    },
    "hardneg": {
        "source": DEFAULT_PROCESSED_DIR / "round4_residual_dev_hardneg.jsonl",
        "step7": DEFAULT_PREDICTION_DIR / "round4_step7_hardneg_predictions.jsonl",
        "round4": DEFAULT_PREDICTION_DIR / "round4_deberta_hardneg_predictions.jsonl",
        "guard": DEFAULT_PREDICTION_DIR / "round4_human_style_guard_hardneg_predictions.jsonl",
    },
}


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


def sample_id(row: Dict, index: int) -> str:
    value = row.get("id")
    if value is None or value == "":
        return str(index)
    return str(value)


def to_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_present(row: Dict, keys: Sequence[str], default=None):
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def prediction_value(row: Dict) -> Optional[int]:
    return to_int(first_present(row, ["prediction", "pred", "label_pred"]))


def probability_value(row: Dict) -> float:
    return to_float(first_present(row, ["prob_llm", "probability", "score", "p_deberta"]))


def index_rows(rows: Sequence[Dict]) -> Tuple[Dict[str, Dict], int]:
    indexed = {}
    duplicate_ids = 0
    for idx, row in enumerate(rows):
        key = sample_id(row, idx)
        if key in indexed:
            duplicate_ids += 1
        indexed[key] = row
    return indexed, duplicate_ids


def normalize_source_rows(path: Path) -> Tuple[Dict[str, Dict], int]:
    rows = load_records(path)
    return index_rows(rows)


def metrics_for(rows: Sequence[Dict], pred_key: str, prob_key: str) -> Dict:
    tp = fp = tn = fn = 0
    probs = []
    labels = []
    for row in rows:
        label = int(row["label"])
        pred = int(row[pred_key])
        labels.append(label)
        probs.append(float(row.get(prob_key, 0.0)))
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 1 and pred == 0:
            fn += 1

    n = len(rows)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "num_samples": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def flip_type(label: int, step7_pred: int, round4_pred: int) -> str:
    if step7_pred == 0 and round4_pred == 1 and label == 1:
        return "fixed_fn_candidate"
    if step7_pred == 0 and round4_pred == 1 and label == 0:
        return "induced_fp"
    if step7_pred == 1 and round4_pred == 0 and label == 0:
        return "round4_fixed_fp"
    if step7_pred == 1 and round4_pred == 0 and label == 1:
        return "round4_induced_fn"
    if step7_pred == label and round4_pred == step7_pred:
        return "stable_step7_correct"
    if step7_pred != label and round4_pred == step7_pred:
        return "both_miss"
    if step7_pred == label and round4_pred != label:
        return "round4_only_miss"
    if step7_pred != label and round4_pred == label:
        return "round4_only_fix"
    return "other"


def guard_prob(row: Optional[Dict]) -> float:
    if row is None:
        return 0.0
    return to_float(row.get("p_human_style"), 0.0)


def guard_veto(row: Optional[Dict]) -> int:
    if row is None:
        return 0
    return int(to_int(row.get("human_style_veto"), 0) or 0)


def word_count(text: str) -> int:
    return len(str(text or "").split())


def row_bucket(text: str, *rows: Optional[Dict], key: str) -> str:
    for row in rows:
        if not row:
            continue
        value = row.get(key)
        if value:
            return str(value)
    return assign_bucket(text) if text else "general_prose"


def align_split(split_name: str, specs: Dict[str, Path]) -> Tuple[List[Dict], Dict]:
    source_by_id, source_dupes = normalize_source_rows(specs["source"])
    step7_rows = load_jsonl(specs["step7"])
    round4_rows = load_jsonl(specs["round4"])
    guard_rows = load_jsonl(specs["guard"])
    step7_by_id, step7_dupes = index_rows(step7_rows)
    round4_by_id, round4_dupes = index_rows(round4_rows)
    guard_by_id, guard_dupes = index_rows(guard_rows)

    aligned = []
    missing_round4 = []
    missing_guard = []
    missing_source = []
    pair_mismatches = []
    skipped_malformed = []

    for idx, step7 in enumerate(step7_rows):
        row_key = sample_id(step7, idx)
        round4 = round4_by_id.get(row_key)
        guard = guard_by_id.get(row_key)
        source = source_by_id.get(row_key)
        if round4 is None:
            missing_round4.append(row_key)
            continue
        if guard is None:
            missing_guard.append(row_key)
            continue
        if source is None:
            missing_source.append(row_key)

        label = to_int(first_present(step7, ["label"], first_present(round4, ["label"], first_present(source or {}, ["label"]))))
        step7_pred = prediction_value(step7)
        round4_pred = prediction_value(round4)
        if label is None or step7_pred is None or round4_pred is None:
            skipped_malformed.append(row_key)
            continue

        text = str(first_present(step7, ["text"], first_present(round4, ["text"], first_present(source or {}, ["text"], ""))))
        pair_id = str(first_present(step7, ["pair_id"], first_present(round4, ["pair_id"], first_present(source or {}, ["pair_id"], ""))))
        pair_values = [
            str(row.get("pair_id"))
            for row in [step7, round4, guard, source]
            if row and row.get("pair_id") not in [None, ""]
        ]
        if len(set(pair_values)) > 1:
            pair_mismatches.append(row_key)

        bucket = row_bucket(text, step7, round4, source, key="bucket")
        round4_bucket = row_bucket(text, step7, round4, source, key="round4_bucket")
        round4_tag = str(first_present(round4, ["round4_tag"], first_present(step7, ["round4_tag"], first_present(source or {}, ["round4_tag"], ""))))
        step7_prob = probability_value(step7)
        round4_prob = probability_value(round4)
        p_human_style = guard_prob(guard)
        item = {
            "id": row_key,
            "pair_id": pair_id,
            "label": int(label),
            "text": text,
            "split": split_name,
            "domain": first_present(step7, ["domain"], first_present(round4, ["domain"], first_present(source or {}, ["domain"], "unknown"))),
            "generator": first_present(step7, ["generator"], first_present(round4, ["generator"], first_present(source or {}, ["generator"], "unknown"))),
            "source": first_present(step7, ["source"], first_present(round4, ["source"], first_present(source or {}, ["source"], "unknown"))),
            "bucket": bucket,
            "round4_bucket": round4_bucket,
            "round4_tag": round4_tag,
            "step7_prob": float(step7_prob),
            "round4_prob": float(round4_prob),
            "prob_delta": float(round4_prob - step7_prob),
            "step7_pred": int(step7_pred),
            "round4_pred": int(round4_pred),
            "guard_p_human_style": float(p_human_style),
            "guard_human_style_veto": guard_veto(guard),
            "word_count": word_count(text),
        }
        item["flip_type"] = flip_type(item["label"], item["step7_pred"], item["round4_pred"])
        item["round4_override_candidate"] = bool(item["step7_pred"] == 0 and item["round4_pred"] == 1)
        if item["flip_type"] in {"fixed_fn_candidate", "induced_fp"}:
            item["flip_guard_label"] = 1 if item["flip_type"] == "induced_fp" else 0
            item["override_safety"] = "unsafe_override" if item["flip_guard_label"] == 1 else "safe_override"
        aligned.append(item)

    diagnostics = {
        "split": split_name,
        "source_rows": len(source_by_id),
        "step7_rows": len(step7_rows),
        "round4_rows": len(round4_rows),
        "guard_rows": len(guard_rows),
        "aligned_rows": len(aligned),
        "missing_round4": len(missing_round4),
        "missing_guard": len(missing_guard),
        "missing_source": len(missing_source),
        "pair_mismatches": len(pair_mismatches),
        "skipped_malformed": len(skipped_malformed),
        "duplicate_ids": {
            "source": source_dupes,
            "step7": step7_dupes,
            "round4": round4_dupes,
            "guard": guard_dupes,
        },
        "example_missing_round4": missing_round4[:10],
        "example_missing_guard": missing_guard[:10],
        "example_pair_mismatches": pair_mismatches[:10],
    }
    return aligned, diagnostics


def bucket_breakdown(rows: Sequence[Dict], bucket_key: str = "round4_bucket") -> Dict:
    counts = defaultdict(Counter)
    for row in rows:
        counts[str(row["flip_type"])][str(row.get(bucket_key) or "unknown")] += 1
    return {name: dict(counter) for name, counter in sorted(counts.items())}


def split_flip_counts(rows: Sequence[Dict]) -> Dict[str, Dict[str, int]]:
    counts = defaultdict(Counter)
    for row in rows:
        counts[str(row["split"])][str(row["flip_type"])] += 1
    return {split: dict(counter) for split, counter in sorted(counts.items())}


def baseline_report(aligned_by_split: Dict[str, List[Dict]], diagnostics: Dict[str, Dict]) -> Dict:
    report = {"splits": {}, "diagnostics": diagnostics}
    for split_name, rows in aligned_by_split.items():
        step7 = metrics_for(rows, "step7_pred", "step7_prob")
        round4 = metrics_for(rows, "round4_pred", "round4_prob")
        report["splits"][split_name] = {
            "step7": step7,
            "round4": round4,
            "row_alignment_passed": (
                diagnostics[split_name]["missing_round4"] == 0
                and diagnostics[split_name]["missing_guard"] == 0
                and diagnostics[split_name]["pair_mismatches"] == 0
                and diagnostics[split_name]["skipped_malformed"] == 0
            ),
        }
    return report


def markdown_metric_row(split_name: str, run_name: str, metrics: Dict) -> str:
    return (
        f"| {split_name} | {run_name} | {metrics['num_samples']} | "
        f"{metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
        f"{metrics['f1']:.4f} | {metrics['false_positives']} | {metrics['false_negatives']} |"
    )


def write_baseline_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round5 Baseline Frozen Report",
        "",
        "Teacher-test labels are not used here. This report freezes the non-teacher Step7, Round4 DeBERTa, and Round4 guard alignment used by Round5 Phase 1.",
        "",
        "## Metrics",
        "",
        "| Split | Run | n | Accuracy | Precision | Recall | F1 | FP | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name, split_report in report["splits"].items():
        lines.append(markdown_metric_row(split_name, "Step7", split_report["step7"]))
        lines.append(markdown_metric_row(split_name, "Round4 DeBERTa", split_report["round4"]))
    lines.extend(["", "## Alignment", ""])
    lines.append("| Split | Aligned | Step7 rows | Round4 rows | Guard rows | Missing Round4 | Missing Guard | Pair Mismatch | Pass |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for split_name, split_report in report["splits"].items():
        diag = report["diagnostics"][split_name]
        lines.append(
            f"| {split_name} | {diag['aligned_rows']} | {diag['step7_rows']} | {diag['round4_rows']} | "
            f"{diag['guard_rows']} | {diag['missing_round4']} | {diag['missing_guard']} | "
            f"{diag['pair_mismatches']} | {split_report['row_alignment_passed']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ledger_summary(summary: Dict, path: Path) -> None:
    lines = [
        "# Round5 Flip Ledger Summary",
        "",
        "This ledger converts the Step7 vs Round4 v1 disagreement pattern into explicit override-safety training signals.",
        "",
        "## Flip Type Counts",
        "",
        "| Split | Flip type | Count |",
        "| --- | --- | ---: |",
    ]
    for split_name, counts in summary["split_flip_counts"].items():
        for name, count in sorted(counts.items()):
            lines.append(f"| {split_name} | {name} | {count} |")

    lines.extend(["", "## Override-Candidate Counts", ""])
    lines.append("| Split | Safe fixed-FN candidates | Unsafe induced-FP candidates | Total override candidates |")
    lines.append("| --- | ---: | ---: | ---: |")
    for split_name, counts in summary["override_candidate_counts"].items():
        lines.append(
            f"| {split_name} | {counts.get('fixed_fn_candidate', 0)} | "
            f"{counts.get('induced_fp', 0)} | {counts.get('total', 0)} |"
        )

    lines.extend(["", "## Round4 Bucket Breakdown", ""])
    lines.append("| Flip type | Round4 bucket | Count |")
    lines.append("| --- | --- | ---: |")
    for flip_name, counter in summary["round4_bucket_breakdown"].items():
        for bucket_name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {flip_name} | {bucket_name} | {count} |")

    lines.extend(["", "## Guard Training Files", ""])
    for key, value in summary["output_files"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Initial Decision", "", summary["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def override_candidate_counts(rows: Sequence[Dict]) -> Dict[str, Dict[str, int]]:
    counts = defaultdict(Counter)
    for row in rows:
        if not row.get("round4_override_candidate"):
            continue
        split_name = str(row["split"])
        counts[split_name][str(row["flip_type"])] += 1
        counts[split_name]["total"] += 1
    return {split: dict(counter) for split, counter in sorted(counts.items())}


def guard_rows(rows: Sequence[Dict], splits: Sequence[str]) -> List[Dict]:
    split_set = set(splits)
    out = []
    for row in rows:
        if row.get("split") not in split_set:
            continue
        if row.get("flip_type") not in {"fixed_fn_candidate", "induced_fp"}:
            continue
        item = dict(row)
        item["label"] = int(item["flip_guard_label"])
        item["original_detection_label"] = int(row["label"])
        item["target"] = int(item["flip_guard_label"])
        out.append(item)
    return out


def build_outputs(all_rows: Sequence[Dict], args) -> Dict:
    ledger_path = Path(args.ledger_output)
    train_path = Path(args.flip_guard_train_output)
    hardpos_path = Path(args.flip_guard_dev_hardpos_output)
    hardneg_path = Path(args.flip_guard_dev_hardneg_output)
    summary_path = Path(args.summary_md)
    baseline_md_path = Path(args.baseline_md)
    baseline_json_path = Path(args.baseline_json)
    summary_json_path = Path(args.summary_json)

    save_jsonl(all_rows, ledger_path)
    train_splits = [part.strip() for part in args.train_splits.split(",") if part.strip()]
    save_jsonl(guard_rows(all_rows, train_splits), train_path)
    save_jsonl(guard_rows(all_rows, ["hardpos"]), hardpos_path)
    save_jsonl(guard_rows(all_rows, ["hardneg"]), hardneg_path)
    return {
        "ledger": str(ledger_path),
        "flip_guard_train": str(train_path),
        "flip_guard_dev_hardpos": str(hardpos_path),
        "flip_guard_dev_hardneg": str(hardneg_path),
        "summary_md": str(summary_path),
        "summary_json": str(summary_json_path),
        "baseline_md": str(baseline_md_path),
        "baseline_json": str(baseline_json_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build Round5 Step7-vs-Round4 flip ledger.")
    for split_name, defaults in SPLIT_DEFAULTS.items():
        parser.add_argument(f"--{split_name}_source", default=str(defaults["source"]))
        parser.add_argument(f"--{split_name}_step7", default=str(defaults["step7"]))
        parser.add_argument(f"--{split_name}_round4", default=str(defaults["round4"]))
        parser.add_argument(f"--{split_name}_guard", default=str(defaults["guard"]))
    parser.add_argument("--ledger_output", default=str(DEFAULT_EVALUATION_DIR / "round5_flip_ledger.jsonl"))
    parser.add_argument("--summary_md", default=str(DEFAULT_EVALUATION_DIR / "round5_flip_ledger_summary.md"))
    parser.add_argument("--summary_json", default=str(DEFAULT_EVALUATION_DIR / "round5_flip_ledger_summary.json"))
    parser.add_argument("--baseline_md", default=str(DEFAULT_EVALUATION_DIR / "round5_baseline_frozen_report.md"))
    parser.add_argument("--baseline_json", default=str(DEFAULT_EVALUATION_DIR / "round5_baseline_frozen_report.json"))
    parser.add_argument("--flip_guard_train_output", default=str(DEFAULT_PROCESSED_DIR / "round5_flip_guard_train.jsonl"))
    parser.add_argument("--flip_guard_dev_hardpos_output", default=str(DEFAULT_PROCESSED_DIR / "round5_flip_guard_dev_hardpos.jsonl"))
    parser.add_argument("--flip_guard_dev_hardneg_output", default=str(DEFAULT_PROCESSED_DIR / "round5_flip_guard_dev_hardneg.jsonl"))
    parser.add_argument(
        "--train_splits",
        default="internal_test",
        help="Comma-separated non-teacher splits used for the initial flip-guard train file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    aligned_by_split = {}
    diagnostics = {}
    all_rows = []

    for split_name in SPLIT_DEFAULTS:
        specs = {
            "source": Path(getattr(args, f"{split_name}_source")),
            "step7": Path(getattr(args, f"{split_name}_step7")),
            "round4": Path(getattr(args, f"{split_name}_round4")),
            "guard": Path(getattr(args, f"{split_name}_guard")),
        }
        rows, diag = align_split(split_name, specs)
        aligned_by_split[split_name] = rows
        diagnostics[split_name] = diag
        all_rows.extend(rows)

    baseline = baseline_report(aligned_by_split, diagnostics)
    output_files = build_outputs(all_rows, args)

    train_rows = guard_rows(all_rows, [part.strip() for part in args.train_splits.split(",") if part.strip()])
    decision = (
        "Proceed to a lightweight flip-guard train only if the train file has both safe and unsafe override examples. "
        "If it is too small or one-sided, prioritize Round5 residual data augmentation before model training."
    )
    summary = {
        "num_ledger_rows": len(all_rows),
        "split_flip_counts": split_flip_counts(all_rows),
        "override_candidate_counts": override_candidate_counts(all_rows),
        "round4_bucket_breakdown": bucket_breakdown(all_rows, "round4_bucket"),
        "text_bucket_breakdown": bucket_breakdown(all_rows, "bucket"),
        "flip_guard_train_rows": len(train_rows),
        "flip_guard_train_label_counts": dict(Counter(str(row.get("flip_guard_label")) for row in train_rows)),
        "diagnostics": diagnostics,
        "output_files": output_files,
        "decision": decision,
    }
    write_json(baseline, Path(args.baseline_json))
    write_baseline_markdown(baseline, Path(args.baseline_md))
    write_json(summary, Path(args.summary_json))
    write_ledger_summary(summary, Path(args.summary_md))

    print("=" * 70)
    print("Round5 flip ledger built")
    print("=" * 70)
    print(f"Ledger rows: {len(all_rows)}")
    print(f"Flip-guard train rows: {len(train_rows)} labels={summary['flip_guard_train_label_counts']}")
    for split_name, counts in summary["override_candidate_counts"].items():
        print(
            f"{split_name}: override_candidates={counts.get('total', 0)} "
            f"safe={counts.get('fixed_fn_candidate', 0)} unsafe={counts.get('induced_fp', 0)}"
        )
    print(f"Baseline report: {args.baseline_md}")
    print(f"Ledger summary: {args.summary_md}")


if __name__ == "__main__":
    main()
