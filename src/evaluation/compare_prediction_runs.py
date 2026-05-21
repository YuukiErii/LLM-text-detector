import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


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


def parse_run_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Run must use NAME=PATH format, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f"Missing run name in: {value}")
    if not path:
        raise ValueError(f"Missing run path in: {value}")
    return name, Path(path)


def normalize_prediction_row(row: Dict, path: Path, row_index: int) -> Optional[Dict]:
    sample_id = row.get("id")
    if sample_id is None or sample_id == "":
        sample_id = str(row_index)

    label = row.get("label")
    prediction = row.get("prediction", row.get("pred", row.get("label_pred")))
    if label not in [0, 1] or prediction not in [0, 1]:
        return None

    prob = row.get("prob_llm", row.get("probability", row.get("score")))
    return {
        "id": str(sample_id),
        "label": int(label),
        "prediction": int(prediction),
        "prob_llm": to_float(prob),
        "domain": row.get("domain") or "unknown",
        "generator": row.get("generator") or "unknown",
        "source": row.get("source") or "unknown",
        "pair_id": row.get("pair_id") or "",
        "path": str(path),
    }


def load_prediction_run(path: Path) -> Dict[str, Dict]:
    rows = load_jsonl(path)
    normalized = {}
    skipped = 0
    duplicate_ids = 0

    for row_index, row in enumerate(rows):
        item = normalize_prediction_row(row, path=path, row_index=row_index)
        if item is None:
            skipped += 1
            continue
        sample_id = item["id"]
        if sample_id in normalized:
            duplicate_ids += 1
        normalized[sample_id] = item

    if not normalized:
        raise ValueError(f"No labeled prediction rows found in {path}")

    if skipped:
        print(f"[Warning] Skipped {skipped} unlabeled or malformed rows in {path}")
    if duplicate_ids:
        print(f"[Warning] Replaced {duplicate_ids} duplicate ids in {path}")

    return normalized


def metric_block(rows: Iterable[Dict]) -> Dict:
    rows = list(rows)
    if not rows:
        return {
            "n": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "false_positives": 0,
            "false_negatives": 0,
        }

    labels = np.array([int(row["label"]) for row in rows])
    preds = np.array([int(row["prediction"]) for row in rows])
    probs = np.array([to_float(row.get("prob_llm")) for row in rows])
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()

    roc_auc = None
    if len(set(labels.tolist())) == 2:
        roc_auc = roc_auc_score(labels, probs)

    return {
        "n": len(rows),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
    }


def group_metric_blocks(rows: Iterable[Dict], key: str, llm_only: bool = False) -> Dict[str, Dict]:
    groups = defaultdict(list)
    for row in rows:
        if llm_only and int(row["label"]) != 1:
            continue
        groups[str(row.get(key) or "unknown")].append(row)
    return {name: metric_block(items) for name, items in sorted(groups.items())}


def error_sets(rows_by_id: Dict[str, Dict]) -> Dict[str, set]:
    false_positives = {
        sample_id
        for sample_id, row in rows_by_id.items()
        if int(row["label"]) == 0 and int(row["prediction"]) == 1
    }
    false_negatives = {
        sample_id
        for sample_id, row in rows_by_id.items()
        if int(row["label"]) == 1 and int(row["prediction"]) == 0
    }
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def build_run_report(name: str, rows_by_id: Dict[str, Dict], path: Path) -> Dict:
    rows = list(rows_by_id.values())
    return {
        "name": name,
        "path": str(path),
        "overall": metric_block(rows),
        "domain": group_metric_blocks(rows, "domain"),
        "llm_generator": group_metric_blocks(rows, "generator", llm_only=True),
        "label_distribution": dict(Counter(str(row["label"]) for row in rows)),
        "prediction_distribution": dict(Counter(str(row["prediction"]) for row in rows)),
    }


def compare_to_baseline(
    baseline_name: str,
    baseline_rows: Dict[str, Dict],
    run_name: str,
    run_rows: Dict[str, Dict],
) -> Dict:
    base_errors = error_sets(baseline_rows)
    run_errors = error_sets(run_rows)
    common_ids = set(baseline_rows) & set(run_rows)

    label_mismatches = []
    for sample_id in sorted(common_ids):
        if int(baseline_rows[sample_id]["label"]) != int(run_rows[sample_id]["label"]):
            label_mismatches.append(sample_id)

    base_overall = metric_block(baseline_rows.values())
    run_overall = metric_block(run_rows.values())

    delta_metrics = {}
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        base_value = base_overall.get(key)
        run_value = run_overall.get(key)
        delta_metrics[key] = None if base_value is None or run_value is None else run_value - base_value

    return {
        "baseline": baseline_name,
        "run": run_name,
        "common_ids": len(common_ids),
        "missing_from_run": sorted(set(baseline_rows) - set(run_rows)),
        "extra_in_run": sorted(set(run_rows) - set(baseline_rows)),
        "label_mismatches": label_mismatches,
        "metric_delta": delta_metrics,
        "false_positive_delta": run_overall["false_positives"] - base_overall["false_positives"],
        "false_negative_delta": run_overall["false_negatives"] - base_overall["false_negatives"],
        "fixed_false_positives": sorted(base_errors["false_positives"] - run_errors["false_positives"]),
        "new_false_positives": sorted(run_errors["false_positives"] - base_errors["false_positives"]),
        "fixed_false_negatives": sorted(base_errors["false_negatives"] - run_errors["false_negatives"]),
        "new_false_negatives": sorted(run_errors["false_negatives"] - base_errors["false_negatives"]),
    }


def markdown_overall_table(run_reports: Dict[str, Dict]) -> List[str]:
    lines = [
        "| Run | n | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN | Confusion [[TN, FP], [FN, TP]] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, report in run_reports.items():
        m = report["overall"]
        lines.append(
            f"| {name} | {m['n']} | {fmt(m['accuracy'])} | {fmt(m['precision'])} | "
            f"{fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['roc_auc'])} | "
            f"{m['false_positives']} | {m['false_negatives']} | {m['confusion_matrix']} |"
        )
    return lines


def markdown_delta_table(comparisons: Dict[str, Dict]) -> List[str]:
    if not comparisons:
        return ["No baseline deltas available."]

    lines = [
        "| Run | dAccuracy | dPrecision | dRecall | dF1 | dROC-AUC | dFP | dFN | Fixed FP | New FP | Fixed FN | New FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, comp in comparisons.items():
        d = comp["metric_delta"]
        lines.append(
            f"| {name} | {fmt(d['accuracy'])} | {fmt(d['precision'])} | {fmt(d['recall'])} | "
            f"{fmt(d['f1'])} | {fmt(d['roc_auc'])} | {comp['false_positive_delta']:+d} | "
            f"{comp['false_negative_delta']:+d} | {len(comp['fixed_false_positives'])} | "
            f"{len(comp['new_false_positives'])} | {len(comp['fixed_false_negatives'])} | "
            f"{len(comp['new_false_negatives'])} |"
        )
    return lines


def markdown_group_table(run_reports: Dict[str, Dict], group_key: str, title: str) -> List[str]:
    lines = [f"## {title}", ""]
    lines.extend(
        [
            "| Run | Group | n | Accuracy | Precision | Recall | F1 | FP | FN |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run_name, report in run_reports.items():
        for group_name, metrics in report[group_key].items():
            lines.append(
                f"| {run_name} | {group_name} | {metrics['n']} | {fmt(metrics['accuracy'])} | "
                f"{fmt(metrics['precision'])} | {fmt(metrics['recall'])} | {fmt(metrics['f1'])} | "
                f"{metrics['false_positives']} | {metrics['false_negatives']} |"
            )
    return lines


def markdown_error_delta_details(comparisons: Dict[str, Dict], max_ids: int) -> List[str]:
    lines = ["## Error Delta Details", ""]
    if not comparisons:
        lines.append("No baseline deltas available.")
        return lines

    for run_name, comp in comparisons.items():
        lines.extend([f"### {run_name}", ""])
        for label, key in [
            ("Fixed false positives", "fixed_false_positives"),
            ("New false positives", "new_false_positives"),
            ("Fixed false negatives", "fixed_false_negatives"),
            ("New false negatives", "new_false_negatives"),
        ]:
            ids = comp[key]
            shown = ids[:max_ids]
            suffix = "" if len(ids) <= max_ids else f" ... (+{len(ids) - max_ids} more)"
            value = ", ".join(shown) + suffix if shown else "none"
            lines.append(f"- {label}: {value}")
        if comp["label_mismatches"]:
            lines.append(f"- Label mismatches: {', '.join(comp['label_mismatches'][:max_ids])}")
        if comp["missing_from_run"]:
            lines.append(f"- Missing from run: {len(comp['missing_from_run'])}")
        if comp["extra_in_run"]:
            lines.append(f"- Extra in run: {len(comp['extra_in_run'])}")
        lines.append("")
    return lines


def build_markdown_report(report: Dict, max_ids: int) -> str:
    lines = [
        f"# {report['title']}",
        "",
        f"Split: `{report['split_name']}`",
        "",
        f"Baseline run: `{report['baseline']}`",
        "",
        "## Overall Metrics",
        "",
    ]
    lines.extend(markdown_overall_table(report["runs"]))
    lines.extend(["", "## Delta vs Baseline", ""])
    lines.extend(markdown_delta_table(report["comparisons"]))
    lines.extend([""])
    lines.extend(markdown_group_table(report["runs"], "domain", "Domain Breakdown"))
    lines.extend([""])
    lines.extend(markdown_group_table(report["runs"], "llm_generator", "LLM Generator Breakdown"))
    lines.extend([""])
    lines.extend(markdown_error_delta_details(report["comparisons"], max_ids=max_ids))
    lines.extend(["", "## Run Paths", ""])
    for name, run in report["runs"].items():
        lines.append(f"- `{name}`: `{run['path']}`")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple labeled prediction JSONL runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Prediction runs as NAME=PATH. Provide at least two for deltas.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Run name to use as delta baseline. Defaults to the first --runs entry.",
    )
    parser.add_argument("--split_name", default="unknown")
    parser.add_argument("--title", default="Prediction Run Comparison")
    parser.add_argument(
        "--output_json",
        default=str(DEFAULT_OUTPUT_DIR / "prediction_run_comparison.json"),
    )
    parser.add_argument(
        "--output_md",
        default=str(DEFAULT_OUTPUT_DIR / "prediction_run_comparison.md"),
    )
    parser.add_argument(
        "--max_ids",
        type=int,
        default=20,
        help="Maximum sample ids to show per error-delta list in Markdown.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_items = [parse_run_arg(value) for value in args.runs]
    run_names = [name for name, _ in run_items]
    if len(set(run_names)) != len(run_names):
        raise ValueError(f"Duplicate run names are not allowed: {run_names}")

    baseline_name = args.baseline or run_names[0]
    if baseline_name not in run_names:
        raise ValueError(f"Baseline run {baseline_name!r} is not in --runs: {run_names}")

    loaded_runs = {}
    run_reports = {}
    for name, path in run_items:
        rows_by_id = load_prediction_run(path)
        loaded_runs[name] = rows_by_id
        run_reports[name] = build_run_report(name, rows_by_id, path=path)

    baseline_rows = loaded_runs[baseline_name]
    comparisons = {}
    for name, rows_by_id in loaded_runs.items():
        if name == baseline_name:
            continue
        comparisons[name] = compare_to_baseline(
            baseline_name=baseline_name,
            baseline_rows=baseline_rows,
            run_name=name,
            run_rows=rows_by_id,
        )

    report = {
        "title": args.title,
        "split_name": args.split_name,
        "baseline": baseline_name,
        "runs": run_reports,
        "comparisons": comparisons,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown_report(report, max_ids=args.max_ids), encoding="utf-8")

    print("=" * 70)
    print("Prediction run comparison finished")
    print("=" * 70)
    print(f"Baseline: {baseline_name}")
    print(f"Runs: {', '.join(run_names)}")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
