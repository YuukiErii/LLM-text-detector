import argparse
import glob
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "outputs" / "round2" / "existing_family_threshold_report.md"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "round2" / "existing_family_threshold_report.json"


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


def run_name_from_path(path: Path) -> str:
    name = path.stem
    if name.startswith("teacher_test_"):
        name = name[len("teacher_test_") :]
    if name.endswith("_predictions"):
        name = name[: -len("_predictions")]
    return name


def expand_prediction_args(values: Sequence[str]) -> List[Path]:
    paths = []
    seen = set()
    for value in values:
        matches = glob.glob(value)
        if not matches:
            matches = [value]
        for match in matches:
            path = Path(match)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path = path.resolve()
            if path in seen:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Prediction file not found: {path}")
            seen.add(path)
            paths.append(path)
    return sorted(paths, key=lambda item: item.name)


def normalize_row(row: Dict, index: int, path: Path) -> Optional[Dict]:
    label = row.get("label")
    if label not in [0, 1]:
        return None
    prediction = row.get("prediction", row.get("pred", row.get("label_pred")))
    if prediction not in [0, 1]:
        prediction = int(to_float(row.get("prob_llm", row.get("probability", row.get("score")))) >= 0.5)
    sample_id = str(row.get("id", index))
    prob = to_float(row.get("prob_llm", row.get("probability", row.get("score"))))
    return {
        "id": sample_id,
        "label": int(label),
        "prediction": int(prediction),
        "probability": prob,
        "path": str(path),
    }


def load_run(path: Path) -> Dict:
    rows = []
    skipped = 0
    for index, row in enumerate(load_jsonl(path)):
        item = normalize_row(row, index=index, path=path)
        if item is None:
            skipped += 1
            continue
        rows.append(item)
    if not rows:
        raise ValueError(f"No labeled rows found in {path}")
    return {
        "name": run_name_from_path(path),
        "path": str(path),
        "rows": rows,
        "rows_by_id": {row["id"]: row for row in rows},
        "skipped": skipped,
    }


def metric_block(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, include_auc: bool = True) -> Dict:
    labels = labels.astype(int)
    preds = preds.astype(int)
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    total = int(len(labels))
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    roc_auc = None
    if include_auc and len(set(labels.tolist())) == 2:
        roc_auc = roc_auc_score(labels, probs)
    return {
        "n": total,
        "accuracy": 0.0 if total == 0 else (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_positives": fp,
        "false_negatives": fn,
    }


def error_ids_for(labels: np.ndarray, preds: np.ndarray, sample_ids: Sequence[str]) -> Dict[str, List[str]]:
    fp = []
    fn = []
    all_errors = []
    for sample_id, label, pred in zip(sample_ids, labels, preds):
        if int(label) == int(pred):
            continue
        all_errors.append(str(sample_id))
        if int(label) == 0 and int(pred) == 1:
            fp.append(str(sample_id))
        elif int(label) == 1 and int(pred) == 0:
            fn.append(str(sample_id))
    return {
        "false_positives": sorted(fp, key=lambda value: int(value) if value.isdigit() else value),
        "false_negatives": sorted(fn, key=lambda value: int(value) if value.isdigit() else value),
        "all_errors": sorted(all_errors, key=lambda value: int(value) if value.isdigit() else value),
    }


def threshold_candidates(probs: np.ndarray) -> np.ndarray:
    unique = sorted(set(float(prob) for prob in probs))
    candidates = {0.0, 1.0}
    for prob in unique:
        candidates.add(prob)
        candidates.add(min(1.0, prob + 1e-12))
    for left, right in zip(unique, unique[1:]):
        candidates.add((left + right) / 2.0)
    return np.array(sorted(candidates))


def best_threshold(labels: np.ndarray, probs: np.ndarray, sample_ids: Sequence[str], metric: str) -> Dict:
    best = None
    roc_auc = None
    if len(set(labels.tolist())) == 2:
        roc_auc = roc_auc_score(labels, probs)
    for threshold in threshold_candidates(probs):
        preds = (probs >= threshold).astype(int)
        metrics = metric_block(labels, preds, probs, include_auc=False)
        metrics["roc_auc"] = None if roc_auc is None else float(roc_auc)
        score = metrics[metric]
        tie_break = metrics["f1"] if metric == "accuracy" else metrics["accuracy"]
        candidate = (score, tie_break, -abs(float(threshold) - 0.5))
        if best is None or candidate > best["rank"]:
            best = {
                "threshold": float(threshold),
                "metrics": metrics,
                "error_ids": error_ids_for(labels, preds, sample_ids),
                "rank": candidate,
            }
    best.pop("rank")
    return best


def arrays_from_rows(rows: List[Dict]) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = [row["id"] for row in rows]
    labels = np.array([int(row["label"]) for row in rows])
    preds = np.array([int(row["prediction"]) for row in rows])
    probs = np.array([float(row["probability"]) for row in rows])
    return sample_ids, labels, preds, probs


def summarize_run(run: Dict) -> Dict:
    sample_ids, labels, preds, probs = arrays_from_rows(run["rows"])
    current_metrics = metric_block(labels, preds, probs)
    current_errors = error_ids_for(labels, preds, sample_ids)
    best_accuracy = best_threshold(labels, probs, sample_ids, metric="accuracy")
    best_f1 = best_threshold(labels, probs, sample_ids, metric="f1")
    current_error_set = set(current_errors["all_errors"])
    best_error_set = set(best_accuracy["error_ids"]["all_errors"])
    return {
        "name": run["name"],
        "path": run["path"],
        "skipped": run["skipped"],
        "current": {
            "metrics": current_metrics,
            "error_ids": current_errors,
        },
        "best_accuracy_threshold": best_accuracy,
        "best_f1_threshold": best_f1,
        "threshold_repair": {
            "fixed_by_best_accuracy_threshold": sorted(current_error_set - best_error_set, key=lambda value: int(value) if value.isdigit() else value),
            "new_errors_at_best_accuracy_threshold": sorted(best_error_set - current_error_set, key=lambda value: int(value) if value.isdigit() else value),
        },
    }


def common_ids_for(runs: Sequence[Dict]) -> List[str]:
    common = set(runs[0]["rows_by_id"])
    for run in runs[1:]:
        common &= set(run["rows_by_id"])
    return sorted(common, key=lambda value: int(value) if value.isdigit() else value)


def average_ensemble_summary(runs: Sequence[Dict], sample_ids: Sequence[str], name: str) -> Dict:
    labels = np.array([runs[0]["rows_by_id"][sample_id]["label"] for sample_id in sample_ids])
    probs = np.array([
        np.mean([run["rows_by_id"][sample_id]["probability"] for run in runs])
        for sample_id in sample_ids
    ])
    best_accuracy = best_threshold(labels, probs, sample_ids, metric="accuracy")
    best_f1 = best_threshold(labels, probs, sample_ids, metric="f1")
    return {
        "name": name,
        "members": [run["name"] for run in runs],
        "n": int(len(sample_ids)),
        "best_accuracy_threshold": best_accuracy,
        "best_f1_threshold": best_f1,
    }


def build_average_ensemble_summaries(runs: Sequence[Dict], max_combo_size: int) -> List[Dict]:
    summaries = []
    max_size = min(max_combo_size, len(runs))
    for size in range(2, max_size + 1):
        for combo in itertools.combinations(runs, size):
            ids = common_ids_for(combo)
            if not ids:
                continue
            name = "avg(" + "+".join(run["name"] for run in combo) + ")"
            summaries.append(average_ensemble_summary(combo, ids, name))
    return summaries


def current_error_set(run_summary: Dict) -> set:
    return set(run_summary["current"]["error_ids"]["all_errors"])


def overlap_matrix(run_summaries: Sequence[Dict]) -> List[Dict]:
    rows = []
    for left in run_summaries:
        left_errors = current_error_set(left)
        row = {"run": left["name"]}
        for right in run_summaries:
            right_errors = current_error_set(right)
            union = left_errors | right_errors
            inter = left_errors & right_errors
            row[right["name"]] = {
                "overlap_count": len(inter),
                "jaccard": None if not union else len(inter) / len(union),
            }
        rows.append(row)
    return rows


def sorted_ids(values: Iterable[str]) -> List[str]:
    return sorted(values, key=lambda value: int(value) if value.isdigit() else value)


def hard_case_summary(run_summaries: Sequence[Dict]) -> Dict:
    current_sets = [current_error_set(summary) for summary in run_summaries]
    best_accuracy_sets = [
        set(summary["best_accuracy_threshold"]["error_ids"]["all_errors"])
        for summary in run_summaries
    ]
    return {
        "wrong_in_all_current_decisions": sorted_ids(set.intersection(*current_sets)) if current_sets else [],
        "wrong_in_any_current_decision": sorted_ids(set.union(*current_sets)) if current_sets else [],
        "wrong_in_all_best_accuracy_thresholds": sorted_ids(set.intersection(*best_accuracy_sets)) if best_accuracy_sets else [],
        "wrong_in_any_best_accuracy_threshold": sorted_ids(set.union(*best_accuracy_sets)) if best_accuracy_sets else [],
    }


def top_average_ensembles(average_summaries: Sequence[Dict], limit: int) -> List[Dict]:
    return sorted(
        average_summaries,
        key=lambda item: (
            item["best_accuracy_threshold"]["metrics"]["accuracy"],
            item["best_accuracy_threshold"]["metrics"]["f1"],
            -len(item["members"]),
        ),
        reverse=True,
    )[:limit]


def list_preview(values: Sequence[str], limit: int = 20) -> str:
    if not values:
        return "none"
    shown = list(values[:limit])
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit} more)"
    return ", ".join(shown) + suffix


def markdown_run_table(run_summaries: Sequence[Dict]) -> List[str]:
    lines = [
        "| Run | Current acc | Current F1 | FP | FN | Best-acc threshold | Best acc | Best F1 threshold | Best F1 | Threshold-fixed | Threshold-new |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in run_summaries:
        current = summary["current"]["metrics"]
        best_acc = summary["best_accuracy_threshold"]
        best_f1 = summary["best_f1_threshold"]
        repair = summary["threshold_repair"]
        lines.append(
            f"| {summary['name']} | {fmt(current['accuracy'])} | {fmt(current['f1'])} | "
            f"{current['false_positives']} | {current['false_negatives']} | "
            f"{best_acc['threshold']:.6f} | {fmt(best_acc['metrics']['accuracy'])} | "
            f"{best_f1['threshold']:.6f} | {fmt(best_f1['metrics']['f1'])} | "
            f"{len(repair['fixed_by_best_accuracy_threshold'])} | {len(repair['new_errors_at_best_accuracy_threshold'])} |"
        )
    return lines


def markdown_average_table(average_summaries: Sequence[Dict], limit: int) -> List[str]:
    lines = [
        "| Rank | Members | Best-acc threshold | Accuracy | F1 | FP | FN |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, summary in enumerate(top_average_ensembles(average_summaries, limit=limit), start=1):
        best = summary["best_accuracy_threshold"]
        metrics = best["metrics"]
        lines.append(
            f"| {index} | {' + '.join(summary['members'])} | {best['threshold']:.6f} | "
            f"{fmt(metrics['accuracy'])} | {fmt(metrics['f1'])} | "
            f"{metrics['false_positives']} | {metrics['false_negatives']} |"
        )
    return lines


def markdown_overlap_table(run_summaries: Sequence[Dict], matrix: Sequence[Dict]) -> List[str]:
    names = [summary["name"] for summary in run_summaries]
    lines = [
        "| Run | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
    ]
    for row in matrix:
        values = []
        for name in names:
            cell = row[name]
            values.append(f"{cell['overlap_count']} ({fmt(cell['jaccard'])})")
        lines.append(f"| {row['run']} | " + " | ".join(values) + " |")
    return lines


def build_markdown(report: Dict, top_k: int) -> str:
    lines = [
        "# Round2 Existing-Family Threshold Diagnostics",
        "",
        "This report is diagnostic only. Oracle thresholds use labels and must not be treated as legitimate model-selection results.",
        "",
        "## Single-Run Threshold Ceiling",
        "",
    ]
    lines.extend(markdown_run_table(report["runs"]))
    lines.extend(["", "## Best Simple Average Ensembles", ""])
    lines.extend(markdown_average_table(report["average_ensembles"], limit=top_k))
    lines.extend(["", "## Current-Decision Error Overlap Matrix", ""])
    lines.append("Each cell is `overlap_count (Jaccard)` using current prediction decisions.")
    lines.append("")
    lines.extend(markdown_overlap_table(report["runs"], report["error_overlap_matrix"]))

    hard = report["hard_cases"]
    lines.extend(
        [
            "",
            "## Hard Cases",
            "",
            f"- Wrong in all current decisions: {list_preview(hard['wrong_in_all_current_decisions'])}",
            f"- Wrong in all best-accuracy thresholds: {list_preview(hard['wrong_in_all_best_accuracy_thresholds'])}",
            f"- Wrong in any current decision: {len(hard['wrong_in_any_current_decision'])}",
            f"- Wrong in any best-accuracy threshold: {len(hard['wrong_in_any_best_accuracy_threshold'])}",
            "",
            "## Threshold Repair Details",
            "",
        ]
    )
    for summary in report["runs"]:
        repair = summary["threshold_repair"]
        lines.extend(
            [
                f"### {summary['name']}",
                "",
                f"- Fixed by best-accuracy threshold: {list_preview(repair['fixed_by_best_accuracy_threshold'])}",
                f"- New errors at best-accuracy threshold: {list_preview(repair['new_errors_at_best_accuracy_threshold'])}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose existing teacher-test prediction files and oracle threshold ceilings.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction files or glob patterns.")
    parser.add_argument("--output_md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--max_combo_size", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = expand_prediction_args(args.predictions)
    runs = [load_run(path) for path in paths]
    run_summaries = [summarize_run(run) for run in runs]
    average_summaries = build_average_ensemble_summaries(runs, max_combo_size=args.max_combo_size)

    report = {
        "prediction_files": [str(path) for path in paths],
        "runs": run_summaries,
        "average_ensembles": average_summaries,
        "error_overlap_matrix": overlap_matrix(run_summaries),
        "hard_cases": hard_case_summary(run_summaries),
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(report, top_k=args.top_k), encoding="utf-8")

    best_single = max(run_summaries, key=lambda item: item["best_accuracy_threshold"]["metrics"]["accuracy"])
    best_avg = None
    if average_summaries:
        best_avg = top_average_ensembles(average_summaries, limit=1)[0]

    print("=" * 70)
    print("Round2 threshold-family diagnostics finished")
    print("=" * 70)
    print(f"Runs: {len(runs)}")
    print(
        "Best single-run oracle accuracy: "
        f"{best_single['name']} @ {best_single['best_accuracy_threshold']['threshold']:.6f} = "
        f"{best_single['best_accuracy_threshold']['metrics']['accuracy']:.4f}"
    )
    if best_avg is not None:
        print(
            "Best average oracle accuracy: "
            f"{' + '.join(best_avg['members'])} @ {best_avg['best_accuracy_threshold']['threshold']:.6f} = "
            f"{best_avg['best_accuracy_threshold']['metrics']['accuracy']:.4f}"
        )
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
