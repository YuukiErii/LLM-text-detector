import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "calibration"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


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


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_rows(rows: Iterable[Dict]) -> List[Dict]:
    normalized = []
    for idx, row in enumerate(rows):
        label = row.get("label")
        if label not in [0, 1]:
            continue
        sample_id = row.get("id")
        if sample_id is None or sample_id == "":
            sample_id = str(idx)
        prob = row.get("prob_llm", row.get("probability"))
        item = dict(row)
        item["id"] = str(sample_id)
        item["label"] = int(label)
        item["prob_llm"] = to_float(prob)
        normalized.append(item)
    return normalized


def arrays(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    probs = np.array([float(row["prob_llm"]) for row in rows], dtype=float)
    return labels, probs


def metric_block(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    roc_auc = None
    if len(set(labels.tolist())) == 2:
        roc_auc = roc_auc_score(labels, probs)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
    }


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_threshold_grid(start: float, stop: float, step: float) -> List[float]:
    values = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def evaluate_grid(labels: np.ndarray, probs: np.ndarray, thresholds: List[float]) -> List[Dict]:
    return [metric_block(labels, probs, threshold) for threshold in thresholds]


def select_best_f1(grid: List[Dict]) -> Dict:
    return max(
        grid,
        key=lambda item: (
            item["f1"],
            item["accuracy"],
            item["precision"],
            item["recall"],
            -abs(item["threshold"] - 0.5),
        ),
    )


def select_precision_target(grid: List[Dict], target: float) -> Dict:
    candidates = [item for item in grid if item["precision"] >= target]
    if candidates:
        selected = max(
            candidates,
            key=lambda item: (
                item["f1"],
                item["recall"],
                item["precision"],
                item["accuracy"],
                -abs(item["threshold"] - 0.5),
            ),
        )
        selected = dict(selected)
        selected["target_satisfied"] = True
        return selected

    selected = max(
        grid,
        key=lambda item: (
            item["precision"],
            item["f1"],
            item["recall"],
            item["accuracy"],
            -abs(item["threshold"] - 0.5),
        ),
    )
    selected = dict(selected)
    selected["target_satisfied"] = False
    return selected


def select_recall_target(grid: List[Dict], target: float) -> Dict:
    candidates = [item for item in grid if item["recall"] >= target]
    if candidates:
        selected = max(
            candidates,
            key=lambda item: (
                item["f1"],
                item["precision"],
                item["recall"],
                item["accuracy"],
                -abs(item["threshold"] - 0.5),
            ),
        )
        selected = dict(selected)
        selected["target_satisfied"] = True
        return selected

    selected = max(
        grid,
        key=lambda item: (
            item["recall"],
            item["f1"],
            item["precision"],
            item["accuracy"],
            -abs(item["threshold"] - 0.5),
        ),
    )
    selected = dict(selected)
    selected["target_satisfied"] = False
    return selected


def fit_probability_views(valid_labels: np.ndarray, valid_probs: np.ndarray):
    views = {"raw": None}

    platt = LogisticRegression(solver="lbfgs", random_state=42)
    platt.fit(valid_probs.reshape(-1, 1), valid_labels)
    views["platt"] = platt

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(valid_probs, valid_labels)
    views["isotonic"] = isotonic

    return views


def apply_probability_view(model, probs: np.ndarray) -> np.ndarray:
    if model is None:
        return probs
    if isinstance(model, LogisticRegression):
        return model.predict_proba(probs.reshape(-1, 1))[:, 1]
    if isinstance(model, IsotonicRegression):
        return model.predict(probs)
    raise TypeError(f"Unsupported calibration model: {type(model)}")


def write_predictions(rows: List[Dict], probs: np.ndarray, threshold: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds = (probs >= threshold).astype(int)

    with output_path.open("w", encoding="utf-8") as f:
        for row, prob, pred in zip(rows, probs, preds):
            item = dict(row)
            item["prediction"] = int(pred)
            item["prob_llm"] = float(prob)
            item["probability"] = float(prob)
            item["calibration_threshold"] = float(threshold)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_selector_plan(
    valid_grid: List[Dict],
    precision_targets: List[float],
    recall_targets: List[float],
) -> Dict[str, Dict]:
    plan = {"best_f1": select_best_f1(valid_grid)}

    for target in precision_targets:
        plan[f"precision_ge_{target:g}"] = select_precision_target(valid_grid, target)

    for target in recall_targets:
        plan[f"recall_ge_{target:g}"] = select_recall_target(valid_grid, target)

    return plan


def summarize_config(
    selector: str,
    method: str,
    valid_selection: Dict,
    valid_labels: np.ndarray,
    valid_probs: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
) -> Dict:
    threshold = float(valid_selection["threshold"])
    return {
        "selector": selector,
        "calibration_method": method,
        "threshold_selected_on_valid": threshold,
        "target_satisfied_on_valid": valid_selection.get("target_satisfied"),
        "valid": metric_block(valid_labels, valid_probs, threshold),
        "internal_test": metric_block(test_labels, test_probs, threshold),
    }


def write_markdown(report: Dict, output_path: Path) -> None:
    lines = [
        f"# Calibration Report: {report['run_name']}",
        "",
        "Selection split: `validation`",
        "Evaluation split: `internal_test`",
        "",
        "## Selected Operating Points",
        "",
        "| Selector | Method | Threshold | Valid Precision | Valid Recall | Valid F1 | Test Precision | Test Recall | Test F1 | Test FP | Test FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in report["selected_configs"]:
        valid = row["valid"]
        test = row["internal_test"]
        lines.append(
            f"| {row['selector']} | {row['calibration_method']} | {row['threshold_selected_on_valid']:.4f} | "
            f"{fmt(valid['precision'])} | {fmt(valid['recall'])} | {fmt(valid['f1'])} | "
            f"{fmt(test['precision'])} | {fmt(test['recall'])} | {fmt(test['f1'])} | "
            f"{test['false_positives']} | {test['false_negatives']} |"
        )

    lines.extend(["", "## Best Configs By Selector", ""])
    lines.extend(
        [
            "| Selector | Best Method By Valid F1 | Valid F1 | Test F1 | Test FP | Test FN |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for selector, row in report["best_by_selector"].items():
        lines.append(
            f"| {selector} | {row['calibration_method']} | {fmt(row['valid']['f1'])} | "
            f"{fmt(row['internal_test']['f1'])} | {row['internal_test']['false_positives']} | "
            f"{row['internal_test']['false_negatives']} |"
        )

    lines.extend(["", "## Notes", ""])
    lines.append("- Thresholds and calibration models are selected only from validation predictions.")
    lines.append("- Internal-test metrics are evaluation-only and should not be used to choose teacher-test settings.")
    lines.append("- Teacher test is intentionally not used here.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-only threshold and probability calibration.")
    parser.add_argument("--valid_predictions", required=True)
    parser.add_argument("--test_predictions", required=True)
    parser.add_argument("--run_name", default="prediction_run")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--threshold_start", type=float, default=0.01)
    parser.add_argument("--threshold_stop", type=float, default=0.99)
    parser.add_argument("--threshold_step", type=float, default=0.001)
    parser.add_argument("--precision_targets", type=str, default="0.97,0.98")
    parser.add_argument("--recall_targets", type=str, default="0.95,0.96")
    parser.add_argument(
        "--write_selected_predictions",
        action="store_true",
        help="Write valid/internal-test prediction JSONL files for every selected config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    valid_path = Path(args.valid_predictions)
    test_path = Path(args.test_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = normalize_rows(load_jsonl(valid_path))
    test_rows = normalize_rows(load_jsonl(test_path))

    if not valid_rows:
        raise ValueError(f"No labeled validation rows found: {valid_path}")
    if not test_rows:
        raise ValueError(f"No labeled internal-test rows found: {test_path}")

    valid_labels, valid_probs_raw = arrays(valid_rows)
    test_labels, test_probs_raw = arrays(test_rows)

    thresholds = build_threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)
    precision_targets = parse_float_list(args.precision_targets)
    recall_targets = parse_float_list(args.recall_targets)
    views = fit_probability_views(valid_labels, valid_probs_raw)

    selected_configs = []
    grid_by_method = {}

    for method, model in views.items():
        valid_probs = apply_probability_view(model, valid_probs_raw)
        test_probs = apply_probability_view(model, test_probs_raw)
        valid_grid = evaluate_grid(valid_labels, valid_probs, thresholds)
        grid_by_method[method] = valid_grid
        selector_plan = build_selector_plan(valid_grid, precision_targets, recall_targets)

        for selector, valid_selection in selector_plan.items():
            summary = summarize_config(
                selector=selector,
                method=method,
                valid_selection=valid_selection,
                valid_labels=valid_labels,
                valid_probs=valid_probs,
                test_labels=test_labels,
                test_probs=test_probs,
            )
            selected_configs.append(summary)

            if args.write_selected_predictions:
                pred_dir = output_dir / "predictions"
                stem = f"{safe_name(args.run_name)}_{safe_name(method)}_{safe_name(selector)}"
                write_predictions(valid_rows, valid_probs, summary["threshold_selected_on_valid"], pred_dir / f"{stem}_valid_predictions.jsonl")
                write_predictions(test_rows, test_probs, summary["threshold_selected_on_valid"], pred_dir / f"{stem}_internal_test_predictions.jsonl")

    best_by_selector = {}
    for row in selected_configs:
        selector = row["selector"]
        current = best_by_selector.get(selector)
        if current is None or (
            row["valid"]["f1"],
            row["valid"]["precision"],
            row["valid"]["recall"],
        ) > (
            current["valid"]["f1"],
            current["valid"]["precision"],
            current["valid"]["recall"],
        ):
            best_by_selector[selector] = row

    report = {
        "run_name": args.run_name,
        "valid_predictions": str(valid_path),
        "test_predictions": str(test_path),
        "threshold_grid": {
            "start": args.threshold_start,
            "stop": args.threshold_stop,
            "step": args.threshold_step,
            "num_thresholds": len(thresholds),
        },
        "precision_targets": precision_targets,
        "recall_targets": recall_targets,
        "selected_configs": selected_configs,
        "best_by_selector": best_by_selector,
    }

    json_path = output_dir / "calibration_report.json"
    md_path = output_dir / "calibration_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print("=" * 70)
    print("Calibration finished")
    print("=" * 70)
    print(f"Run: {args.run_name}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print("Best by selector:")
    for selector, row in best_by_selector.items():
        print(
            f"  {selector}: {row['calibration_method']} "
            f"thr={row['threshold_selected_on_valid']:.4f} "
            f"valid_f1={row['valid']['f1']:.4f} "
            f"test_f1={row['internal_test']['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
