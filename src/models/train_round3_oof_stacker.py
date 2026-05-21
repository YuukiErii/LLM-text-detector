import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.round3_fusion_utils import (  # noqa: E402
    baseline_metrics,
    feature_dict,
    fmt,
    labels_for,
    load_split_sets,
    metrics_for_rows,
    metrics_table_lines,
    prediction_rows,
    safe_name,
    save_jsonl,
    split_metrics,
    write_json,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round3_oof_stacker"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round3_oof_stacker_report.md"


def build_model(c_value: float, class_weight: str) -> Pipeline:
    resolved_class_weight = None if class_weight == "none" else class_weight
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight=resolved_class_weight,
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def oof_predict(rows: Sequence[Dict], run_names: Sequence[str], folds: int, c_value: float, class_weight: str) -> np.ndarray:
    x_all = [feature_dict(row, run_names) for row in rows]
    y_all = labels_for(rows)
    min_class = int(min(np.bincount(y_all)))
    n_splits = max(2, min(folds, min_class))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = np.zeros(len(rows), dtype=float)
    for train_idx, valid_idx in splitter.split(x_all, y_all):
        model = build_model(c_value=c_value, class_weight=class_weight)
        model.fit([x_all[i] for i in train_idx], y_all[train_idx])
        probs[valid_idx] = model.predict_proba([x_all[i] for i in valid_idx])[:, 1]
    return probs


def train_final(rows: Sequence[Dict], run_names: Sequence[str], c_value: float, class_weight: str) -> Pipeline:
    model = build_model(c_value=c_value, class_weight=class_weight)
    model.fit([feature_dict(row, run_names) for row in rows], labels_for(rows))
    return model


def threshold_candidates(min_threshold: float, max_threshold: float, step: float) -> List[float]:
    values = []
    current = min_threshold
    while current <= max_threshold + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def select_threshold(rows: Sequence[Dict], probs: Sequence[float], args) -> Tuple[float, Dict]:
    split_names = sorted({str(row.get("split_name", "unknown")) for row in rows})
    baseline_by_split = {
        split_name: baseline_metrics([row for row in rows if row.get("split_name") == split_name], args.baseline_run)
        for split_name in split_names
    }
    candidates = []
    for threshold in threshold_candidates(args.min_threshold, args.max_threshold, args.threshold_step):
        overall = metrics_for_rows(rows, probs, threshold)
        by_split = split_metrics(rows, probs, threshold)
        guard_name = safe_name(args.guard_split)
        valid_name = safe_name(args.valid_split)
        round2_name = safe_name(args.round2_split)
        constraints = []
        if guard_name in by_split and guard_name in baseline_by_split:
            constraints.append(
                by_split[guard_name]["false_positives"]
                <= baseline_by_split[guard_name]["false_positives"] + args.guard_fp_tolerance
            )
        if valid_name in by_split and valid_name in baseline_by_split:
            constraints.append(by_split[valid_name]["f1"] >= baseline_by_split[valid_name]["f1"] - args.valid_f1_tolerance)

        guard_bonus = by_split.get(guard_name, {}).get("f1", 0.0)
        round2_bonus = by_split.get(round2_name, {}).get("f1", 0.0)
        guard_fp = by_split.get(guard_name, {}).get("false_positives", 0)
        score = overall["f1"] + 0.35 * guard_bonus + 0.20 * round2_bonus - 0.01 * guard_fp
        candidates.append(
            {
                "threshold": threshold,
                "score": float(score),
                "constraints_passed": bool(all(constraints)) if constraints else True,
                "overall": overall,
                "by_split": by_split,
            }
        )

    feasible = [item for item in candidates if item["constraints_passed"]]
    if feasible:
        best = max(feasible, key=lambda item: (item["score"], item["overall"]["f1"], item["threshold"]))
    else:
        guard_name = safe_name(args.guard_split)
        best = max(
            candidates,
            key=lambda item: (
                -item["by_split"].get(guard_name, {}).get("false_positives", 10**9),
                item["by_split"].get(guard_name, {}).get("recall", 0.0),
                item["overall"]["f1"],
            ),
        )

    report = {
        "selected_threshold": best["threshold"],
        "selected_constraints_passed": best["constraints_passed"],
        "baseline_by_split": baseline_by_split,
        "selected": best,
        "top_candidates": sorted(candidates, key=lambda item: item["score"], reverse=True)[:20],
    }
    return float(best["threshold"]), report


def predict_with_model(model: Pipeline, rows: Sequence[Dict], run_names: Sequence[str]) -> np.ndarray:
    return model.predict_proba([feature_dict(row, run_names) for row in rows])[:, 1]


def write_report(report: Dict, path: Path) -> None:
    lines = [
        "# Round3 Phase D OOF Stacker Report",
        "",
        "This stacker uses fixed upstream prediction files as deployable base signals",
        "and trains the meta-model with out-of-fold predictions across the meta",
        "training splits. Teacher-test labels are not used for training or threshold",
        "selection.",
        "",
        "## Configuration",
        "",
        f"- Base runs: {', '.join(report['run_names'])}",
        f"- OOF folds: {report['folds']}",
        f"- C: {report['C']}",
        f"- class_weight: {report['class_weight']}",
        f"- selected threshold: {report['threshold']:.4f}",
        f"- threshold constraints passed: {report['threshold_report']['selected_constraints_passed']}",
        "",
        "## OOF Meta Metrics",
        "",
    ]
    lines.extend(metrics_table_lines({"oof_train": report["oof_metrics"]}))
    lines.extend(["", "## OOF Metrics By Training Split", ""])
    lines.extend(metrics_table_lines(report["oof_split_metrics"]))
    lines.extend(["", "## Step7 Baselines On Training Splits", ""])
    lines.extend(metrics_table_lines(report["threshold_report"]["baseline_by_split"]))
    lines.extend(["", "## Evaluation Splits", ""])
    lines.extend(metrics_table_lines(report["eval_metrics"]))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            "## Output Files",
            "",
            f"- Model: `{report['model_path']}`",
            f"- OOF predictions: `{report['oof_predictions']}`",
            f"- JSON report: `{report['json_report']}`",
        ]
    )
    for split_name, path_value in report.get("oof_prediction_files", {}).items():
        if split_name != "combined":
            lines.append(f"- {split_name} OOF predictions: `{path_value}`")
    for split_name, path_value in report["eval_prediction_files"].items():
        lines.append(f"- {split_name} predictions: `{path_value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Round3 OOF stacking fusion model.")
    parser.add_argument(
        "--train_set",
        action="append",
        nargs="+",
        required=True,
        help="SPLIT_NAME followed by NAME=PATH prediction specs. Repeat for each meta-training split.",
    )
    parser.add_argument(
        "--eval_set",
        action="append",
        nargs="+",
        default=[],
        help="SPLIT_NAME followed by NAME=PATH prediction specs. Repeat for each evaluation split.",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction_dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--C", type=float, default=0.25)
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--baseline_run", default="step7")
    parser.add_argument("--guard_split", default="guard_dev")
    parser.add_argument("--valid_split", default="valid")
    parser.add_argument("--round2_split", default="round2_dev")
    parser.add_argument("--guard_fp_tolerance", type=int, default=0)
    parser.add_argument("--valid_f1_tolerance", type=float, default=0.002)
    parser.add_argument("--min_threshold", type=float, default=0.30)
    parser.add_argument("--max_threshold", type=float, default=0.90)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    run_names, train_rows = load_split_sets(args.train_set)
    if not train_rows:
        raise ValueError("No meta-training rows loaded.")

    oof_probs = oof_predict(train_rows, run_names, folds=args.folds, c_value=args.C, class_weight=args.class_weight)
    threshold, threshold_report = select_threshold(train_rows, oof_probs, args)
    model = train_final(train_rows, run_names, c_value=args.C, class_weight=args.class_weight)

    model_path = output_dir / "stacking_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"model": model, "run_names": run_names, "threshold": threshold}, f)

    oof_predictions_path = output_dir / "oof_meta_predictions.jsonl"
    save_jsonl(prediction_rows(train_rows, oof_probs, threshold, "round3_oof_stacker_oof"), oof_predictions_path)
    oof_prediction_files = {"combined": str(oof_predictions_path)}
    for split_name in sorted({str(row["split_name"]) for row in train_rows}):
        indices = [index for index, row in enumerate(train_rows) if row["split_name"] == split_name]
        split_rows = [train_rows[index] for index in indices]
        split_probs = [float(oof_probs[index]) for index in indices]
        split_path = prediction_dir / f"round3_oof_stacker_{split_name}_oof_predictions.jsonl"
        save_jsonl(prediction_rows(split_rows, split_probs, threshold, "round3_oof_stacker_oof"), split_path)
        oof_prediction_files[split_name] = str(split_path)

    eval_metrics = {}
    eval_prediction_files = {}
    if args.eval_set:
        eval_run_names, eval_rows = load_split_sets(args.eval_set)
        if eval_run_names != run_names:
            raise ValueError(f"Eval run names {eval_run_names} do not match train run names {run_names}")
        by_split: Dict[str, List[Dict]] = {}
        for row in eval_rows:
            by_split.setdefault(str(row["split_name"]), []).append(row)
        for split_name, rows in sorted(by_split.items()):
            probs = predict_with_model(model, rows, run_names)
            pred_path = prediction_dir / f"round3_oof_stacker_{split_name}_predictions.jsonl"
            save_jsonl(prediction_rows(rows, probs, threshold, "round3_oof_stacker"), pred_path)
            metrics = metrics_for_rows(rows, probs, threshold)
            eval_metrics[split_name] = metrics
            eval_prediction_files[split_name] = str(pred_path)
            write_json(metrics, prediction_dir / f"round3_oof_stacker_{split_name}_metrics.json")

    oof_metrics = metrics_for_rows(train_rows, oof_probs, threshold)
    oof_split_metrics = split_metrics(train_rows, oof_probs, threshold)
    decision = (
        "The OOF stacker may advance to Phase E only as a guarded repair signal. "
        "It is not allowed to globally replace Step7 unless the guard-dev FP and "
        "internal-test gates both pass."
    )
    if not threshold_report["selected_constraints_passed"]:
        decision = (
            "The OOF threshold search could not satisfy the configured precision constraints. "
            "Use this model only for diagnostics or behind a stricter Phase E override gate."
        )

    json_report_path = output_dir / "oof_stacker_report.json"
    report = {
        "run_names": run_names,
        "train_sets": args.train_set,
        "eval_sets": args.eval_set,
        "folds": args.folds,
        "C": args.C,
        "class_weight": args.class_weight,
        "threshold": threshold,
        "threshold_report": threshold_report,
        "oof_metrics": oof_metrics,
        "oof_split_metrics": oof_split_metrics,
        "eval_metrics": eval_metrics,
        "eval_prediction_files": eval_prediction_files,
        "oof_prediction_files": oof_prediction_files,
        "decision": decision,
        "model_path": str(model_path),
        "oof_predictions": str(oof_predictions_path),
        "json_report": str(json_report_path),
    }
    write_json(report, json_report_path)
    write_report(report, Path(args.report_md))

    print("=" * 70)
    print("Round3 OOF stacker trained")
    print("=" * 70)
    print(f"Train rows: {len(train_rows)}")
    print(f"Runs: {', '.join(run_names)}")
    print(f"Threshold: {threshold:.4f}")
    print(f"OOF F1: {oof_metrics['f1']:.4f}")
    print(f"Model: {model_path}")
    print(f"Report: {args.report_md}")
    for split_name, block in eval_metrics.items():
        print(
            f"{split_name}: f1={block['f1']:.4f} acc={block['accuracy']:.4f} "
            f"FP={block['false_positives']} FN={block['false_negatives']}"
        )


if __name__ == "__main__":
    main()
