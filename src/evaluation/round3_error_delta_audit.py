import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket


DEFAULT_STEP7 = PROJECT_ROOT / "outputs" / "predictions" / "round2_step7_teacher_test_predictions.jsonl"
DEFAULT_CANDIDATES = [
    "bucket_routed=outputs/predictions/round2_bucket_routed_teacher_test_predictions.jsonl",
    "stacker_step7=outputs/predictions/round2_stacker_teacher_test_predictions.jsonl",
    "roberta_single=outputs/predictions/round2_roberta_teacher_test_predictions.jsonl",
    "stacker_with_roberta=outputs/predictions/round2_stacker_with_roberta_teacher_test_predictions.jsonl",
]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "round3" / "error_delta_audit.csv"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "outputs" / "round3" / "error_delta_audit.md"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "round3" / "error_delta_by_bucket.json"


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


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Candidate must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def probability(row: Dict) -> float:
    for key in ["probability", "prob_llm", "p_llm", "score"]:
        if row.get(key) is not None:
            return float(row[key])
    return float(row.get("prediction", 0))


def normalized_prediction_rows(path: Path) -> Dict[str, Dict]:
    rows = {}
    for index, row in enumerate(load_jsonl(path)):
        if row.get("label") not in [0, 1] or row.get("prediction") not in [0, 1]:
            continue
        item = dict(row)
        item["id"] = str(item.get("id", index))
        item["label"] = int(item["label"])
        item["prediction"] = int(item["prediction"])
        item["probability"] = probability(item)
        rows[item["id"]] = item
    if not rows:
        raise ValueError(f"No labeled prediction rows found: {path}")
    return rows


def rough_domain_from_bucket(bucket: str) -> str:
    if bucket.startswith("poetry_"):
        return "poetry"
    if bucket == "academic_formal":
        return "academic"
    if bucket.startswith("literary_"):
        return "literature"
    return "general_prose"


def label_name(label: int) -> str:
    return "LLM" if int(label) == 1 else "human"


def is_error(row: Dict) -> bool:
    return int(row["label"]) != int(row["prediction"])


def metric_block(rows: Sequence[Dict]) -> Dict:
    y_true = np.array([int(row["label"]) for row in rows], dtype=int)
    y_pred = np.array([int(row["prediction"]) for row in rows], dtype=int)
    y_prob = np.array([float(row["probability"]) for row in rows], dtype=float)
    fp_ids = [str(row["id"]) for row in rows if row["label"] == 0 and row["prediction"] == 1]
    fn_ids = [str(row["id"]) for row in rows if row["label"] == 1 and row["prediction"] == 0]
    out = {
        "n": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "fp": len(fp_ids),
        "fn": len(fn_ids),
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
        "error_ids": sorted(set(fp_ids + fn_ids), key=sort_key),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = None
    return out


def sort_key(value: str):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def summarize_text(text: str, max_chars: int = 180) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def build_audit_rows(step7_rows: Dict[str, Dict], candidates: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    audit_rows = []
    for row_id in sorted(step7_rows, key=sort_key):
        base = step7_rows[row_id]
        label = int(base["label"])
        step7_pred = int(base["prediction"])
        text = str(base.get("text", ""))
        bucket = str(base.get("bucket") or assign_bucket(text))
        rough_domain = str(base.get("rough_domain") or base.get("domain") or rough_domain_from_bucket(bucket))
        fixed_by = []
        broken_by = []
        candidate_preds = {}
        candidate_probs = {}
        candidate_delta = {}

        for name, rows in candidates.items():
            candidate = rows.get(row_id)
            if candidate is None:
                raise ValueError(f"Candidate {name} is missing id {row_id}")
            if int(candidate["label"]) != label:
                raise ValueError(f"Candidate {name} has mismatched label for id {row_id}")
            pred = int(candidate["prediction"])
            prob = float(candidate["probability"])
            candidate_preds[name] = pred
            candidate_probs[name] = prob

            fixed = step7_pred != label and pred == label
            broken = step7_pred == label and pred != label
            if fixed:
                fixed_by.append(name)
            if broken:
                broken_by.append(name)

            if fixed:
                delta = "fixed_step7_error"
            elif broken:
                delta = "broke_step7_correct"
            elif pred != label:
                delta = "same_error_as_step7"
            else:
                delta = "same_correct_as_step7"
            candidate_delta[name] = delta

        notes = []
        if step7_pred == 0 and label == 1 and fixed_by:
            notes.append("candidate_fixed_step7_fn")
        if step7_pred == 1 and label == 0 and not fixed_by:
            notes.append("persistent_step7_fp")
        if step7_pred == 0 and label == 1 and not fixed_by:
            notes.append("persistent_step7_fn")
        if broken_by and label == 0:
            notes.append("candidate_induced_human_fp")
        if broken_by and label == 1:
            notes.append("candidate_induced_llm_fn")

        item = {
            "id": row_id,
            "label": label,
            "label_name": label_name(label),
            "text": text,
            "text_preview": summarize_text(text),
            "rough_domain": rough_domain,
            "bucket": bucket,
            "step7_pred": step7_pred,
            "step7_prob": float(base["probability"]),
            "step7_correct": int(step7_pred == label),
            "candidate_preds": candidate_preds,
            "candidate_probs": candidate_probs,
            "candidate_delta": candidate_delta,
            "fixed_by_candidates": fixed_by,
            "broken_by_candidates": broken_by,
            "is_step7_fp": int(label == 0 and step7_pred == 1),
            "is_step7_fn": int(label == 1 and step7_pred == 0),
            "is_new_fp": int(label == 0 and bool(broken_by)),
            "is_new_fn": int(label == 1 and bool(broken_by)),
            "notes": ";".join(notes),
        }
        for name in candidates:
            item[f"{name}_pred"] = candidate_preds[name]
            item[f"{name}_prob"] = candidate_probs[name]
            item[f"{name}_delta"] = candidate_delta[name]
        audit_rows.append(item)
    return audit_rows


def write_csv(rows: Sequence[Dict], candidate_names: Sequence[str], path: Path) -> None:
    fieldnames = [
        "id",
        "label",
        "label_name",
        "rough_domain",
        "bucket",
        "step7_pred",
        "step7_prob",
        "step7_correct",
    ]
    for name in candidate_names:
        fieldnames.extend([f"{name}_pred", f"{name}_prob", f"{name}_delta"])
    fieldnames.extend(
        [
            "fixed_by_candidates",
            "broken_by_candidates",
            "is_step7_fp",
            "is_step7_fn",
            "is_new_fp",
            "is_new_fn",
            "notes",
            "text",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["fixed_by_candidates"] = json_list(row["fixed_by_candidates"])
            out["broken_by_candidates"] = json_list(row["broken_by_candidates"])
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def bucket_summary(rows: Sequence[Dict], candidate_names: Sequence[str]) -> Dict:
    summary = defaultdict(lambda: defaultdict(int))
    for row in rows:
        bucket = row["bucket"]
        summary[bucket]["total"] += 1
        summary[bucket][f"label_{row['label']}"] += 1
        summary[bucket]["step7_errors"] += int(row["step7_correct"] == 0)
        summary[bucket]["step7_fp"] += int(row["is_step7_fp"])
        summary[bucket]["step7_fn"] += int(row["is_step7_fn"])
        for name in candidate_names:
            delta = row[f"{name}_delta"]
            summary[bucket][f"{name}_{delta}"] += 1
            summary[bucket][f"{name}_fp"] += int(row["label"] == 0 and row[f"{name}_pred"] == 1)
            summary[bucket][f"{name}_fn"] += int(row["label"] == 1 and row[f"{name}_pred"] == 0)
    return {bucket: dict(values) for bucket, values in sorted(summary.items())}


def candidate_delta_summary(rows: Sequence[Dict], candidate_names: Sequence[str]) -> Dict:
    out = {}
    for name in candidate_names:
        fixed_rows = [row for row in rows if row[f"{name}_delta"] == "fixed_step7_error"]
        broken_rows = [row for row in rows if row[f"{name}_delta"] == "broke_step7_correct"]
        out[name] = {
            "fixed_step7_errors": len(fixed_rows),
            "fixed_step7_fn": sum(1 for row in fixed_rows if row["label"] == 1),
            "fixed_step7_fp": sum(1 for row in fixed_rows if row["label"] == 0),
            "broke_step7_correct": len(broken_rows),
            "induced_fp": sum(1 for row in broken_rows if row["label"] == 0),
            "induced_fn": sum(1 for row in broken_rows if row["label"] == 1),
            "fixed_ids": [row["id"] for row in fixed_rows],
            "broken_ids": [row["id"] for row in broken_rows],
            "fixed_by_bucket": dict(Counter(row["bucket"] for row in fixed_rows)),
            "broken_by_bucket": dict(Counter(row["bucket"] for row in broken_rows)),
        }
    return out


def write_json(summary: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def ids_preview(ids: Sequence[str], limit: int = 16) -> str:
    if not ids:
        return "none"
    shown = ", ".join(ids[:limit])
    if len(ids) > limit:
        shown += f", ... (+{len(ids) - limit})"
    return shown


def table_row(values: Sequence[str]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def write_markdown(report: Dict, path: Path) -> None:
    candidate_names = report["candidate_names"]
    lines = [
        "# Round3 Phase A Error-Delta Audit",
        "",
        "This is a diagnostic-only teacher-test audit. It compares Round2 candidates",
        "against the Step7 baseline to identify repaired false negatives and newly",
        "induced false positives. These labels must not be used for later tuning.",
        "",
        "## Metrics",
        "",
        table_row(["Candidate", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "FP", "FN", "Confusion"]),
        table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for name, block in report["metrics"].items():
        lines.append(
            table_row(
                [
                    name,
                    fmt(block["accuracy"]),
                    fmt(block["precision"]),
                    fmt(block["recall"]),
                    fmt(block["f1"]),
                    fmt(block["roc_auc"]),
                    block["fp"],
                    block["fn"],
                    block["confusion_matrix"],
                ]
            )
        )

    lines.extend(["", "## Delta Versus Step7", ""])
    lines.append(
        table_row(
            [
                "Candidate",
                "Fixed Step7 errors",
                "Fixed FN",
                "Fixed FP",
                "Broke Step7 correct",
                "Induced FP",
                "Induced FN",
                "Fixed buckets",
                "Broken buckets",
            ]
        )
    )
    lines.append(table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---", "---"]))
    for name in candidate_names:
        block = report["candidate_delta"][name]
        lines.append(
            table_row(
                [
                    name,
                    block["fixed_step7_errors"],
                    block["fixed_step7_fn"],
                    block["fixed_step7_fp"],
                    block["broke_step7_correct"],
                    block["induced_fp"],
                    block["induced_fn"],
                    block["fixed_by_bucket"],
                    block["broken_by_bucket"],
                ]
            )
        )

    lines.extend(["", "## Fixed FN Patterns", ""])
    fixed_fn_rows = [
        row
        for row in report["audit_rows"]
        if row["is_step7_fn"] and row["fixed_by_candidates"]
    ]
    for row in fixed_fn_rows:
        lines.append(
            f"- id `{row['id']}` bucket `{row['bucket']}` fixed by "
            f"{', '.join(row['fixed_by_candidates'])}: {row['text_preview']}"
        )
    if not fixed_fn_rows:
        lines.append("- none")

    lines.extend(["", "## Candidate-Induced Human FP Patterns", ""])
    new_fp_rows = [
        row
        for row in report["audit_rows"]
        if row["is_new_fp"]
    ]
    for row in new_fp_rows:
        lines.append(
            f"- id `{row['id']}` bucket `{row['bucket']}` broken by "
            f"{', '.join(row['broken_by_candidates'])}: {row['text_preview']}"
        )
    if not new_fp_rows:
        lines.append("- none")

    lines.extend(["", "## Bucket Summary", ""])
    for bucket, block in report["bucket_summary"].items():
        lines.append(
            f"- `{bucket}`: total={block.get('total', 0)}, "
            f"step7_errors={block.get('step7_errors', 0)}, "
            f"step7_fp={block.get('step7_fp', 0)}, step7_fn={block.get('step7_fn', 0)}"
        )

    lines.extend(
        [
            "",
            "## Working Interpretation",
            "",
            "- Safe override candidates should come from rows where Step7 says human,",
            "  at least one candidate fixes the LLM label, and the same bucket does not",
            "  show a large candidate-induced human FP pattern.",
            "- Buckets with candidate-induced human FP should become precision-guard",
            "  buckets in Phase B/E rather than global threshold-lowering targets.",
            "- RoBERTa-style gains should be treated as diagnostic unless they can be",
            "  paired with hard-negative protection.",
            "",
            "## Output Files",
            "",
            f"- CSV: `{report['output_csv']}`",
            f"- JSON: `{report['output_json']}`",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Round3 Phase A: audit Round2 candidate error deltas.")
    parser.add_argument("--step7", default=str(DEFAULT_STEP7))
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output_md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    return parser.parse_args()


def main():
    args = parse_args()
    step7_path = Path(args.step7)
    candidate_specs = [parse_run(value) for value in args.candidates]
    step7_rows = normalized_prediction_rows(step7_path)
    candidates = {name: normalized_prediction_rows(path) for name, path in candidate_specs}
    candidate_names = [name for name, _path in candidate_specs]

    for name, rows in candidates.items():
        missing = set(step7_rows) - set(rows)
        extra = set(rows) - set(step7_rows)
        if missing or extra:
            raise ValueError(f"{name} id mismatch: missing={len(missing)}, extra={len(extra)}")

    audit_rows = build_audit_rows(step7_rows, candidates)
    rows_by_run = {"step7": [step7_rows[row_id] for row_id in sorted(step7_rows, key=sort_key)]}
    for name, rows in candidates.items():
        rows_by_run[name] = [rows[row_id] for row_id in sorted(step7_rows, key=sort_key)]

    metrics = {name: metric_block(rows) for name, rows in rows_by_run.items()}
    candidate_delta = candidate_delta_summary(audit_rows, candidate_names)
    by_bucket = bucket_summary(audit_rows, candidate_names)
    report = {
        "step7": str(step7_path),
        "candidates": {name: str(path) for name, path in candidate_specs},
        "candidate_names": candidate_names,
        "metrics": metrics,
        "candidate_delta": candidate_delta,
        "bucket_summary": by_bucket,
        "audit_rows": audit_rows,
        "output_csv": str(Path(args.output_csv)),
        "output_json": str(Path(args.output_json)),
    }

    write_csv(audit_rows, candidate_names, Path(args.output_csv))
    write_json(
        {
            "metrics": metrics,
            "candidate_delta": candidate_delta,
            "bucket_summary": by_bucket,
            "fixed_fn_ids_any_candidate": [
                row["id"] for row in audit_rows if row["is_step7_fn"] and row["fixed_by_candidates"]
            ],
            "candidate_induced_fp_ids_any_candidate": [
                row["id"] for row in audit_rows if row["is_new_fp"]
            ],
        },
        Path(args.output_json),
    )
    write_markdown(report, Path(args.output_md))

    print("=" * 70)
    print("Round3 Phase A error-delta audit written")
    print("=" * 70)
    print(f"Rows: {len(audit_rows)}")
    for name in ["step7"] + candidate_names:
        block = metrics[name]
        print(f"{name}: acc={block['accuracy']:.4f}, f1={block['f1']:.4f}, fp={block['fp']}, fn={block['fn']}")
    print(f"CSV: {args.output_csv}")
    print(f"Markdown: {args.output_md}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
