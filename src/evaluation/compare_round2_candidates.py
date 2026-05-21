import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round2_internal_comparison.md"
DEFAULT_OVERLAP_CSV = PROJECT_ROOT / "outputs" / "evaluation" / "round2_error_overlap_matrix.csv"


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
        raise ValueError(f"Run spec must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def normalize_rows(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for index, row in enumerate(rows):
        if row.get("label") not in [0, 1] or row.get("prediction") not in [0, 1]:
            continue
        item = dict(row)
        item["id"] = str(item.get("id", index))
        item["label"] = int(item["label"])
        item["prediction"] = int(item["prediction"])
        item["probability"] = float(item.get("probability", item.get("prob_llm", 0.0)))
        out.append(item)
    return out


def metrics(rows: Sequence[Dict]) -> Dict:
    y_true = np.array([row["label"] for row in rows], dtype=int)
    y_pred = np.array([row["prediction"] for row in rows], dtype=int)
    y_prob = np.array([row["probability"] for row in rows], dtype=float)
    fp_ids = [row["id"] for row in rows if row["label"] == 0 and row["prediction"] == 1]
    fn_ids = [row["id"] for row in rows if row["label"] == 1 and row["prediction"] == 0]
    out = {
        "n": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "false_positives": len(fp_ids),
        "false_negatives": len(fn_ids),
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
        "error_ids": sorted(set(fp_ids + fn_ids), key=lambda value: int(value) if value.isdigit() else value),
        "fn_by_generator": dict(Counter(row.get("generator", "unknown") for row in rows if row["label"] == 1 and row["prediction"] == 0)),
        "fp_by_bucket": dict(Counter(row.get("bucket", row.get("domain", "unknown")) for row in rows if row["label"] == 0 and row["prediction"] == 1)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = None
    return out


def error_set(block: Dict) -> set:
    return set(block["error_ids"])


def overlap_rows(report: Dict) -> List[Dict]:
    names = list(report["runs"])
    rows = []
    for left in names:
        left_set = error_set(report["runs"][left])
        row = {"run": left}
        for right in names:
            right_set = error_set(report["runs"][right])
            union = left_set | right_set
            inter = left_set & right_set
            row[right] = "" if not union else f"{len(inter)}|{len(inter) / len(union):.4f}"
        rows.append(row)
    return rows


def save_overlap_csv(report: Dict, path: Path) -> None:
    names = list(report["runs"])
    rows = overlap_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        f"# {report['title']}",
        "",
        "| Candidate | n | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN | Confusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, block in report["runs"].items():
        lines.append(
            f"| {name} | {block['n']} | {fmt(block['accuracy'])} | {fmt(block['precision'])} | "
            f"{fmt(block['recall'])} | {fmt(block['f1'])} | {fmt(block['roc_auc'])} | "
            f"{block['false_positives']} | {block['false_negatives']} | {block['confusion_matrix']} |"
        )
    lines.extend(["", "## Error Overlap", "", "Each cell is `overlap_count|jaccard`.", ""])
    names = list(report["runs"])
    lines.append("| Run | " + " | ".join(names) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in names) + " |")
    for row in overlap_rows(report):
        lines.append("| " + row["run"] + " | " + " | ".join(row[name] for name in names) + " |")
    lines.extend(["", "## Error Details", ""])
    for name, block in report["runs"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- False positives: {', '.join(block['fp_ids']) if block['fp_ids'] else 'none'}",
                f"- False negatives: {', '.join(block['fn_ids']) if block['fn_ids'] else 'none'}",
                f"- FN by generator: {block['fn_by_generator']}",
                f"- FP by bucket/domain: {block['fp_by_bucket']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare round2 candidate prediction files.")
    parser.add_argument("--runs", nargs="+", required=True, help="Candidate specs in NAME=PATH format.")
    parser.add_argument("--title", default="Round2 Candidate Comparison")
    parser.add_argument("--output_md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output_json", default="")
    parser.add_argument("--overlap_csv", default=str(DEFAULT_OVERLAP_CSV))
    return parser.parse_args()


def main():
    args = parse_args()
    report = {
        "title": args.title,
        "runs": {},
    }
    for name, path in [parse_run(value) for value in args.runs]:
        rows = normalize_rows(load_jsonl(path))
        if not rows:
            raise ValueError(f"No labeled prediction rows found for {name}: {path}")
        report["runs"][name] = metrics(rows)

    write_markdown(report, Path(args.output_md))
    save_overlap_csv(report, Path(args.overlap_csv))
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Round2 candidates compared")
    print("=" * 70)
    for name, block in report["runs"].items():
        print(f"{name}: f1={block['f1']:.4f} acc={block['accuracy']:.4f} FP={block['false_positives']} FN={block['false_negatives']}")
    print(f"Markdown: {args.output_md}")
    print(f"Overlap CSV: {args.overlap_csv}")


if __name__ == "__main__":
    main()
