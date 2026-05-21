import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


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
    return name.strip(), Path(path.strip())


def row_matches(row: Dict, domain: str, generator: str) -> bool:
    if domain and row.get("domain") != domain:
        return False
    if generator and row.get("generator") != generator:
        return False
    return row.get("label") in [0, 1] and row.get("prediction") in [0, 1]


def metrics(rows: List[Dict]) -> Dict:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [int(row["prediction"]) for row in rows]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    fp_ids = [row.get("id", "") for row in rows if row["label"] == 0 and row["prediction"] == 1]
    fn_ids = [row.get("id", "") for row in rows if row["label"] == 1 and row["prediction"] == 0]
    return {
        "n": len(rows),
        "accuracy": accuracy_score(y_true, y_pred) if rows else None,
        "precision": precision_score(y_true, y_pred, zero_division=0) if rows else None,
        "recall": recall_score(y_true, y_pred, zero_division=0) if rows else None,
        "f1": f1_score(y_true, y_pred, zero_division=0) if rows else None,
        "confusion_matrix": cm,
        "false_positives": len(fp_ids),
        "false_negatives": len(fn_ids),
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
        "fn_by_generator": dict(Counter(row.get("generator", "unknown") for row in rows if row["label"] == 1 and row["prediction"] == 0)),
        "fp_by_source": dict(Counter(row.get("source", "unknown") for row in rows if row["label"] == 0 and row["prediction"] == 1)),
    }


def fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def write_markdown(report: Dict, output_path: Path) -> None:
    lines = [
        f"# {report['title']}",
        "",
        f"Domain filter: `{report['domain'] or 'any'}`",
        f"Generator filter: `{report['generator'] or 'any'}`",
        "",
        "## Metrics",
        "",
        "| Run | n | Accuracy | Precision | Recall | F1 | FP | FN | Confusion [[TN, FP], [FN, TP]] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, block in report["runs"].items():
        lines.append(
            f"| {name} | {block['n']} | {fmt(block['accuracy'])} | {fmt(block['precision'])} | "
            f"{fmt(block['recall'])} | {fmt(block['f1'])} | {block['false_positives']} | "
            f"{block['false_negatives']} | {block['confusion_matrix']} |"
        )

    lines.extend(["", "## Error IDs", ""])
    for name, block in report["runs"].items():
        lines.extend([f"### {name}", ""])
        lines.append(f"- False positives: {', '.join(block['fp_ids']) if block['fp_ids'] else 'none'}")
        lines.append(f"- False negatives: {', '.join(block['fn_ids']) if block['fn_ids'] else 'none'}")
        lines.append(f"- FN by generator: {block['fn_by_generator']}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize metrics and error IDs for a filtered subset.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--domain", default="")
    parser.add_argument("--generator", default="")
    parser.add_argument("--title", default="Subset Error Summary")
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_DIR / "subset_error_summary.json"))
    parser.add_argument("--output_md", default=str(DEFAULT_OUTPUT_DIR / "subset_error_summary.md"))
    return parser.parse_args()


def main():
    args = parse_args()
    run_specs = [parse_run_arg(value) for value in args.runs]
    report = {
        "title": args.title,
        "domain": args.domain,
        "generator": args.generator,
        "runs": {},
    }
    for name, path in run_specs:
        rows = [row for row in load_jsonl(path) if row_matches(row, args.domain, args.generator)]
        report["runs"][name] = metrics(rows)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output_md)

    print("=" * 70)
    print("Subset error summary finished")
    print("=" * 70)
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
