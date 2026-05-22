import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTION_FILES = {
    "internal_test": PROJECT_ROOT / "outputs" / "predictions" / "round6_safe_selector_internal_test_predictions.jsonl",
    "hardpos": PROJECT_ROOT / "outputs" / "predictions" / "round6_safe_selector_hardpos_predictions.jsonl",
    "hardneg": PROJECT_ROOT / "outputs" / "predictions" / "round6_safe_selector_hardneg_predictions.jsonl",
}
DEFAULT_LEDGER = PROJECT_ROOT / "outputs" / "evaluation" / "round5_flip_ledger.jsonl"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "outputs" / "evaluation" / "round7_exact_candidate_audit.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round7_exact_candidate_audit.md"

SAFE_FLIP = "fixed_fn_candidate"
UNSAFE_FLIP = "induced_fp"
EXACT_FLIPS = {SAFE_FLIP, UNSAFE_FLIP}
HISTOGRAM_EDGES = [round(index / 10, 1) for index in range(11)]


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


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def row_id(row: Dict, index: int) -> str:
    value = row.get("id")
    return str(value) if value not in [None, ""] else f"row_{index:06d}"


def prob(row: Dict, key: str) -> Optional[float]:
    value = row.get(key)
    if value in [None, ""]:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_value(row: Dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key)
        if value in [None, ""]:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safety(row: Dict) -> str:
    flip_type = str(row.get("flip_type") or "")
    if flip_type == SAFE_FLIP:
        return "safe_override"
    if flip_type == UNSAFE_FLIP:
        return "unsafe_override"
    return str(row.get("override_safety") or "unknown")


def selector_pass(row: Dict) -> bool:
    if row.get("safe_selector_pass") not in [None, ""]:
        return bool(int_value(row, "safe_selector_pass"))
    threshold = prob(row, "safe_selector_threshold")
    p_safe = prob(row, "p_safe_override")
    return bool(threshold is not None and p_safe is not None and p_safe >= threshold)


def index_ledger(rows: Sequence[Dict]) -> Tuple[Dict[Tuple[str, str], Dict], int]:
    indexed = {}
    duplicate_keys = 0
    for index, row in enumerate(rows):
        key = (str(row.get("split") or "unknown"), row_id(row, index))
        if key in indexed:
            duplicate_keys += 1
        indexed[key] = row
    return indexed, duplicate_keys


def merge_metadata(prediction: Dict, ledger: Optional[Dict], split_name: str, index: int) -> Dict:
    item = dict(ledger or {})
    item.update(prediction)
    item["id"] = row_id(item, index)
    item["split"] = str(item.get("split") or split_name)
    item["round7_exact_candidate"] = bool(
        item.get("flip_type") in EXACT_FLIPS
        and int_value(item, "step7_pred") == 0
        and int_value(item, "round4_pred") == 1
    )
    item["round7_override_safety"] = safety(item)
    item["round7_selector_pass"] = selector_pass(item)
    return item


def load_prediction_rows(split_specs: Sequence[Tuple[str, Path]], ledger_by_key: Dict[Tuple[str, str], Dict]) -> Tuple[List[Dict], Dict]:
    rows = []
    diagnostics = {}
    for split_name, path in split_specs:
        prediction_rows = load_jsonl(path)
        missing_ledger = 0
        exact_rows = 0
        for index, prediction in enumerate(prediction_rows):
            key = (split_name, row_id(prediction, index))
            ledger = ledger_by_key.get(key)
            if ledger is None:
                missing_ledger += 1
            item = merge_metadata(prediction, ledger, split_name, index)
            if item["round7_exact_candidate"]:
                exact_rows += 1
            rows.append(item)
        diagnostics[split_name] = {
            "prediction_file": str(path),
            "prediction_rows": len(prediction_rows),
            "missing_round5_ledger_rows": missing_ledger,
            "exact_candidate_rows": exact_rows,
        }
    return rows, diagnostics


def nested_counts(rows: Sequence[Dict], outer_key: str, inner_key: str) -> Dict[str, Dict[str, int]]:
    counts = defaultdict(Counter)
    for row in rows:
        counts[str(row.get(outer_key) or "unknown")][str(row.get(inner_key) or "unknown")] += 1
    return {outer: dict(sorted(inner.items())) for outer, inner in sorted(counts.items())}


def selector_outcome_summary(rows: Sequence[Dict]) -> Dict[str, Dict]:
    by_split = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split") or "unknown")].append(row)

    summary = {}
    for split_name, split_rows in sorted(by_split.items()):
        safe_rows = [row for row in split_rows if row["round7_override_safety"] == "safe_override"]
        unsafe_rows = [row for row in split_rows if row["round7_override_safety"] == "unsafe_override"]
        safe_passed = sum(1 for row in safe_rows if row["round7_selector_pass"])
        unsafe_blocked = sum(1 for row in unsafe_rows if not row["round7_selector_pass"])
        summary[split_name] = {
            "safe_total": len(safe_rows),
            "safe_passed": safe_passed,
            "safe_blocked": len(safe_rows) - safe_passed,
            "safe_pass_rate": safe_passed / len(safe_rows) if safe_rows else None,
            "unsafe_total": len(unsafe_rows),
            "unsafe_blocked": unsafe_blocked,
            "unsafe_false_passed": len(unsafe_rows) - unsafe_blocked,
            "unsafe_block_rate": unsafe_blocked / len(unsafe_rows) if unsafe_rows else None,
        }
    return summary


def histogram(values: Sequence[Optional[float]]) -> Dict[str, int]:
    counts = Counter()
    missing = 0
    for value in values:
        if value is None:
            missing += 1
            continue
        clipped = min(1.0, max(0.0, float(value)))
        index = min(int(clipped * 10), 9)
        lower = HISTOGRAM_EDGES[index]
        upper = HISTOGRAM_EDGES[index + 1]
        label = f"[{lower:.1f}, {upper:.1f}{']' if index == 9 else ')'}"
        counts[label] += 1
    ordered = {f"[{HISTOGRAM_EDGES[index]:.1f}, {HISTOGRAM_EDGES[index + 1]:.1f}{']' if index == 9 else ')'}": counts.get(
        f"[{HISTOGRAM_EDGES[index]:.1f}, {HISTOGRAM_EDGES[index + 1]:.1f}{']' if index == 9 else ')'}",
        0,
    ) for index in range(10)}
    if missing:
        ordered["missing"] = missing
    return ordered


def probability_histograms(rows: Sequence[Dict]) -> Dict[str, Dict[str, Dict[str, int]]]:
    out = {}
    for key in ["p_safe_override", "p_unsafe_override", "round4_prob"]:
        out[key] = {}
        for name in ["safe_override", "unsafe_override"]:
            subset = [row for row in rows if row["round7_override_safety"] == name]
            out[key][name] = histogram([prob(row, key) for row in subset])
    return out


def case_summary(row: Dict) -> Dict:
    return {
        "split": str(row.get("split") or "unknown"),
        "id": str(row.get("id") or ""),
        "pair_id": str(row.get("pair_id") or ""),
        "round4_bucket": str(row.get("round4_bucket") or row.get("bucket") or "unknown"),
        "bucket": str(row.get("bucket") or "unknown"),
        "domain": str(row.get("domain") or "unknown"),
        "generator": str(row.get("generator") or "unknown"),
        "source": str(row.get("source") or "unknown"),
        "word_count": int_value(row, "word_count"),
        "step7_prob": prob(row, "step7_prob"),
        "round4_prob": prob(row, "round4_prob"),
        "prob_delta": prob(row, "prob_delta"),
        "p_unsafe_override": prob(row, "p_unsafe_override"),
        "p_safe_override": prob(row, "p_safe_override"),
        "safe_selector_threshold": prob(row, "safe_selector_threshold"),
    }


def case_lists(rows: Sequence[Dict], case_limit: int) -> Dict[str, List[Dict]]:
    unsafe_false_pass = [
        row for row in rows if row["round7_override_safety"] == "unsafe_override" and row["round7_selector_pass"]
    ]
    safe_false_block = [
        row for row in rows if row["round7_override_safety"] == "safe_override" and not row["round7_selector_pass"]
    ]
    unsafe_false_pass.sort(key=lambda row: (-(prob(row, "p_safe_override") or -1.0), str(row.get("id") or "")))
    safe_false_block.sort(key=lambda row: ((prob(row, "p_safe_override") or 2.0), str(row.get("id") or "")))
    return {
        "selector_false_pass_unsafe_cases": [case_summary(row) for row in unsafe_false_pass[:case_limit]],
        "selector_false_block_safe_cases": [case_summary(row) for row in safe_false_block[:case_limit]],
    }


def render_rate(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_count_table(lines: List[str], title: str, counts: Dict[str, Dict[str, int]], inner_label: str) -> None:
    lines.extend(["", f"## {title}", "", f"| Split | {inner_label} | Rows |", "| --- | --- | ---: |"])
    for split_name, inner in counts.items():
        for key, value in inner.items():
            lines.append(f"| {split_name} | {key} | {value} |")


def write_histogram_table(lines: List[str], title: str, histograms: Dict[str, Dict[str, int]]) -> None:
    lines.extend(["", f"### {title}", "", "| Bin | Safe | Unsafe |", "| --- | ---: | ---: |"])
    safe_counts = histograms["safe_override"]
    unsafe_counts = histograms["unsafe_override"]
    bins = list(dict.fromkeys(list(safe_counts.keys()) + list(unsafe_counts.keys())))
    for name in bins:
        lines.append(f"| {name} | {safe_counts.get(name, 0)} | {unsafe_counts.get(name, 0)} |")


def write_case_table(lines: List[str], title: str, rows: Sequence[Dict]) -> None:
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| Split | Id | Round4 bucket | p_safe | p_unsafe | Round4 prob | Delta | Words |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['id']} | {row['round4_bucket']} | "
            f"{row['p_safe_override'] if row['p_safe_override'] is not None else 'NA'} | "
            f"{row['p_unsafe_override'] if row['p_unsafe_override'] is not None else 'NA'} | "
            f"{row['round4_prob'] if row['round4_prob'] is not None else 'NA'} | "
            f"{row['prob_delta'] if row['prob_delta'] is not None else 'NA'} | {row['word_count']} |"
        )


def write_markdown(report: Dict, path: Path) -> None:
    internal = report["selector_outcomes"].get("internal_test", {})
    lines = [
        "# Round7 Exact Candidate Audit",
        "",
        "This audit uses non-teacher Round5/Round6 artifacts only. Case tables omit row text so the report stays a decision-surface diagnostic.",
        "",
        "## Decision Surface",
        "",
        "```text",
        "exact candidate = Step7 predicts human and Round4 predicts LLM",
        "safe override = original label is LLM",
        "unsafe override = original label is human",
        "```",
        "",
        "## Round6 Exact Selector Outcome",
        "",
        "| Split | Safe total | Safe passed | Safe blocked | Safe pass rate | Unsafe total | Unsafe blocked | Unsafe false-pass | Unsafe block rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name, block in report["selector_outcomes"].items():
        lines.append(
            f"| {split_name} | {block['safe_total']} | {block['safe_passed']} | {block['safe_blocked']} | "
            f"{render_rate(block['safe_pass_rate'])} | {block['unsafe_total']} | {block['unsafe_blocked']} | "
            f"{block['unsafe_false_passed']} | {render_rate(block['unsafe_block_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Key Diagnostic",
            "",
            "```text",
            f"internal_test exact unsafe blocked by Round6 safe selector = {internal.get('unsafe_blocked', 0)} / {internal.get('unsafe_total', 0)}",
            "Round7 should train and validate on exact disagreement candidates before another selector search.",
            "```",
        ]
    )
    write_count_table(lines, "Exact Candidates By Split And Safety", report["candidate_counts_by_split_safety"], "Safety")
    write_count_table(lines, "Exact Candidates By Split And Round4 Bucket", report["candidate_counts_by_split_bucket"], "Round4 bucket")
    lines.extend(["", "## Probability Histograms"])
    for key, histograms in report["probability_histograms"].items():
        write_histogram_table(lines, key, histograms)
    write_case_table(lines, "Selector False-Pass Unsafe Cases", report["cases"]["selector_false_pass_unsafe_cases"])
    write_case_table(lines, "Selector False-Block Safe Cases", report["cases"]["selector_false_block_safe_cases"])
    lines.extend(["", "## Input Alignment", "", "| Split | Prediction rows | Exact candidates | Missing Round5 ledger rows |", "| --- | ---: | ---: | ---: |"])
    for split_name, block in report["input_diagnostics"]["prediction_files"].items():
        lines.append(
            f"| {split_name} | {block['prediction_rows']} | {block['exact_candidate_rows']} | "
            f"{block['missing_round5_ledger_rows']} |"
        )
    lines.extend(["", "## Decision", "", "```text", report["decision"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_split_spec(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--prediction requires SPLIT=PATH")
    split_name, raw_path = value.split("=", 1)
    split_name = split_name.strip()
    if not split_name or not raw_path.strip():
        raise ValueError("--prediction requires a non-empty SPLIT and PATH")
    return split_name, Path(raw_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Round6 exact disagreement candidates before Round7 data work.")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="Prediction input as SPLIT=PATH. Defaults to the Round6 internal_test/hardpos/hardneg selector outputs.",
    )
    parser.add_argument("--report_json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--case_limit", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    ledger_rows = load_jsonl(Path(args.ledger))
    ledger_by_key, duplicate_ledger_keys = index_ledger(ledger_rows)
    split_specs = (
        [parse_split_spec(value) for value in args.prediction]
        if args.prediction
        else list(DEFAULT_PREDICTION_FILES.items())
    )
    merged_rows, prediction_diagnostics = load_prediction_rows(split_specs, ledger_by_key)
    exact_rows = [row for row in merged_rows if row["round7_exact_candidate"]]
    report = {
        "inputs": {
            "round5_flip_ledger": str(Path(args.ledger)),
            "prediction_files": {split_name: str(path) for split_name, path in split_specs},
            "teacher_test_used": False,
        },
        "definition": {
            "exact_candidate": "step7_pred=0 and round4_pred=1",
            "safe_override": SAFE_FLIP,
            "unsafe_override": UNSAFE_FLIP,
        },
        "num_merged_prediction_rows": len(merged_rows),
        "num_exact_candidate_rows": len(exact_rows),
        "candidate_counts_by_split_safety": nested_counts(exact_rows, "split", "round7_override_safety"),
        "candidate_counts_by_split_bucket": nested_counts(exact_rows, "split", "round4_bucket"),
        "candidate_counts_by_safety_bucket": nested_counts(exact_rows, "round7_override_safety", "round4_bucket"),
        "selector_outcomes": selector_outcome_summary(exact_rows),
        "probability_histograms": probability_histograms(exact_rows),
        "cases": case_lists(exact_rows, args.case_limit),
        "input_diagnostics": {
            "round5_ledger_rows": len(ledger_rows),
            "duplicate_round5_ledger_keys": duplicate_ledger_keys,
            "prediction_files": prediction_diagnostics,
        },
    }
    internal = report["selector_outcomes"].get("internal_test", {})
    report["decision"] = (
        "ROUND7_PHASE0_AUDIT = complete; "
        f"Round6 internal exact unsafe block = {internal.get('unsafe_blocked', 0)} / {internal.get('unsafe_total', 0)}. "
        "Proceed to exact-candidate dataset construction before any new selector training."
    )
    write_json(report, Path(args.report_json))
    write_markdown(report, Path(args.report_md))
    print("=" * 70)
    print("Round7 exact-candidate audit complete")
    print("=" * 70)
    print(f"Exact candidate rows: {len(exact_rows)}")
    print(
        "Internal exact unsafe blocked: "
        f"{internal.get('unsafe_blocked', 0)} / {internal.get('unsafe_total', 0)}"
    )
    print(f"JSON report: {args.report_json}")
    print(f"Markdown report: {args.report_md}")


if __name__ == "__main__":
    main()
