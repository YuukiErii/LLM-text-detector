import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "round6_override_train.jsonl"
DEFAULT_STEP7 = PROJECT_ROOT / "outputs" / "predictions" / "round7_step7_round6_override_train_predictions.jsonl"
DEFAULT_ROUND4 = PROJECT_ROOT / "outputs" / "predictions" / "round7_round4_round6_override_train_predictions.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "round7_train_exact_like_mined.jsonl"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "outputs" / "evaluation" / "round7_train_exact_like_mine_report.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round7_train_exact_like_mine_report.md"


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


def row_id(row: Dict, index: int) -> str:
    value = row.get("id")
    return str(value) if value not in [None, ""] else f"row_{index:06d}"


def int_value(value, default: Optional[int] = None) -> Optional[int]:
    try:
        if value in [None, ""]:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def float_value(row: Dict, keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value not in [None, ""]:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def prediction_value(row: Dict) -> Optional[int]:
    for key in ["prediction", "pred", "label_pred"]:
        value = int_value(row.get(key))
        if value is not None:
            return value
    return None


def index_rows(rows: Sequence[Dict]) -> Tuple[Dict[str, Dict], int]:
    indexed = {}
    duplicate_ids = 0
    for index, row in enumerate(rows):
        key = row_id(row, index)
        if key in indexed:
            duplicate_ids += 1
        indexed[key] = row
    return indexed, duplicate_ids


def origin_allowed(row: Dict, allowed_origins: Sequence[str]) -> bool:
    if not allowed_origins:
        return True
    origin = str(row.get("round6_origin_split") or row.get("round7_origin_split") or "")
    return origin in set(allowed_origins)


def stage_allowed(row: Dict, excluded_stages: Sequence[str]) -> bool:
    if not excluded_stages:
        return True
    stage = str(row.get("round4_source_stage") or row.get("source_stage") or "")
    return stage not in set(excluded_stages)


def original_detection_label(row: Dict) -> Optional[int]:
    value = int_value(row.get("original_detection_label"))
    if value in [0, 1]:
        return value
    value = int_value(row.get("label"))
    return value if value in [0, 1] else None


def mined_row(source: Dict, step7: Dict, round4: Dict, row_key: str, index: int) -> Optional[Dict]:
    detection_label = original_detection_label(source)
    step7_pred = prediction_value(step7)
    round4_pred = prediction_value(round4)
    if detection_label not in [0, 1] or step7_pred is None or round4_pred is None:
        return None
    if step7_pred != 0 or round4_pred != 1:
        return None

    step7_prob = float_value(step7, ["prob_llm", "probability"])
    round4_prob = float_value(round4, ["round4_prob", "prob_llm", "probability"])
    flip_type = "fixed_fn_candidate" if detection_label == 1 else "induced_fp"
    safety = "safe_override" if detection_label == 1 else "unsafe_override"
    item = dict(source)
    item["id"] = row_key
    item["label"] = detection_label
    item["original_detection_label"] = detection_label
    item["split"] = "round7_train_exact_like_mined"
    item["round7_mine_index"] = index
    item["round7_mine_source"] = "round6_override_train_predictions"
    item["round7_mine_origin_split"] = str(source.get("round6_origin_split") or "unknown")
    item["round7_mine_source_kind"] = str(source.get("round6_source_kind") or "unknown")
    item["step7_pred"] = int(step7_pred)
    item["round4_pred"] = int(round4_pred)
    item["step7_prob"] = float(step7_prob)
    item["round4_prob"] = float(round4_prob)
    item["prob_delta"] = float(round4_prob - step7_prob)
    item["flip_type"] = flip_type
    item["override_safety"] = safety
    item["round4_override_candidate"] = True
    return item


def mine_candidates(
    source_rows: Sequence[Dict],
    step7_rows: Sequence[Dict],
    round4_rows: Sequence[Dict],
    origins: Sequence[str],
    excluded_stages: Sequence[str],
) -> Tuple[List[Dict], Dict]:
    step7_by_id, step7_dupes = index_rows(step7_rows)
    round4_by_id, round4_dupes = index_rows(round4_rows)
    candidates = []
    missing_step7 = []
    missing_round4 = []
    skipped_origin = 0
    skipped_stage = 0
    malformed = []
    for index, source in enumerate(source_rows):
        if not origin_allowed(source, origins):
            skipped_origin += 1
            continue
        if not stage_allowed(source, excluded_stages):
            skipped_stage += 1
            continue
        key = row_id(source, index)
        step7 = step7_by_id.get(key)
        round4 = round4_by_id.get(key)
        if step7 is None:
            missing_step7.append(key)
            continue
        if round4 is None:
            missing_round4.append(key)
            continue
        item = mined_row(source, step7, round4, key, index)
        if item is None:
            if original_detection_label(source) not in [0, 1] or prediction_value(step7) is None or prediction_value(round4) is None:
                malformed.append(key)
            continue
        candidates.append(item)
    diagnostics = {
        "source_rows": len(source_rows),
        "step7_rows": len(step7_rows),
        "round4_rows": len(round4_rows),
        "allowed_round6_origin_splits": list(origins),
        "excluded_round4_source_stages": list(excluded_stages),
        "skipped_source_origin_rows": skipped_origin,
        "skipped_source_stage_rows": skipped_stage,
        "missing_step7_rows": len(missing_step7),
        "missing_round4_rows": len(missing_round4),
        "malformed_rows": len(malformed),
        "step7_duplicate_ids": step7_dupes,
        "round4_duplicate_ids": round4_dupes,
        "example_missing_step7_ids": missing_step7[:10],
        "example_missing_round4_ids": missing_round4[:10],
        "example_malformed_ids": malformed[:10],
    }
    return candidates, diagnostics


def bucket_counts(rows: Sequence[Dict]) -> Dict[str, Dict[str, int]]:
    counts = {}
    for safety in ["safe_override", "unsafe_override"]:
        subset = [row for row in rows if row.get("override_safety") == safety]
        counts[safety] = dict(sorted(Counter(str(row.get("round4_bucket") or row.get("bucket") or "unknown") for row in subset).items()))
    return counts


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round7 Train Exact-Like Candidate Mine",
        "",
        "This mine aligns non-teacher Round6 train-pool rows with Step7 and Round4 predictions.",
        "",
        "## Candidate Counts",
        "",
        "| Safety | Rows |",
        "| --- | ---: |",
    ]
    for safety, count in report["safety_counts"].items():
        lines.append(f"| {safety} | {count} |")
    lines.extend(["", "## Round4 Bucket Counts", "", "| Safety | Round4 bucket | Rows |", "| --- | --- | ---: |"])
    for safety, buckets in report["round4_bucket_counts"].items():
        for bucket, count in buckets.items():
            lines.append(f"| {safety} | {bucket} | {count} |")
    lines.extend(["", "## Alignment", "", "| Item | Count |", "| --- | ---: |"])
    for key in [
        "source_rows",
        "step7_rows",
        "round4_rows",
        "skipped_source_origin_rows",
        "skipped_source_stage_rows",
        "missing_step7_rows",
        "missing_round4_rows",
        "malformed_rows",
    ]:
        lines.append(f"| {key} | {report['diagnostics'][key]} |")
    lines.extend(["", "## Decision", "", "```text", report["decision"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_origins(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Mine Round7 train-side exact-like Step7-vs-Round4 candidates.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--step7", default=str(DEFAULT_STEP7))
    parser.add_argument("--round4", default=str(DEFAULT_ROUND4))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report_json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--round6_origin_splits", default="round4_residual_train")
    parser.add_argument(
        "--exclude_round4_source_stages",
        default="",
        help="Comma-separated Round4 source stages to skip, such as step7_base_train.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_rows = load_jsonl(Path(args.source))
    step7_rows = load_jsonl(Path(args.step7))
    round4_rows = load_jsonl(Path(args.round4))
    origins = parse_origins(args.round6_origin_splits)
    excluded_stages = parse_origins(args.exclude_round4_source_stages)
    candidates, diagnostics = mine_candidates(source_rows, step7_rows, round4_rows, origins, excluded_stages)
    safety_counts = dict(sorted(Counter(str(row.get("override_safety") or "unknown") for row in candidates).items()))
    report = {
        "inputs": {
            "source": str(Path(args.source)),
            "step7_predictions": str(Path(args.step7)),
            "round4_predictions": str(Path(args.round4)),
        },
        "output": str(Path(args.output)),
        "num_candidates": len(candidates),
        "safety_counts": safety_counts,
        "round4_bucket_counts": bucket_counts(candidates),
        "diagnostics": diagnostics,
    }
    report["decision"] = (
        "ROUND7_TRAIN_EXACT_LIKE_MINE = complete; feed this non-teacher pool into the Round7 dataset report."
        if candidates
        else "ROUND7_TRAIN_EXACT_LIKE_MINE = empty; use a wider train-side source or generate exact-like rows."
    )
    save_jsonl(candidates, Path(args.output))
    write_json(report, Path(args.report_json))
    write_markdown(report, Path(args.report_md))
    print("=" * 70)
    print("Round7 train exact-like candidate mine complete")
    print("=" * 70)
    print(f"Candidates: {len(candidates)} safety={safety_counts}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report_json}")


if __name__ == "__main__":
    main()
