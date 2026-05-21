import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "round2_human_hardneg_source.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_round2_chatgpt_hard_positive.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_round2_chatgpt_hard_positive_report.json"

DEFAULT_QUOTAS = (
    "chatgpt_conservative_literary=450,"
    "chatgpt_old_fiction=350,"
    "chatgpt_archaic_poetry=450,"
    "chatgpt_poetry_freeverse=250,"
    "chatgpt_natural_academic=500"
)

PROMPTS = {
    "chatgpt_conservative_literary": (
        "Rewrite the following literary passage as a conservative human-like paraphrase. "
        "Preserve the scene, narrator, mood, pacing, and approximate length. Keep the prose literary rather than explanatory. "
        "Change wording and syntax enough that it is a real rewrite, but do not summarize or modernize aggressively. "
        "Return only the rewritten passage.\n\nPASSAGE:\n{text}"
    ),
    "chatgpt_old_fiction": (
        "Rewrite the following passage so it still reads like older literary fiction. Preserve formal or old-fashioned flavor, "
        "narrative distance, concrete details, and approximate length. Avoid modern template wording and avoid sounding like a summary. "
        "Return only the rewritten passage.\n\nPASSAGE:\n{text}"
    ),
    "chatgpt_archaic_poetry": (
        "Rewrite the following poem or stanza as poetry. Preserve line breaks as much as possible, keep archaic or formal poetic diction "
        "when present, and retain the same image sequence and emotional movement. Change the wording without turning it into prose. "
        "Return only the poem.\n\nPOEM:\n{text}"
    ),
    "chatgpt_poetry_freeverse": (
        "Rewrite the following poem or short lyrical fragment as a natural poem. Keep the lineated shape, speaker, imagery, and mood, "
        "but vary the diction and phrasing. Avoid explanation, commentary, or prose paraphrase. Return only the poem.\n\nPOEM:\n{text}"
    ),
    "chatgpt_natural_academic": (
        "Paraphrase the following NLP or computational-linguistics paragraph in polished academic English. Preserve all technical claims, "
        "terminology, methods, datasets, limitations, and caveats. Keep approximately the same length and avoid adding new claims. "
        "Return only the rewritten paragraph.\n\nPARAGRAPH:\n{text}"
    ),
}


def parse_quotas(value: str) -> Dict[str, int]:
    quotas = {}
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, raw_count = part.split("=", 1)
        elif ":" in part:
            key, raw_count = part.split(":", 1)
        else:
            raise ValueError(f"Quota must use name=count format, got: {part}")
        key = key.strip()
        if key not in PROMPTS:
            raise ValueError(f"Unsupported prompt bucket: {key}")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"Quota must be non-negative, got: {part}")
        quotas[key] = count
    return quotas


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_bucket(row: Dict) -> str:
    domain = str(row.get("domain", ""))
    bucket = assign_bucket(str(row.get("text", "")))
    if domain == "academic" or bucket == "academic_formal":
        return "chatgpt_natural_academic"
    if domain == "poetry" or bucket.startswith("poetry"):
        if bucket == "poetry_classical":
            return "chatgpt_archaic_poetry"
        return "chatgpt_poetry_freeverse"
    if bucket == "literary_old_prose":
        return "chatgpt_old_fiction"
    return "chatgpt_conservative_literary"


def expanded_buckets(row: Dict) -> List[str]:
    primary = source_bucket(row)
    if primary == "chatgpt_conservative_literary":
        return [primary, "chatgpt_old_fiction"]
    if primary == "chatgpt_poetry_freeverse":
        return [primary, "chatgpt_archaic_poetry"]
    return [primary]


def select_sources(rows: List[Dict], quotas: Dict[str, int], seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        if row.get("label") != 0:
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        for bucket in expanded_buckets(row):
            buckets[bucket].append(row)

    selected = {}
    for bucket, quota in quotas.items():
        candidates = list(buckets.get(bucket, []))
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise ValueError(f"Not enough rows for {bucket}: need {quota}, found {len(candidates)}")
        selected[bucket] = candidates[:quota]
    return selected


def build_task(row: Dict, prompt_bucket: str, index: int) -> Dict:
    text = str(row.get("text", "")).strip()
    task_id = f"rewrite_round2_chatgpt_hardpos_{index:06d}"
    source_metadata = dict(row.get("metadata") or {})
    return {
        "task_id": task_id,
        "source_id": row.get("id"),
        "pair_id": row.get("pair_id"),
        "domain": row.get("domain", "unknown"),
        "source": row.get("source", "unknown"),
        "source_text": text,
        "prompt_type": prompt_bucket,
        "prompt": PROMPTS[prompt_bucket].format(text=text),
        "metadata": {
            "step": "round2_phase1_hard_positive",
            "target_generator": "chatgpt",
            "source_bucket": assign_bucket(text),
            "source_metadata": source_metadata,
        },
    }


def summarize(rows: List[Dict]) -> Dict:
    return {
        "num_rows": len(rows),
        "domain_distribution": dict(Counter(row.get("domain", "unknown") for row in rows)),
        "prompt_type_distribution": dict(Counter(row.get("prompt_type", "unknown") for row in rows)),
        "source_bucket_distribution": dict(
            Counter(row.get("metadata", {}).get("source_bucket", "unknown") for row in rows)
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare round2 ChatGPT hard-positive prompt tasks.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--quotas", default=DEFAULT_QUOTAS)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    quotas = parse_quotas(args.quotas)

    rows = load_records(input_path)
    selected = select_sources(rows, quotas=quotas, seed=args.seed)

    tasks = []
    for bucket, bucket_rows in selected.items():
        for row in bucket_rows:
            tasks.append(build_task(row, bucket, len(tasks) + 1))

    save_jsonl(tasks, output_path)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "quotas": quotas,
        "seed": args.seed,
        "available_source_buckets": dict(Counter(source_bucket(row) for row in rows if row.get("label") == 0)),
        "tasks": summarize(tasks),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Prepared round2 hard-positive prompts")
    print("=" * 70)
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Prompt distribution: {report['tasks']['prompt_type_distribution']}")


if __name__ == "__main__":
    main()
