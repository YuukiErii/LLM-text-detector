import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_gemini.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_gemini.jsonl"
DEFAULT_FAILED_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_gemini_failed.jsonl"

DEFAULT_RERUN_PROMPTS_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_gemini_literature_rerun.jsonl"
DEFAULT_RERUN_TASK_IDS_PATH = PROJECT_ROOT / "data" / "processed" / "gemini_literature_rerun_task_ids.txt"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "gemini_literature_rerun_report.json"


VALID_ENDINGS = (".", "?", "!", ")", "]", '"', "'", "”", "’")


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"[Warning] File does not exist: {path}")
        return []

    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse {path}, line {line_id}: {e}")

    return samples


def save_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_task_ids(task_ids: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for task_id in task_ids:
            f.write(task_id + "\n")


def word_count(text: str) -> int:
    return len(text.split())


def lexical_jaccard(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[A-Za-z']+", a.lower()))
    tokens_b = set(re.findall(r"[A-Za-z']+", b.lower()))

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def looks_truncated(text: str) -> bool:
    text = str(text or "").strip()

    if not text:
        return True

    if text[-1] not in VALID_ENDINGS:
        return True

    # Obvious dangling endings.
    dangling_patterns = [
        r"\b(and|or|but|because|while|although|though|that|which|who|whom|whose|where|when|if|to|of|in|on|at|by|with|from|into|for|as|than|the|a|an)\s*$",
        r"[,;:]\s*$",
        r"\b(can|could|would|should|may|might|must|shall|will|is|are|was|were|be|been|being|have|has|had|do|does|did)\s*$",
    ]

    lower = text.lower()
    for pattern in dangling_patterns:
        if re.search(pattern, lower):
            return True

    return False


def is_failed_quality(sample: Dict) -> bool:
    quality = sample.get("quality", {})

    if not isinstance(quality, dict):
        return False

    return quality.get("passed_basic_quality_check") is False


def should_rerun_output_sample(sample: Dict, min_length_ratio: float, max_jaccard: float) -> List[str]:
    reasons = []

    task_id = sample.get("task_id")
    text = sample.get("text", "")
    source_text = sample.get("source_text", "")

    if not task_id:
        reasons.append("missing_task_id")

    if not isinstance(text, str) or not text.strip():
        reasons.append("empty_text")
        return reasons

    if looks_truncated(text):
        reasons.append("possibly_truncated")

    source_words = word_count(source_text)
    rewrite_words = word_count(text)
    length_ratio = rewrite_words / max(source_words, 1)

    if length_ratio < min_length_ratio:
        reasons.append("too_short_relative")

    jaccard = lexical_jaccard(source_text, text)
    if jaccard > max_jaccard:
        reasons.append("too_similar_to_source")

    quality = sample.get("quality", {})
    if isinstance(quality, dict):
        issues = quality.get("quality_issues", [])
        if isinstance(issues, list):
            for issue in issues:
                if issue not in reasons:
                    reasons.append(issue)

        if quality.get("passed_basic_quality_check") is False:
            if "quality_failed" not in reasons:
                reasons.append("quality_failed")

    finish_reason = sample.get("finish_reason")
    if finish_reason == "length":
        reasons.append("finish_reason_length")

    return reasons


def collect_failed_task_ids(failed_samples: List[Dict]) -> Dict[str, List[str]]:
    task_to_reasons = {}

    for sample in failed_samples:
        task_id = sample.get("task_id")
        if not task_id:
            continue

        reasons = []

        error = str(sample.get("error", ""))
        if error:
            reasons.append(error)

        finish_reason = sample.get("finish_reason")
        if finish_reason:
            reasons.append(f"finish_reason={finish_reason}")

        quality = sample.get("quality", {})
        if isinstance(quality, dict):
            issues = quality.get("quality_issues", [])
            if isinstance(issues, list):
                reasons.extend(issues)

        if not reasons:
            reasons.append("failed_file_record")

        task_to_reasons[task_id] = reasons

    return task_to_reasons


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare rerun prompts for Gemini literature by collecting failed and possibly truncated samples."
        )
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default=str(DEFAULT_PROMPTS_PATH),
        help="Original rewrite prompts for Gemini literature.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Current Gemini literature output JSONL.",
    )

    parser.add_argument(
        "--failed",
        type=str,
        default=str(DEFAULT_FAILED_PATH),
        help="Current Gemini literature failed JSONL.",
    )

    parser.add_argument(
        "--rerun_prompts",
        type=str,
        default=str(DEFAULT_RERUN_PROMPTS_PATH),
        help="Output JSONL containing prompts to rerun.",
    )

    parser.add_argument(
        "--rerun_task_ids",
        type=str,
        default=str(DEFAULT_RERUN_TASK_IDS_PATH),
        help="Output TXT containing task_ids to rerun.",
    )

    parser.add_argument(
        "--report",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help="Output rerun report JSON.",
    )

    parser.add_argument(
        "--min_length_ratio",
        type=float,
        default=0.55,
        help="Rerun samples with rewrite/source length ratio below this threshold.",
    )

    parser.add_argument(
        "--max_jaccard",
        type=float,
        default=0.82,
        help="Rerun samples with lexical Jaccard above this threshold.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    prompts_path = Path(args.prompts)
    output_path = Path(args.output)
    failed_path = Path(args.failed)
    rerun_prompts_path = Path(args.rerun_prompts)
    rerun_task_ids_path = Path(args.rerun_task_ids)
    report_path = Path(args.report)

    prompts = load_jsonl(prompts_path)
    output_samples = load_jsonl(output_path)
    failed_samples = load_jsonl(failed_path)

    prompt_by_task_id = {
        sample.get("task_id"): sample
        for sample in prompts
        if sample.get("task_id")
    }

    rerun_reasons = {}

    # 1. Collect bad / suspicious samples from successful output file.
    for sample in output_samples:
        task_id = sample.get("task_id")
        if not task_id:
            continue

        reasons = should_rerun_output_sample(
            sample=sample,
            min_length_ratio=args.min_length_ratio,
            max_jaccard=args.max_jaccard,
        )

        if reasons:
            rerun_reasons.setdefault(task_id, [])
            rerun_reasons[task_id].extend(reasons)

    # 2. Collect all failed samples from failed file.
    failed_task_reasons = collect_failed_task_ids(failed_samples)
    for task_id, reasons in failed_task_reasons.items():
        rerun_reasons.setdefault(task_id, [])
        rerun_reasons[task_id].extend(reasons)

    # 3. Deduplicate reasons.
    for task_id, reasons in rerun_reasons.items():
        seen = set()
        deduped = []
        for reason in reasons:
            reason = str(reason)
            if reason not in seen:
                seen.add(reason)
                deduped.append(reason)
        rerun_reasons[task_id] = deduped

    rerun_task_ids = sorted(rerun_reasons.keys())

    rerun_prompts = []
    missing_prompt_task_ids = []

    for task_id in rerun_task_ids:
        prompt_item = prompt_by_task_id.get(task_id)

        if prompt_item is None:
            missing_prompt_task_ids.append(task_id)
            continue

        item = dict(prompt_item)
        item["rerun_reasons"] = rerun_reasons[task_id]
        item["rerun_source"] = "gemini_literature_v1_failed_or_truncated"
        rerun_prompts.append(item)

    save_jsonl(rerun_prompts, rerun_prompts_path)
    save_task_ids([item["task_id"] for item in rerun_prompts], rerun_task_ids_path)

    reason_counter = Counter()
    for reasons in rerun_reasons.values():
        for reason in reasons:
            reason_counter[reason] += 1

    report = {
        "prompts_path": str(prompts_path),
        "output_path": str(output_path),
        "failed_path": str(failed_path),
        "rerun_prompts_path": str(rerun_prompts_path),
        "rerun_task_ids_path": str(rerun_task_ids_path),
        "total_original_prompts": len(prompts),
        "total_output_samples": len(output_samples),
        "total_failed_samples": len(failed_samples),
        "total_rerun_task_ids": len(rerun_task_ids),
        "total_rerun_prompts_saved": len(rerun_prompts),
        "missing_prompt_task_ids": missing_prompt_task_ids,
        "min_length_ratio": args.min_length_ratio,
        "max_jaccard": args.max_jaccard,
        "reason_distribution": dict(reason_counter),
        "sample_rerun_task_ids": rerun_task_ids[:20],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Prepare Gemini Literature Rerun Prompts")
    print("=" * 70)
    print(f"Original prompts: {len(prompts)}")
    print(f"Current output samples: {len(output_samples)}")
    print(f"Current failed samples: {len(failed_samples)}")
    print(f"Rerun task ids: {len(rerun_task_ids)}")
    print(f"Rerun prompts saved: {len(rerun_prompts)}")
    print(f"Rerun prompts path: {rerun_prompts_path}")
    print(f"Rerun task ids path: {rerun_task_ids_path}")
    print(f"Report path: {report_path}")

    print("\nReason distribution:")
    for reason, count in reason_counter.most_common():
        print(f"  {reason}: {count}")

    if missing_prompt_task_ids:
        print("\n[Warning] Missing prompts for task_ids:")
        for task_id in missing_prompt_task_ids[:20]:
            print(f"  {task_id}")
        if len(missing_prompt_task_ids) > 20:
            print(f"  ... and {len(missing_prompt_task_ids) - 20} more")


if __name__ == "__main__":
    main()