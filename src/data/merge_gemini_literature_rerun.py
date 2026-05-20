import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_V1_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_gemini.jsonl"
DEFAULT_RERUN_PROMPTS_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_gemini_literature_rerun.jsonl"
DEFAULT_RERUN_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_gemini_rerun.jsonl"
DEFAULT_CLEAN_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_gemini_clean.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "merge_gemini_literature_rerun_report.json"

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


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def word_count(text: str) -> int:
    return len(str(text or "").split())


def lexical_jaccard(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[A-Za-z']+", str(a or "").lower()))
    tokens_b = set(re.findall(r"[A-Za-z']+", str(b or "").lower()))

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def looks_truncated(text: str) -> bool:
    text = str(text or "").strip()

    if not text:
        return True

    if text[-1] not in VALID_ENDINGS:
        return True

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


def is_valid_rewrite_sample(
    sample: Dict,
    min_length_ratio: float,
    max_jaccard: float,
    require_quality_pass: bool,
    reject_truncated: bool,
) -> bool:
    task_id = sample.get("task_id")
    text = sample.get("text", "")
    source_text = sample.get("source_text", "")

    if not task_id:
        return False

    if not isinstance(text, str) or not text.strip():
        return False

    if sample.get("label") != 1:
        return False

    if sample.get("generator") != "gemini":
        return False

    if require_quality_pass:
        quality = sample.get("quality", {})
        if not isinstance(quality, dict):
            return False
        if quality.get("passed_basic_quality_check") is not True:
            return False

    source_words = word_count(source_text)
    rewrite_words = word_count(text)
    length_ratio = rewrite_words / max(source_words, 1)

    if length_ratio < min_length_ratio:
        return False

    jaccard = lexical_jaccard(source_text, text)
    if jaccard > max_jaccard:
        return False

    finish_reason = sample.get("finish_reason")
    if finish_reason == "length":
        return False

    if reject_truncated and looks_truncated(text):
        return False

    return True


def build_task_map(samples: List[Dict], name: str) -> Dict[str, Dict]:
    task_map = {}
    duplicate_count = 0

    for sample in samples:
        task_id = sample.get("task_id")
        if not task_id:
            continue

        if task_id in task_map:
            duplicate_count += 1

        # Later occurrence wins. This is useful if the same task was retried
        # and appended multiple times.
        task_map[task_id] = sample

    if duplicate_count > 0:
        print(f"[Warning] {name}: found {duplicate_count} duplicate task_id entries. Kept the last one.")

    return task_map


def collect_rerun_task_ids(rerun_prompts: List[Dict]) -> Set[str]:
    task_ids = set()

    for item in rerun_prompts:
        task_id = item.get("task_id")
        if task_id:
            task_ids.add(task_id)

    return task_ids


def summarize(samples: List[Dict]) -> Dict:
    label_counter = Counter(sample.get("label") for sample in samples)
    generator_counter = Counter(sample.get("generator", "unknown") for sample in samples)
    source_counter = Counter(sample.get("source", "unknown") for sample in samples)
    model_counter = Counter(sample.get("model", "unknown") for sample in samples)
    domain_counter = Counter(sample.get("domain", "unknown") for sample in samples)
    prompt_type_counter = Counter(sample.get("prompt_type", "unknown") for sample in samples)

    quality_counter = Counter()
    finish_reason_counter = Counter()

    length_ratios = []
    jaccards = []
    source_word_counts = []
    rewrite_word_counts = []

    possibly_truncated_count = 0

    for sample in samples:
        quality = sample.get("quality", {})
        if isinstance(quality, dict):
            issues = quality.get("quality_issues", [])
            if isinstance(issues, list) and issues:
                for issue in issues:
                    quality_counter[issue] += 1
            else:
                quality_counter["no_issue"] += 1

        finish_reason_counter[str(sample.get("finish_reason", "unknown"))] += 1

        source_text = sample.get("source_text", "")
        text = sample.get("text", "")

        source_wc = word_count(source_text)
        rewrite_wc = word_count(text)
        ratio = rewrite_wc / max(source_wc, 1)
        jaccard = lexical_jaccard(source_text, text)

        source_word_counts.append(source_wc)
        rewrite_word_counts.append(rewrite_wc)
        length_ratios.append(ratio)
        jaccards.append(jaccard)

        if looks_truncated(text):
            possibly_truncated_count += 1

    def stats(values: List[float]) -> Dict:
        if not values:
            return {
                "min": None,
                "max": None,
                "mean": None,
            }

        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "num_samples": len(samples),
        "label_distribution": dict(label_counter),
        "generator_distribution": dict(generator_counter),
        "source_distribution": dict(source_counter),
        "model_distribution": dict(model_counter),
        "domain_distribution": dict(domain_counter),
        "prompt_type_distribution": dict(prompt_type_counter),
        "quality_issue_distribution": dict(quality_counter),
        "finish_reason_distribution": dict(finish_reason_counter),
        "source_word_count": stats(source_word_counts),
        "rewrite_word_count": stats(rewrite_word_counts),
        "length_ratio": stats(length_ratios),
        "lexical_jaccard": stats(jaccards),
        "possibly_truncated_by_merge_check": possibly_truncated_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Gemini literature V1 outputs with rerun outputs to create a clean Gemini dataset."
    )

    parser.add_argument(
        "--v1_output",
        type=str,
        default=str(DEFAULT_V1_OUTPUT_PATH),
        help="Gemini literature V1 output JSONL.",
    )

    parser.add_argument(
        "--rerun_prompts",
        type=str,
        default=str(DEFAULT_RERUN_PROMPTS_PATH),
        help="Rerun prompts JSONL generated by prepare_gemini_literature_rerun_prompts.py.",
    )

    parser.add_argument(
        "--rerun_output",
        type=str,
        default=str(DEFAULT_RERUN_OUTPUT_PATH),
        help="Gemini literature rerun output JSONL.",
    )

    parser.add_argument(
        "--clean_output",
        type=str,
        default=str(DEFAULT_CLEAN_OUTPUT_PATH),
        help="Output clean Gemini literature JSONL.",
    )

    parser.add_argument(
        "--report",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help="Output merge report JSON.",
    )

    parser.add_argument(
        "--min_length_ratio",
        type=float,
        default=0.55,
        help="Minimum rewrite/source length ratio for accepted samples.",
    )

    parser.add_argument(
        "--max_jaccard",
        type=float,
        default=0.82,
        help="Maximum lexical Jaccard for accepted samples.",
    )

    parser.add_argument(
        "--allow_truncated",
        action="store_true",
        help="If set, do not reject samples that look truncated by sentence ending.",
    )

    parser.add_argument(
        "--allow_quality_failed",
        action="store_true",
        help="If set, do not require quality.passed_basic_quality_check=True.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    v1_output_path = Path(args.v1_output)
    rerun_prompts_path = Path(args.rerun_prompts)
    rerun_output_path = Path(args.rerun_output)
    clean_output_path = Path(args.clean_output)
    report_path = Path(args.report)

    v1_samples = load_jsonl(v1_output_path)
    rerun_prompts = load_jsonl(rerun_prompts_path)
    rerun_samples = load_jsonl(rerun_output_path)

    rerun_task_ids = collect_rerun_task_ids(rerun_prompts)

    v1_by_task_id = build_task_map(v1_samples, "v1_output")
    rerun_by_task_id = build_task_map(rerun_samples, "rerun_output")

    require_quality_pass = not args.allow_quality_failed
    reject_truncated = not args.allow_truncated

    clean_samples = []
    kept_v1_count = 0
    replaced_by_rerun_count = 0
    dropped_rerun_task_count = 0
    missing_rerun_output_count = 0
    invalid_rerun_output_count = 0
    invalid_v1_output_count = 0

    dropped_task_ids = []
    missing_rerun_task_ids = []
    invalid_rerun_task_ids = []
    invalid_v1_task_ids = []

    # First, handle all V1 samples.
    for task_id, v1_sample in v1_by_task_id.items():
        # If this task was selected for rerun, prefer rerun output.
        if task_id in rerun_task_ids:
            rerun_sample = rerun_by_task_id.get(task_id)

            if rerun_sample is None:
                missing_rerun_output_count += 1
                missing_rerun_task_ids.append(task_id)

                # Fallback: keep V1 only if it passes strict clean validation.
                if is_valid_rewrite_sample(
                    sample=v1_sample,
                    min_length_ratio=args.min_length_ratio,
                    max_jaccard=args.max_jaccard,
                    require_quality_pass=require_quality_pass,
                    reject_truncated=reject_truncated,
                ):
                    clean_samples.append(v1_sample)
                    kept_v1_count += 1
                else:
                    dropped_rerun_task_count += 1
                    dropped_task_ids.append(task_id)
                continue

            if is_valid_rewrite_sample(
                sample=rerun_sample,
                min_length_ratio=args.min_length_ratio,
                max_jaccard=args.max_jaccard,
                require_quality_pass=require_quality_pass,
                reject_truncated=reject_truncated,
            ):
                item = dict(rerun_sample)
                item["merge_source"] = "rerun_replacement"
                item["replaced_v1"] = True
                clean_samples.append(item)
                replaced_by_rerun_count += 1
            else:
                invalid_rerun_output_count += 1
                invalid_rerun_task_ids.append(task_id)

                # Fallback: keep V1 only if it passes strict clean validation.
                if is_valid_rewrite_sample(
                    sample=v1_sample,
                    min_length_ratio=args.min_length_ratio,
                    max_jaccard=args.max_jaccard,
                    require_quality_pass=require_quality_pass,
                    reject_truncated=reject_truncated,
                ):
                    item = dict(v1_sample)
                    item["merge_source"] = "v1_fallback_after_invalid_rerun"
                    clean_samples.append(item)
                    kept_v1_count += 1
                else:
                    dropped_rerun_task_count += 1
                    dropped_task_ids.append(task_id)

        else:
            # This task was not selected for rerun. Keep V1 if it passes clean validation.
            if is_valid_rewrite_sample(
                sample=v1_sample,
                min_length_ratio=args.min_length_ratio,
                max_jaccard=args.max_jaccard,
                require_quality_pass=require_quality_pass,
                reject_truncated=reject_truncated,
            ):
                item = dict(v1_sample)
                item["merge_source"] = "v1_kept"
                clean_samples.append(item)
                kept_v1_count += 1
            else:
                invalid_v1_output_count += 1
                invalid_v1_task_ids.append(task_id)
                dropped_task_ids.append(task_id)

    # Second, add rerun outputs whose task_id was not in V1 at all.
    added_rerun_not_in_v1_count = 0

    for task_id, rerun_sample in rerun_by_task_id.items():
        if task_id in v1_by_task_id:
            continue

        if is_valid_rewrite_sample(
            sample=rerun_sample,
            min_length_ratio=args.min_length_ratio,
            max_jaccard=args.max_jaccard,
            require_quality_pass=require_quality_pass,
            reject_truncated=reject_truncated,
        ):
            item = dict(rerun_sample)
            item["merge_source"] = "rerun_added_not_in_v1"
            clean_samples.append(item)
            added_rerun_not_in_v1_count += 1
        else:
            invalid_rerun_output_count += 1
            invalid_rerun_task_ids.append(task_id)
            dropped_task_ids.append(task_id)

    # Sort by task_id for reproducibility.
    clean_samples.sort(key=lambda x: x.get("task_id", ""))

    save_jsonl(clean_samples, clean_output_path)

    report = {
        "v1_output_path": str(v1_output_path),
        "rerun_prompts_path": str(rerun_prompts_path),
        "rerun_output_path": str(rerun_output_path),
        "clean_output_path": str(clean_output_path),
        "total_v1_samples": len(v1_samples),
        "total_v1_unique_task_ids": len(v1_by_task_id),
        "total_rerun_prompt_task_ids": len(rerun_task_ids),
        "total_rerun_output_samples": len(rerun_samples),
        "total_rerun_unique_task_ids": len(rerun_by_task_id),
        "total_clean_samples": len(clean_samples),
        "kept_v1_count": kept_v1_count,
        "replaced_by_rerun_count": replaced_by_rerun_count,
        "added_rerun_not_in_v1_count": added_rerun_not_in_v1_count,
        "missing_rerun_output_count": missing_rerun_output_count,
        "invalid_rerun_output_count": invalid_rerun_output_count,
        "invalid_v1_output_count": invalid_v1_output_count,
        "dropped_rerun_task_count": dropped_rerun_task_count,
        "dropped_task_ids_count": len(set(dropped_task_ids)),
        "missing_rerun_task_ids_sample": missing_rerun_task_ids[:30],
        "invalid_rerun_task_ids_sample": invalid_rerun_task_ids[:30],
        "invalid_v1_task_ids_sample": invalid_v1_task_ids[:30],
        "dropped_task_ids_sample": sorted(set(dropped_task_ids))[:30],
        "validation_settings": {
            "min_length_ratio": args.min_length_ratio,
            "max_jaccard": args.max_jaccard,
            "require_quality_pass": require_quality_pass,
            "reject_truncated": reject_truncated,
        },
        "clean_summary": summarize(clean_samples),
    }

    save_json(report, report_path)

    print("=" * 70)
    print("Merge Gemini Literature Rerun")
    print("=" * 70)
    print(f"V1 samples: {len(v1_samples)}")
    print(f"V1 unique task_ids: {len(v1_by_task_id)}")
    print(f"Rerun prompt task_ids: {len(rerun_task_ids)}")
    print(f"Rerun output samples: {len(rerun_samples)}")
    print(f"Rerun unique task_ids: {len(rerun_by_task_id)}")
    print("-" * 70)
    print(f"Kept V1 samples: {kept_v1_count}")
    print(f"Replaced by rerun: {replaced_by_rerun_count}")
    print(f"Added rerun not in V1: {added_rerun_not_in_v1_count}")
    print(f"Missing rerun outputs: {missing_rerun_output_count}")
    print(f"Invalid rerun outputs: {invalid_rerun_output_count}")
    print(f"Invalid V1 outputs: {invalid_v1_output_count}")
    print(f"Dropped task_ids: {len(set(dropped_task_ids))}")
    print("-" * 70)
    print(f"Clean samples saved: {len(clean_samples)}")
    print(f"Clean output path: {clean_output_path}")
    print(f"Report path: {report_path}")

    summary = report["clean_summary"]
    print("\nClean summary:")
    print(f"  label_distribution: {summary['label_distribution']}")
    print(f"  generator_distribution: {summary['generator_distribution']}")
    print(f"  domain_distribution: {summary['domain_distribution']}")
    print(f"  prompt_type_distribution: {summary['prompt_type_distribution']}")
    print(f"  finish_reason_distribution: {summary['finish_reason_distribution']}")
    print(f"  quality_issue_distribution: {summary['quality_issue_distribution']}")
    print(f"  possibly_truncated_by_merge_check: {summary['possibly_truncated_by_merge_check']}")


if __name__ == "__main__":
    main()