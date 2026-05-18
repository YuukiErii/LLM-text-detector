import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_deepseek.jsonl"


def resolve_path(path_str: str) -> Path:
    """
    Resolve input path.

    Supports both:
    1. absolute path
    2. project-relative path, e.g. data/processed/llm_rewrite_doubao.jsonl
    """
    path = Path(path_str)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_jsonl(path: Path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse line {line_id}: {e}")

    return samples


def safe_mean(values):
    return round(float(np.mean(values)), 4) if values else 0.0


def safe_median(values):
    return round(float(np.median(values)), 4) if values else 0.0


def safe_min(values):
    return min(values) if values else 0.0


def safe_max(values):
    return max(values) if values else 0.0


def print_numeric_stats(title: str, values):
    print(f"\n{title}:")
    print("Mean:", safe_mean(values))
    print("Median:", safe_median(values))
    print("Min:", safe_min(values))
    print("Max:", safe_max(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect generated LLM rewrite JSONL files."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=(
            "Path to LLM rewrite JSONL file. "
            "Can be absolute or relative to project root. "
            "Default: data/processed/llm_rewrite_deepseek.jsonl"
        ),
    )

    parser.add_argument(
        "--show_examples",
        type=int,
        default=5,
        help="Number of normal examples to print.",
    )

    parser.add_argument(
        "--show_failed",
        type=int,
        default=10,
        help="Number of failed examples to print.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = resolve_path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find file: {input_path}")

    samples = load_jsonl(input_path)

    print("=" * 70)
    print("LLM Rewrite Inspection")
    print("=" * 70)
    print(f"Input file: {input_path}")
    print("Total samples:", len(samples))

    if not samples:
        print("No samples found.")
        return

    print("\nBasic field distribution:")
    print("Label distribution:", Counter(s.get("label", "unknown") for s in samples))
    print("Generator distribution:", Counter(s.get("generator", "unknown") for s in samples))
    print("Source distribution:", Counter(s.get("source", "unknown") for s in samples))
    print("Model distribution:", Counter(s.get("model", "unknown") for s in samples))
    print("Prompt type distribution:", Counter(s.get("prompt_type", "unknown") for s in samples))
    print("Domain distribution:", Counter(s.get("domain", "unknown") for s in samples))

    qualities = [s.get("quality", {}) for s in samples]

    passed = [
        q.get("passed_basic_quality_check", False)
        for q in qualities
    ]

    print("\nQuality pass rate:")
    print(f"Passed: {sum(passed)} / {len(passed)}")
    print(f"Pass rate: {sum(passed) / max(len(passed), 1):.4f}")

    source_word_counts = [
        q.get("source_word_count", 0)
        for q in qualities
    ]

    rewrite_word_counts = [
        q.get("rewrite_word_count", 0)
        for q in qualities
    ]

    length_ratios = [
        q.get("length_ratio", 0)
        for q in qualities
    ]

    jaccards = [
        q.get("lexical_jaccard", 0)
        for q in qualities
    ]

    print_numeric_stats("Source word count", source_word_counts)
    print_numeric_stats("Rewrite word count", rewrite_word_counts)
    print_numeric_stats("Length ratio", length_ratios)
    print_numeric_stats("Lexical Jaccard similarity", jaccards)

    issue_counter = Counter()

    for q in qualities:
        issues = q.get("quality_issues", [])

        if not issues:
            issue_counter["no_issue"] += 1
        else:
            for issue in issues:
                issue_counter[issue] += 1

    print("\nQuality issues:")
    for issue, count in issue_counter.most_common():
        print(f"  {issue}: {count}")

    failed_samples = [
        s for s in samples
        if not s.get("quality", {}).get("passed_basic_quality_check", False)
    ]

    print("\nFailed examples:")
    if not failed_samples:
        print("No failed examples.")
    else:
        for i, sample in enumerate(failed_samples[:args.show_failed], start=1):
            print("-" * 70)
            print(f"Failed example {i}")
            print("task_id:", sample.get("task_id"))
            print("generator:", sample.get("generator"))
            print("model:", sample.get("model"))
            print("prompt_type:", sample.get("prompt_type"))
            print("issues:", sample.get("quality", {}).get("quality_issues"))
            print("quality:", sample.get("quality"))
            print("\nsource_text:")
            print(sample.get("source_text", "")[:800])
            print("\nrewrite_text:")
            print(sample.get("text", "")[:800])

    print("\nNormal examples:")
    for i, sample in enumerate(samples[:args.show_examples], start=1):
        print("-" * 70)
        print(f"Example {i}")
        print("task_id:", sample.get("task_id"))
        print("generator:", sample.get("generator"))
        print("model:", sample.get("model"))
        print("prompt_type:", sample.get("prompt_type"))
        print("quality:", sample.get("quality"))

        print("\nsource:")
        print(sample.get("source_text", "")[:500])

        print("\nrewrite:")
        print(sample.get("text", "")[:500])


if __name__ == "__main__":
    main()