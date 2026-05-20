import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HUMAN_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "human_seed.jsonl",
]

DEFAULT_LLM_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_deepseek.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_chatgpt.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_gemini_clean.jsonl",
    PROJECT_ROOT / "data" / "processed" / "llm_rewrite_doubao.jsonl",
]

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "full_dataset.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "full_dataset_report.json"

VALID_ENDINGS = (".", "?", "!", ")", "]", '"', "'", "”", "’")


def load_jsonl(path: Path) -> List[Dict]:
    samples = []

    if not path.exists():
        print(f"[Warning] File does not exist, skipped: {path}")
        return samples

    with path.open("r", encoding="utf-8") as f:
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

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def looks_truncated(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return True

    if text[-1] not in VALID_ENDINGS:
        return True

    lower = text.lower()
    dangling_patterns = [
        r"\b(and|or|but|because|while|although|though|that|which|who|whom|whose|where|when|if|to|of|in|on|at|by|with|from|into|for|as|than|the|a|an)\s*$",
        r"[,;:]\s*$",
        r"\b(can|could|would|should|may|might|must|shall|will|is|are|was|were|be|been|being|have|has|had|do|does|did)\s*$",
    ]

    return any(re.search(pattern, lower) for pattern in dangling_patterns)


def normalize_human_sample(sample: Dict) -> Dict:
    return {
        "id": sample["id"],
        "text": sample["text"],
        "label": 0,
        "domain": sample.get("domain", "unknown"),
        "source": sample.get("source", "unknown"),
        "generator": "human",
        "model": "human",
        "pair_id": sample["pair_id"],
        "source_id": sample.get("id"),
        "prompt_type": "human",
        "generation": "human",
        "metadata": sample.get("metadata", {}),
    }


def validate_human_sample(sample: Dict) -> Tuple[bool, List[str]]:
    reasons = []

    if sample.get("label") != 0:
        reasons.append("bad_label")
    if not sample.get("id"):
        reasons.append("missing_id")
    if not sample.get("pair_id"):
        reasons.append("missing_pair_id")

    text = sample.get("text", "")
    if not isinstance(text, str) or not text.strip():
        reasons.append("empty_text")

    return len(reasons) == 0, reasons


def normalize_llm_sample(sample: Dict) -> Dict:
    return {
        "id": sample["id"],
        "text": sample["text"],
        "label": 1,
        "domain": sample.get("domain", "unknown"),
        "source": sample.get("source", sample.get("generator", "unknown")),
        "generator": sample.get("generator", sample.get("source", "unknown")),
        "model": sample.get("model", "unknown"),
        "pair_id": sample["pair_id"],
        "source_id": sample.get("source_id"),
        "prompt_type": sample.get("prompt_type", "unknown"),
        "generation": sample.get("generation", "llm_rewrite"),
        "quality": sample.get("quality", {}),
        "finish_reason": sample.get("finish_reason"),
        "metadata": sample.get("metadata", {}),
    }


def validate_llm_sample(sample: Dict, reject_truncated: bool) -> Tuple[bool, List[str]]:
    reasons = []
    domain = str(sample.get("domain", "")).lower()

    if sample.get("label") != 1:
        reasons.append("bad_label")
    if not sample.get("id"):
        reasons.append("missing_id")
    if not sample.get("pair_id"):
        reasons.append("missing_pair_id")

    text = sample.get("text", "")
    if not isinstance(text, str) or not text.strip():
        reasons.append("empty_text")

    quality = sample.get("quality", {})
    if isinstance(quality, dict):
        if quality.get("passed_basic_quality_check") is False:
            reasons.append("quality_failed")

        issues = quality.get("quality_issues", [])
        if reject_truncated and domain != "poetry" and isinstance(issues, list) and "possibly_truncated" in issues:
            reasons.append("possibly_truncated_quality_issue")

    if sample.get("finish_reason") == "length":
        reasons.append("finish_reason_length")

    if reject_truncated and domain != "poetry" and looks_truncated(text):
        reasons.append("possibly_truncated_text")

    return len(reasons) == 0, reasons


def build_report(samples: List[Dict], filtered: Counter) -> Dict:
    label_counter = Counter(str(s.get("label")) for s in samples)
    generator_counter = Counter(s.get("generator", "unknown") for s in samples)
    source_counter = Counter(s.get("source", "unknown") for s in samples)
    prompt_type_counter = Counter(s.get("prompt_type", "unknown") for s in samples)
    domain_counter = Counter(s.get("domain", "unknown") for s in samples)
    domain_label_counter = Counter(f"{s.get('domain', 'unknown')}|{s.get('label')}" for s in samples)
    domain_generator_counter = Counter(f"{s.get('domain', 'unknown')}|{s.get('generator', 'unknown')}" for s in samples)

    pair_to_labels = defaultdict(list)
    pair_to_domains = defaultdict(set)
    for sample in samples:
        pair_id = sample.get("pair_id")
        pair_to_labels[pair_id].append(sample.get("label"))
        pair_to_domains[pair_id].add(sample.get("domain", "unknown"))

    text_lengths = [len(s.get("text", "").split()) for s in samples]

    return {
        "total_samples": len(samples),
        "label_distribution": dict(label_counter),
        "domain_distribution": dict(domain_counter),
        "generator_distribution": dict(generator_counter),
        "source_distribution_top50": dict(source_counter.most_common(50)),
        "prompt_type_distribution": dict(prompt_type_counter),
        "domain_label_distribution": dict(domain_label_counter),
        "domain_generator_distribution": dict(domain_generator_counter),
        "pair_stats": {
            "total_unique_pair_ids": len(pair_to_labels),
            "pairs_with_human": sum(1 for labels in pair_to_labels.values() if 0 in labels),
            "pairs_with_llm": sum(1 for labels in pair_to_labels.values() if 1 in labels),
            "pairs_with_both": sum(1 for labels in pair_to_labels.values() if 0 in labels and 1 in labels),
            "pairs_with_multiple_domains": sum(1 for domains in pair_to_domains.values() if len(domains) > 1),
        },
        "word_length": {
            "min": min(text_lengths) if text_lengths else None,
            "max": max(text_lengths) if text_lengths else None,
            "mean": sum(text_lengths) / len(text_lengths) if text_lengths else None,
        },
        "filtered_samples": dict(filtered),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build full dataset from human and LLM rewrite JSONL files.")

    parser.add_argument(
        "--human",
        type=str,
        nargs="+",
        default=[str(path) for path in DEFAULT_HUMAN_PATHS],
        help="Path(s) to human seed JSONL files.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        nargs="*",
        default=[str(path) for path in DEFAULT_LLM_PATHS],
        help="Paths to LLM rewrite JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for full dataset JSONL.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help="Output path for dataset report JSON.",
    )
    parser.add_argument(
        "--allow_truncated",
        action="store_true",
        help="If set, do not filter outputs with suspicious incomplete endings.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    human_paths = [Path(path) for path in args.human]
    llm_paths = [Path(path) for path in args.llm]
    output_path = Path(args.output)
    report_path = Path(args.report)
    reject_truncated = not args.allow_truncated

    print("=" * 70)
    print("Build Full Dataset")
    print("=" * 70)
    print("Human files:")
    for path in human_paths:
        print(f"  {path}")
    print("LLM files:")
    for path in llm_paths:
        print(f"  {path}")
    print(f"Reject truncated: {reject_truncated}")

    filtered = Counter()
    all_samples = []
    seen_ids = set()

    for path in human_paths:
        raw_samples = load_jsonl(path)
        kept = 0

        for sample in raw_samples:
            valid, reasons = validate_human_sample(sample)
            if not valid:
                for reason in reasons:
                    filtered[f"human:{path.name}:{reason}"] += 1
                continue

            item = normalize_human_sample(sample)
            if item["id"] in seen_ids:
                filtered[f"human:{path.name}:duplicate_id"] += 1
                continue

            seen_ids.add(item["id"])
            all_samples.append(item)
            kept += 1

        print(f"Loaded human file: {path} raw={len(raw_samples)} kept={kept}")

    for path in llm_paths:
        raw_samples = load_jsonl(path)
        kept = 0

        for sample in raw_samples:
            valid, reasons = validate_llm_sample(sample, reject_truncated=reject_truncated)
            if not valid:
                for reason in reasons:
                    filtered[f"llm:{path.name}:{reason}"] += 1
                continue

            item = normalize_llm_sample(sample)
            if item["id"] in seen_ids:
                filtered[f"llm:{path.name}:duplicate_id"] += 1
                continue

            seen_ids.add(item["id"])
            all_samples.append(item)
            kept += 1

        print(f"Loaded LLM file: {path} raw={len(raw_samples)} kept={kept}")

    save_jsonl(all_samples, output_path)

    report = build_report(all_samples, filtered)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 70)
    print(f"Saved full dataset: {output_path}")
    print(f"Saved report: {report_path}")
    print(f"Total samples: {report['total_samples']}")
    print(f"Label distribution: {report['label_distribution']}")
    print(f"Domain distribution: {report['domain_distribution']}")
    print(f"Generator distribution: {report['generator_distribution']}")
    print(f"Pair stats: {report['pair_stats']}")
    print(f"Filtered samples: {report['filtered_samples']}")


if __name__ == "__main__":
    main()
