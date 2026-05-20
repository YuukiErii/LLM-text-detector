import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "academic_seed.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_academic.jsonl"

RANDOM_SEED = 42


ACADEMIC_PROMPTS = {
    "academic_paraphrase": (
        "Paraphrase the following academic paragraph in clear, polished research English. "
        "Preserve all technical meaning, claims, terminology, examples, and approximate length. "
        "Do not summarize, omit details, or add new information. "
        "Return only the rewritten paragraph.\n\n"
        "PARAGRAPH:\n{text}"
    ),
    "academic_modernize": (
        "Rewrite the following academic paragraph in fluent contemporary academic English. "
        "Keep the meaning, technical content, argument structure, and approximate length unchanged. "
        "Improve clarity and wording, but do not simplify away important details. "
        "Return only the rewritten paragraph.\n\n"
        "PARAGRAPH:\n{text}"
    ),
    "academic_style_transfer": (
        "Rewrite the following research paragraph as if it were written in a polished NLP conference paper. "
        "Preserve the original claims, technical terminology, examples, and approximate length. "
        "Use different wording and sentence structures. "
        "Do not summarize or explain the task. "
        "Return only the rewritten paragraph.\n\n"
        "PARAGRAPH:\n{text}"
    ),
}


def load_jsonl(path: Path) -> List[Dict]:
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse line {line_id} in {path}: {e}")

    return samples


def save_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def is_valid_human_sample(sample: Dict) -> bool:
    text = sample.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return False

    if sample.get("label") != 0:
        return False

    if sample.get("domain") != "academic":
        return False

    if not sample.get("pair_id"):
        return False

    return True


def word_count(text: str) -> int:
    return len(text.split())


def looks_complete(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    # Academic paragraphs should normally end with a sentence-final marker.
    # This helps filter obvious truncation such as "The system also uses a pretrained Generated News"
    valid_endings = (".", "?", "!", ")", "]", '"', "'")

    if text[-1] not in valid_endings:
        return False

    return True


def build_prompt_task(sample: Dict, prompt_type: str, task_index: int) -> Dict:
    text = sample["text"].strip()
    prompt_template = ACADEMIC_PROMPTS[prompt_type]
    prompt = prompt_template.format(text=text)

    source_id = sample["id"]
    pair_id = sample["pair_id"]

    return {
        "task_id": f"rewrite_academic_{task_index:06d}",
        "source_id": source_id,
        "pair_id": pair_id,
        "text": text,
        "source_text": text,
        "label": 1,
        "domain": "academic",
        "source": sample.get("source", "academic"),
        "generation": "rewrite_prompt",
        "prompt_type": prompt_type,
        "prompt": prompt,
        "metadata": sample.get("metadata", {}),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare academic rewrite prompts from academic_seed.jsonl."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Input academic seed JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output academic rewrite prompts JSONL file.",
    )

    parser.add_argument(
        "--min_words",
        type=int,
        default=60,
        help="Minimum word count for academic paragraphs.",
    )

    parser.add_argument(
        "--max_words",
        type=int,
        default=330,
        help="Maximum word count for academic paragraphs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )

    parser.add_argument(
        "--allow_incomplete_ending",
        action="store_true",
        help="If set, do not filter paragraphs that lack sentence-final punctuation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    rng = random.Random(args.seed)

    samples = load_jsonl(input_path)

    print("=" * 70)
    print("Prepare Academic Rewrite Prompts")
    print("=" * 70)
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Loaded samples: {len(samples)}")

    valid_samples = []
    filtered_reasons = Counter()

    for sample in samples:
        if not is_valid_human_sample(sample):
            filtered_reasons["invalid_basic_fields"] += 1
            continue

        text = sample.get("text", "").strip()
        wc = word_count(text)

        if wc < args.min_words:
            filtered_reasons["too_short"] += 1
            continue

        if wc > args.max_words:
            filtered_reasons["too_long"] += 1
            continue

        if not args.allow_incomplete_ending and not looks_complete(text):
            filtered_reasons["incomplete_ending"] += 1
            continue

        valid_samples.append(sample)

    rng.shuffle(valid_samples)

    prompt_types = list(ACADEMIC_PROMPTS.keys())

    tasks = []
    prompt_type_counter = Counter()

    for i, sample in enumerate(valid_samples, start=1):
        prompt_type = prompt_types[(i - 1) % len(prompt_types)]
        prompt_type_counter[prompt_type] += 1

        task = build_prompt_task(
            sample=sample,
            prompt_type=prompt_type,
            task_index=i,
        )
        tasks.append(task)

    save_jsonl(tasks, output_path)

    print("-" * 70)
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Saved rewrite tasks: {len(tasks)}")
    print(f"Output path: {output_path}")

    print("\nFiltered reasons:")
    if filtered_reasons:
        for reason, count in filtered_reasons.most_common():
            print(f"  {reason}: {count}")
    else:
        print("  none")

    print("\nPrompt type distribution:")
    for prompt_type, count in prompt_type_counter.items():
        print(f"  {prompt_type}: {count}")

    if tasks:
        print("\nExample prompt:")
        print("-" * 70)
        print(tasks[0]["prompt"][:1500])


if __name__ == "__main__":
    main()