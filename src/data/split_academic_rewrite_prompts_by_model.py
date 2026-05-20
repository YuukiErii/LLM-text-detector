import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_academic.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42


MODEL_RATIOS = {
    "chatgpt": 0.40,
    "deepseek": 0.40,
    "gemini": 0.10,
    "doubao": 0.10,
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


def assign_models(samples: List[Dict], seed: int) -> Dict[str, List[Dict]]:
    """
    Assign academic rewrite prompts to models using fixed ratios.

    We shuffle first, then assign exact counts based on ratios.
    """
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)

    total = len(samples)

    n_chatgpt = round(total * MODEL_RATIOS["chatgpt"])
    n_deepseek = round(total * MODEL_RATIOS["deepseek"])
    n_gemini = round(total * MODEL_RATIOS["gemini"])

    # Put the remainder into doubao to guarantee exact total.
    n_doubao = total - n_chatgpt - n_deepseek - n_gemini

    counts = {
        "chatgpt": n_chatgpt,
        "deepseek": n_deepseek,
        "gemini": n_gemini,
        "doubao": n_doubao,
    }

    assigned = {
        "chatgpt": [],
        "deepseek": [],
        "gemini": [],
        "doubao": [],
    }

    start = 0
    for model_name in ["chatgpt", "deepseek", "gemini", "doubao"]:
        end = start + counts[model_name]

        for sample in samples[start:end]:
            item = dict(sample)
            item["assigned_generator"] = model_name
            assigned[model_name].append(item)

        start = end

    return assigned


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split academic rewrite prompts by model."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Input rewrite_prompts_academic.jsonl path.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    samples = load_jsonl(input_path)

    print("=" * 70)
    print("Split Academic Rewrite Prompts by Model")
    print("=" * 70)
    print(f"Input path: {input_path}")
    print(f"Loaded samples: {len(samples)}")
    print("Ratios:", MODEL_RATIOS)

    assigned = assign_models(samples, seed=args.seed)

    output_paths = {
        "chatgpt": output_dir / "rewrite_prompts_academic_chatgpt.jsonl",
        "deepseek": output_dir / "rewrite_prompts_academic_deepseek.jsonl",
        "gemini": output_dir / "rewrite_prompts_academic_gemini.jsonl",
        "doubao": output_dir / "rewrite_prompts_academic_doubao.jsonl",
    }

    for model_name, model_samples in assigned.items():
        save_jsonl(model_samples, output_paths[model_name])

    all_with_generator = []
    for model_name, model_samples in assigned.items():
        all_with_generator.extend(model_samples)

    combined_path = output_dir / "rewrite_prompts_academic_with_generator.jsonl"
    save_jsonl(all_with_generator, combined_path)

    print("\nSaved files:")
    for model_name, path in output_paths.items():
        print(f"  {model_name}: {path} ({len(assigned[model_name])})")

    print(f"  combined: {combined_path} ({len(all_with_generator)})")

    print("\nDistribution:")
    counter = Counter()
    prompt_type_counter = Counter()

    for model_name, model_samples in assigned.items():
        counter[model_name] = len(model_samples)
        for sample in model_samples:
            prompt_type_counter[(model_name, sample.get("prompt_type", "unknown"))] += 1

    print("Model distribution:")
    for model_name, count in counter.items():
        print(f"  {model_name}: {count}")

    print("\nPrompt type distribution by model:")
    for key, count in sorted(prompt_type_counter.items()):
        model_name, prompt_type = key
        print(f"  {model_name} / {prompt_type}: {count}")


if __name__ == "__main__":
    main()