import json
import random
from pathlib import Path
from collections import defaultdict, Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42

GENERATOR_NAMES = ["chatgpt", "gemini", "deepseek", "doubao"]


def load_jsonl(path: Path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def assign_by_source(tasks):
    random.seed(RANDOM_SEED)

    source_groups = defaultdict(list)
    for task in tasks:
        source_groups[task.get("source", "unknown")].append(task)

    split_tasks = {name: [] for name in GENERATOR_NAMES}

    for source, group in source_groups.items():
        random.shuffle(group)

        for idx, task in enumerate(group):
            generator = GENERATOR_NAMES[idx % len(GENERATOR_NAMES)]
            task["generator"] = generator
            split_tasks[generator].append(task)

    return split_tasks


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_PATH}")

    tasks = load_jsonl(INPUT_PATH)

    print("=" * 60)
    print("Stratified Split Rewrite Prompts by Generator")
    print("=" * 60)
    print(f"Loaded tasks: {len(tasks)}")

    split_tasks = assign_by_source(tasks)

    print("\nActual counts:")
    for name in GENERATOR_NAMES:
        print(f"  {name}: {len(split_tasks[name])}")

    for name, subset in split_tasks.items():
        output_path = OUTPUT_DIR / f"rewrite_prompts_{name}.jsonl"
        save_jsonl(subset, output_path)
        print(f"Saved {name}: {output_path}")

    all_with_generator = []
    for name in GENERATOR_NAMES:
        all_with_generator.extend(split_tasks[name])

    output_all_path = OUTPUT_DIR / "rewrite_prompts_with_generator.jsonl"
    save_jsonl(all_with_generator, output_all_path)
    print(f"\nSaved combined file: {output_all_path}")

    print("\nPrompt type distribution by generator:")
    for name, subset in split_tasks.items():
        counter = Counter(item.get("prompt_type", "unknown") for item in subset)
        print(f"\n{name}:")
        for prompt_type, count in counter.items():
            print(f"  {prompt_type}: {count}")

    print("\nTop sources by generator:")
    for name, subset in split_tasks.items():
        counter = Counter(item.get("source", "unknown") for item in subset)
        print(f"\n{name}:")
        for source, count in counter.most_common(10):
            print(f"  {source}: {count}")


if __name__ == "__main__":
    main()