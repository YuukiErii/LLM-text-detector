import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "human_seed.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts.jsonl"


PROMPTS = [
    {
        "prompt_type": "literary_modernize",
        "template": (
            "Rewrite the following literary passage in fluent contemporary English. "
            "Preserve the meaning, scene, narrative perspective, and approximate length. "
            "Do not summarize. Use different wording and sentence structures.\n\n"
            "PASSAGE:\n{ text }"
        ),
    },
    {
        "prompt_type": "literary_paraphrase",
        "template": (
            "Paraphrase the following literary passage. Preserve the plot, imagery, "
            "characters, and emotional tone, but replace the original wording with "
            "natural polished prose. Keep approximately the same length.\n\n"
            "PASSAGE:\n{ text }"
        ),
    },
    {
        "prompt_type": "style_imitation",
        "template": (
            "Rewrite the following passage as a polished literary imitation. Keep the "
            "same events and mood, but use different names, wording, and sentence "
            "structures. Keep approximately the same length.\n\n"
            "PASSAGE:\n{ text }"
        ),
    },
]


def load_jsonl(path: Path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_PATH}")

    samples = load_jsonl(INPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tasks = []

    for i, sample in enumerate(samples):
        prompt_config = PROMPTS[i % len(PROMPTS)]

        text = sample["text"]
        prompt = prompt_config["template"].replace("{ text }", text)

        task = {
            "task_id": f"rewrite_{i + 1:06d}",
            "source_id": sample["id"],
            "pair_id": sample["pair_id"],
            "domain": sample["domain"],
            "source": sample["source"],
            "source_text": text,
            "prompt_type": prompt_config["prompt_type"],
            "prompt": prompt,
        }

        tasks.append(task)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("Rewrite Prompt Preparation")
    print("=" * 60)
    print(f"Loaded human samples: {len(samples)}")
    print(f"Saved rewrite tasks: {len(tasks)}")
    print(f"Output path: {OUTPUT_PATH}")

    print("\nExample prompt:")
    print("-" * 60)
    print(tasks[0]["prompt"][:1200])


if __name__ == "__main__":
    main()