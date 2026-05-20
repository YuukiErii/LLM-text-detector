import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "poetry_seed.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_poetry.jsonl"


PROMPTS = [
    {
        "prompt_type": "poetry_modernize",
        "template": (
            "Rewrite the following poem or stanza in contemporary poetic English. "
            "Preserve the speaker, imagery, emotional tone, and approximate number of lines. "
            "Use different wording and phrasing. Do not explain or summarize.\n\n"
            "POEM:\n{ text }"
        ),
    },
    {
        "prompt_type": "poetry_paraphrase",
        "template": (
            "Paraphrase the following poem or stanza as a poem, not prose. Preserve the "
            "meaning, images, mood, and approximate length, while changing the wording "
            "and sentence structure. Return only the rewritten poem.\n\n"
            "POEM:\n{ text }"
        ),
    },
    {
        "prompt_type": "poetry_style_transfer",
        "template": (
            "Create a polished poetic rewrite of the following stanza. Keep the same "
            "core scene and feeling, but use fresh diction, different phrasing, and a "
            "natural contemporary poetic style. Keep a similar number of lines. "
            "Return only the rewritten poem.\n\n"
            "POEM:\n{ text }"
        ),
    },
]


def load_jsonl(path: Path) -> List[Dict]:
    samples = []

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


def build_prompt_task(sample: Dict, prompt_config: Dict, task_index: int) -> Dict:
    text = sample["text"]
    prompt = prompt_config["template"].replace("{ text }", text)

    return {
        "task_id": f"rewrite_poetry_{task_index:06d}",
        "source_id": sample["id"],
        "pair_id": sample["pair_id"],
        "domain": "poetry",
        "source": sample.get("source", "unknown"),
        "source_text": text,
        "prompt_type": prompt_config["prompt_type"],
        "prompt": prompt,
        "metadata": sample.get("metadata", {}),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LLM rewrite prompts for poetry samples.")

    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    samples = load_jsonl(input_path)

    tasks = []
    for idx, sample in enumerate(samples, start=1):
        prompt_config = PROMPTS[(idx - 1) % len(PROMPTS)]
        tasks.append(build_prompt_task(sample, prompt_config, idx))

    save_jsonl(tasks, output_path)

    print("=" * 70)
    print("Prepare Poetry Rewrite Prompts")
    print("=" * 70)
    print(f"Loaded poetry samples: {len(samples)}")
    print(f"Saved rewrite tasks: {len(tasks)}")
    print(f"Output path: {output_path}")
    print("Prompt type distribution:", dict(Counter(task["prompt_type"] for task in tasks)))

    if tasks:
        print("\nExample prompt:")
        print("-" * 70)
        print(tasks[0]["prompt"][:1200])


if __name__ == "__main__":
    main()
