import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "human_seed.jsonl"


def load_jsonl(path: Path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cannot find file: {INPUT_PATH}")

    samples = load_jsonl(INPUT_PATH)
    texts = [s["text"] for s in samples]

    word_lengths = [len(t.split()) for t in texts]
    char_lengths = [len(t) for t in texts]
    sources = [s.get("source", "unknown") for s in samples]
    domains = [s.get("domain", "unknown") for s in samples]

    print("=" * 60)
    print("Human Seed Dataset Inspection")
    print("=" * 60)

    print("Total samples:", len(samples))
    print("Domain distribution:", Counter(domains))
    print("Top sources:")
    for source, count in Counter(sources).most_common(20):
        print(f"  {source}: {count}")

    print("\nWord length:")
    print("Mean:", round(np.mean(word_lengths), 2))
    print("Median:", round(np.median(word_lengths), 2))
    print("Min:", np.min(word_lengths))
    print("Max:", np.max(word_lengths))

    print("\nCharacter length:")
    print("Mean:", round(np.mean(char_lengths), 2))
    print("Median:", round(np.median(char_lengths), 2))
    print("Min:", np.min(char_lengths))
    print("Max:", np.max(char_lengths))

    print("\nSample examples:")
    for i, sample in enumerate(samples[:10], start=1):
        print("-" * 60)
        print(f"Example {i}")
        print("id:", sample["id"])
        print("source:", sample["source"])
        print("word_count:", len(sample["text"].split()))
        print(sample["text"][:700])


if __name__ == "__main__":
    main()