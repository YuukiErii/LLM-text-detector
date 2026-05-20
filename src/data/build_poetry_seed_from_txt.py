import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "external_human" / "poetry" / "gutenberg_poetry"
DEFAULT_METADATA_PATH = DEFAULT_INPUT_DIR / "metadata.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "poetry_seed.jsonl"


BAD_KEYWORDS = [
    "project gutenberg",
    "ebook",
    "license",
    "copyright",
    "table of contents",
    "contents",
    "index",
    "transcriber's note",
    "british museum",
    "edition",
    "editor",
    "manuscript",
    "line-numbers",
    "line numbers",
    "introduction and notes",
    "poetic career",
    "poetical works",
    "facsimile",
    "end of the project gutenberg",
    "start of the project gutenberg",
]


def load_metadata(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}

    rows = json.loads(path.read_text(encoding="utf-8"))
    metadata = {}

    for item in rows:
        local_path = item.get("local_path", "")
        if local_path:
            metadata[Path(local_path).name] = item

    return metadata


def strip_gutenberg_boilerplate(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    start_match = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    end_match = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", text, re.I | re.S)

    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]

    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def split_blocks(text: str) -> Iterable[str]:
    current = []
    blank_run = 0

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            blank_run += 1
            if blank_run >= 2 and current:
                yield "\n".join(current)
                current = []
            continue

        blank_run = 0
        current.append(line)

    if current:
        yield "\n".join(current)


def normalize_block(block: str) -> str:
    lines = []

    for line in block.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"_([A-Za-z][^_]*)_", r"\1", line)
        line = re.sub(r"\s+_\d+\b", "", line)
        line = re.sub(r"\s+\[\d+\]", "", line)
        line = re.sub(r"\^\d+\b", "", line)
        line = re.sub(r"\s+\d{2,5}$", "", line)
        line = line.strip()
        if re.fullmatch(r"\d+\.?", line):
            continue
        if re.fullmatch(r"[IVXLCDM]+\.?", line):
            continue
        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def is_heading_like(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) > 3:
        return False

    text = " ".join(lines)
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return True

    uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    return len(words) <= 12 and uppercase_words / max(len(words), 1) > 0.6


def is_valid_poetry_block(block: str, min_words: int, max_words: int, min_lines: int, max_lines: int) -> bool:
    lower = block.lower()
    if any(keyword in lower for keyword in BAD_KEYWORDS):
        return False

    lines = [line.strip() for line in block.splitlines() if line.strip()]
    line_count = len(lines)
    wc = word_count(block)

    if wc < min_words or wc > max_words:
        return False
    if line_count < min_lines or line_count > max_lines:
        return False
    if is_heading_like(block):
        return False

    avg_line_len = sum(len(line) for line in lines) / max(line_count, 1)
    if avg_line_len > 95:
        return False

    terminal_punct = tuple(",.;:!?)]}\"'”’—")
    punct_ended_lines = sum(1 for line in lines if line.rstrip().endswith(terminal_punct))
    if punct_ended_lines / max(line_count, 1) < 0.35:
        return False

    alpha_chars = sum(ch.isalpha() for ch in block)
    if alpha_chars < 0.55 * max(len(block), 1):
        return False

    return True


def build_samples(
    input_dir: Path,
    metadata: Dict[str, Dict],
    min_words: int,
    max_words: int,
    min_lines: int,
    max_lines: int,
    max_per_source: int,
    max_samples: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    samples = []
    seen_texts = set()
    source_counts = Counter()

    txt_files = sorted(input_dir.glob("*.txt"))
    candidate_by_source = defaultdict(list)

    for txt_path in txt_files:
        meta = metadata.get(txt_path.name, {})
        raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        clean_text = strip_gutenberg_boilerplate(raw_text)

        for block in split_blocks(clean_text):
            text = normalize_block(block)
            if not is_valid_poetry_block(text, min_words, max_words, min_lines, max_lines):
                continue

            dedupe_key = re.sub(r"\s+", " ", text.lower()).strip()
            if dedupe_key in seen_texts:
                continue
            seen_texts.add(dedupe_key)

            candidate_by_source[txt_path.name].append(
                {
                    "text": text,
                    "metadata": {
                        "gutenberg_id": meta.get("gutenberg_id"),
                        "title": meta.get("title", txt_path.stem),
                        "authors": meta.get("authors", []),
                        "source_file": txt_path.name,
                        "word_count": word_count(text),
                        "line_count": len([line for line in text.splitlines() if line.strip()]),
                    },
                }
            )

    source_names = sorted(candidate_by_source)
    for source_name in source_names:
        rng.shuffle(candidate_by_source[source_name])

    while source_names and (max_samples <= 0 or len(samples) < max_samples):
        progressed = False

        for source_name in list(source_names):
            if max_samples > 0 and len(samples) >= max_samples:
                break
            if source_counts[source_name] >= max_per_source:
                source_names.remove(source_name)
                continue
            if not candidate_by_source[source_name]:
                source_names.remove(source_name)
                continue

            item = candidate_by_source[source_name].pop()
            source_counts[source_name] += 1
            samples.append(item)
            progressed = True

        if not progressed:
            break

    output = []
    for idx, item in enumerate(samples, start=1):
        pair_id = f"pair_poetry_{idx:06d}"
        meta = item["metadata"]
        source = f"gutenberg_poetry:{meta.get('gutenberg_id') or meta.get('source_file')}"

        output.append(
            {
                "id": f"human_poetry_{idx:06d}",
                "text": item["text"],
                "label": 0,
                "domain": "poetry",
                "source": source,
                "pair_id": pair_id,
                "generation": "human",
                "metadata": meta,
            }
        )

    return output


def save_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build stanza-level poetry human seed from Gutenberg text files.")

    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--min_words", type=int, default=24)
    parser.add_argument("--max_words", type=int, default=170)
    parser.add_argument("--min_lines", type=int, default=3)
    parser.add_argument("--max_lines", type=int, default=18)
    parser.add_argument("--max_per_source", type=int, default=80)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Cannot find input directory: {input_dir}")

    metadata = load_metadata(metadata_path)
    samples = build_samples(
        input_dir=input_dir,
        metadata=metadata,
        min_words=args.min_words,
        max_words=args.max_words,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        max_per_source=args.max_per_source,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    save_jsonl(samples, output_path)

    source_counter = Counter(sample["source"] for sample in samples)
    prompt_lengths = [word_count(sample["text"]) for sample in samples]

    print("=" * 70)
    print("Build Poetry Seed")
    print("=" * 70)
    print(f"Input dir: {input_dir}")
    print(f"Saved samples: {len(samples)}")
    print(f"Output path: {output_path}")
    print("Top sources:")
    for source, count in source_counter.most_common(20):
        print(f"  {source}: {count}")
    if prompt_lengths:
        print("Word count:")
        print(f"  min: {min(prompt_lengths)}")
        print(f"  max: {max(prompt_lengths)}")
        print(f"  mean: {sum(prompt_lengths) / len(prompt_lengths):.2f}")

    if samples:
        print("\nExample:")
        print("-" * 70)
        print(samples[0]["text"][:1000])


if __name__ == "__main__":
    main()
