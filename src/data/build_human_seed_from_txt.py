import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "external_human" / "gutenberg"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "human_seed.jsonl"


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def split_paragraphs(text: str):
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    return [p for p in paragraphs if p]


def is_valid_paragraph(paragraph: str) -> bool:
    words = paragraph.split()
    word_count = len(words)

    if word_count < 50:
        return False
    if word_count > 400:
        return False

    # Remove obvious table-of-contents, copyright, and project-description lines.
    lower = paragraph.lower()
    bad_keywords = [
        "project gutenberg",
        "ebook",
        "license",
        "copyright",
        "contents",
        "chapter",
        "table of contents",
        "illustration",
    ]

    if any(keyword in lower for keyword in bad_keywords):
        return False

    return True


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    txt_files = list(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {INPUT_DIR}")

    samples = []
    sample_id = 0

    for txt_path in txt_files:
        print(f"Reading {txt_path.name}")

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        text = clean_text(text)
        paragraphs = split_paragraphs(text)

        valid_paragraphs = [p for p in paragraphs if is_valid_paragraph(p)]

        print(f"  paragraphs: {len(paragraphs)}")
        print(f"  valid: {len(valid_paragraphs)}")

        source_name = txt_path.stem

        for p in valid_paragraphs:
            sample_id += 1
            pair_id = f"pair_lit_{sample_id:06d}"

            sample = {
                "id": f"human_lit_{sample_id:06d}",
                "text": p,
                "label": 0,
                "domain": "literature",
                "source": f"gutenberg:{source_name}",
                "pair_id": pair_id,
                "generation": "human",
            }
            samples.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"Saved {len(samples)} human samples to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
