import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from huggingface_hub import snapshot_download
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "academic_seed.jsonl"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "external_human" / "academic" / "acl_ocl"

KEYWORDS = [
    "parsing",
    "parser",
    "syntax",
    "syntactic",
    "semantic",
    "semantics",
    "machine translation",
    "translation",
    "alignment",
    "coreference",
    "pronoun",
    "discourse",
    "grammar",
    "language model",
    "natural language generation",
    "generation",
    "summarization",
    "evaluation",
    "ambiguity",
    "corpus",
    "annotation",
    "dependency",
    "constituency",
]


BAD_SECTIONS = [
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "related work",  # Can be kept or filtered; filter part of the template-like content here.
]


def open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_json_records(path: Path) -> Iterable[Dict]:
    """
    Read JSON / JSONL / gzipped JSONL robustly.

    Supports:
    1. single JSON dict file
    2. JSON array file
    3. JSONL file
    """
    with open_text_file(path) as f:
        content = f.read().strip()

    if not content:
        return

    # Try whole-file JSON first.
    try:
        data = json.loads(content)

        if isinstance(data, dict):
            yield data
            return

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return

    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
            if isinstance(item, dict):
                yield item
        except json.JSONDecodeError:
            continue

def find_data_files(root: Path) -> List[Path]:
    patterns = [
        "*.jsonl",
        "*.json",
        "*.jsonl.gz",
        "*.json.gz",
    ]

    files = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))

    # Remove README / metadata-like files.
    files = [
        p for p in files
        if "README" not in p.name.upper()
        and "dataset_infos" not in p.name
    ]

    return sorted(set(files))


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())


def has_keyword(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in KEYWORDS)


def is_bad_section(section: Optional[str]) -> bool:
    if not section:
        return False

    lower = section.lower()
    return any(bad in lower for bad in BAD_SECTIONS)


def is_valid_academic_paragraph(text: str, section: Optional[str]) -> bool:
    text = clean_text(text)

    if is_bad_section(section):
        return False

    wc = word_count(text)

    if wc < 80:
        return False

    if wc > 320:
        return False

    # Filter paragraphs that look strongly like formulas, tables, or references.
    lower = text.lower()

    bad_patterns = [
        "http://",
        "https://",
        "www.",
        "copyright",
        "isbn",
        "proceedings of",
        "all rights reserved",
        "table ",
        "figure ",
        "fig. ",
        "et al.",
    ]

    # "et al." is common in papers, so do not over-filter it.
    # Only clear reference-like structures are filtered below.
    bad_patterns = [
        "http://",
        "https://",
        "www.",
        "copyright",
        "isbn",
        "all rights reserved",
    ]

    if any(p in lower for p in bad_patterns):
        return False

    # Too many digits or symbols usually indicate tables or formulas.
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.12:
        return False

    # Prefer NLP/CL keywords so the training seed is closer to academic teacher-test prose.
    if not has_keyword(text):
        return False

    return True


def extract_body_paragraphs(record: Dict) -> Iterable[Dict]:
    """
    ACL-OCL record structure observed:
    record["pdf_parse"]["body_text"] = [
        {"text": "...", "section": "...", ...},
        ...
    ]
    """
    pdf_parse = record.get("pdf_parse", {})

    if not isinstance(pdf_parse, dict):
        return

    body_text = pdf_parse.get("body_text", [])

    if not isinstance(body_text, list):
        return

    for paragraph in body_text:
        if not isinstance(paragraph, dict):
            continue

        text = paragraph.get("text", "")
        section = paragraph.get("section", "")

        if not isinstance(text, str):
            continue

        text = clean_text(text)

        if is_valid_academic_paragraph(text, section):
            yield {
                "text": text,
                "section": section,
            }


def build_sample(record: Dict, paragraph: Dict, sample_id: int) -> Dict:
    paper_id = str(record.get("paper_id", "unknown"))
    title = record.get("title", "")
    year = record.get("year", "")
    venue = record.get("venue", "")

    pair_id = f"pair_academic_{sample_id:06d}"

    return {
        "id": f"human_academic_{sample_id:06d}",
        "text": paragraph["text"],
        "label": 0,
        "domain": "academic",
        "source": f"acl_ocl:{paper_id}",
        "pair_id": pair_id,
        "generation": "human",
        "metadata": {
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "venue": venue,
            "section": paragraph.get("section", ""),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build academic human seed dataset from ACL-OCL without datasets schema casting."
    )

    parser.add_argument(
        "--target_count",
        type=int,
        default=1200,
        help="Number of academic paragraphs to extract.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Local directory to store downloaded ACL-OCL files.",
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=-1,
        help="Maximum number of data files to scan. Use -1 for all.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)

    print("=" * 70)
    print("Build Academic Seed from ACL-OCL")
    print("=" * 70)
    print("Downloading / locating ACL-OCL dataset files...")

    snapshot_dir = snapshot_download(
        repo_id="WINGNUS/ACL-OCL",
        repo_type="dataset",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "Base_JSON/prefixA/json/acl/*.json",
            "Base_JSON/prefixC/json/coling/*.json",
            "Base_JSON/prefixE/json/emnlp/*.json",
            "Base_JSON/prefixF/json/findings/*.json",
        ],
    )

    snapshot_dir = Path(snapshot_dir)
    print(f"Snapshot dir: {snapshot_dir}")

    data_files = find_data_files(snapshot_dir)

    if args.max_files > 0:
        data_files = data_files[: args.max_files]

    print(f"Found data files: {len(data_files)}")

    if not data_files:
        raise FileNotFoundError(f"No JSON/JSONL data files found under {snapshot_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_id = 0
    seen_texts = set()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for data_file in data_files:
            print(f"\nScanning file: {data_file}")

            for record in tqdm(iter_json_records(data_file), desc=data_file.name):
                for paragraph in extract_body_paragraphs(record):
                    text = paragraph["text"]

                    # Simple deduplication.
                    key = text[:300].lower()
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)

                    sample_id += 1
                    sample = build_sample(record, paragraph, sample_id)

                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                    if sample_id >= args.target_count:
                        print("\nReached target count.")
                        print(f"Saved {sample_id} samples to: {output_path}")
                        return

    print("\nFinished scanning all files.")
    print(f"Saved {sample_id} samples to: {output_path}")

    if sample_id < args.target_count:
        print(
            f"[Warning] Only extracted {sample_id} samples, "
            f"less than target_count={args.target_count}. "
            f"Consider relaxing filters or increasing max_files."
        )


if __name__ == "__main__":
    main()
