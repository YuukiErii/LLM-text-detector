import argparse
import gzip
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from build_human_seed_from_txt import clean_text as clean_literature_text
from build_human_seed_from_txt import split_paragraphs
from build_poetry_seed_from_txt import build_samples as build_poetry_samples
from build_poetry_seed_from_txt import load_metadata as load_poetry_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXISTING_HUMAN = [PROJECT_ROOT / "data" / "processed" / "human_seed_combined.jsonl"]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "human_hard_negative_seed.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "human_hard_negative_seed_report.json"

DEFAULT_POETRY_DIR = PROJECT_ROOT / "data" / "raw" / "external_human" / "poetry" / "gutenberg_poetry"
DEFAULT_POETRY_METADATA = DEFAULT_POETRY_DIR / "metadata.json"
DEFAULT_LITERATURE_DIR = PROJECT_ROOT / "data" / "raw" / "external_human" / "gutenberg"
DEFAULT_ACADEMIC_DIR = PROJECT_ROOT / "data" / "raw" / "external_human" / "academic" / "acl_ocl"

ACADEMIC_KEYWORDS = [
    "algorithm",
    "alignment",
    "annotation",
    "approach",
    "classifier",
    "corpus",
    "dependency",
    "discourse",
    "evaluation",
    "generation",
    "grammar",
    "language model",
    "machine translation",
    "model",
    "natural language",
    "parsing",
    "semantic",
    "syntax",
    "translation",
]

BAD_SECTIONS = [
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
]

BAD_TEXT_PATTERNS = [
    "project gutenberg",
    "http://",
    "https://",
    "www.",
    "copyright",
    "all rights reserved",
    "isbn",
]


def normalize_key(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def load_jsonl(path: Path) -> List[Dict]:
    samples = []
    if not path.exists():
        return samples

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                samples.append(item)

    return samples


def load_existing_text_keys(paths: List[Path]) -> Set[str]:
    keys = set()
    for path in paths:
        for sample in load_jsonl(path):
            text = sample.get("text", "")
            if isinstance(text, str) and text.strip():
                keys.add(normalize_key(text))
    return keys


def open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def iter_json_records(path: Path) -> Iterable[Dict]:
    with open_text_file(path) as f:
        content = f.read().strip()

    if not content:
        return

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

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            yield item


def next_id(prefix: str, index: int) -> Dict[str, str]:
    return {
        "id": f"human_hardneg_{index:06d}",
        "pair_id": f"pair_hardneg_{index:06d}",
        "source_id": f"{prefix}_{index:06d}",
    }


def make_hardneg_sample(text: str, domain: str, source: str, metadata: Dict, index: int) -> Dict:
    ids = next_id(domain, index)
    item = {
        "id": ids["id"],
        "text": text,
        "label": 0,
        "domain": domain,
        "source": source,
        "pair_id": ids["pair_id"],
        "generation": "human",
        "metadata": dict(metadata),
    }
    item["metadata"]["hard_negative"] = True
    item["metadata"]["word_count"] = word_count(text)
    return item


def collect_poetry_candidates(
    input_dir: Path,
    metadata_path: Path,
    existing_keys: Set[str],
    target_count: int,
    seed: int,
) -> List[Dict]:
    if target_count <= 0 or not input_dir.exists():
        return []

    metadata = load_poetry_metadata(metadata_path)
    raw_samples = build_poetry_samples(
        input_dir=input_dir,
        metadata=metadata,
        min_words=18,
        max_words=190,
        min_lines=2,
        max_lines=22,
        max_per_source=500,
        max_samples=max(target_count + len(existing_keys), target_count * 4),
        seed=seed,
    )

    candidates = []
    seen = set()
    for sample in raw_samples:
        text = sample.get("text", "")
        key = normalize_key(text)
        if not key or key in existing_keys or key in seen:
            continue
        seen.add(key)
        meta = dict(sample.get("metadata", {}))
        meta["hard_negative_reason"] = "poetry_style_or_lineation"
        candidates.append(
            {
                "text": text,
                "domain": "poetry",
                "source": "hard_negative:" + sample.get("source", "gutenberg_poetry"),
                "metadata": meta,
                "score": 10 + min(meta.get("line_count", 0), 20),
            }
        )

    return candidates[:target_count]


def literature_hardness_score(text: str) -> int:
    lower = text.lower()
    score = 0
    score += 3 if any(token in lower for token in ["thou", "thee", "hath", "doth", "ere", "nay"]) else 0
    score += 2 if any(mark in text for mark in [";", ":", "--", "—"]) else 0
    score += 2 if any(mark in text for mark in ['"', "“", "”", "'"]) else 0
    score += 1 if "!" in text or "?" in text else 0
    score += 1 if re.search(r"\b[A-Z][a-z]+(?:'s)?\b", text) else 0
    return score


def collect_literature_candidates(input_dir: Path, existing_keys: Set[str], target_count: int, seed: int) -> List[Dict]:
    if target_count <= 0 or not input_dir.exists():
        return []

    rng = random.Random(seed)
    candidates = []
    seen = set()

    for txt_path in sorted(input_dir.glob("*.txt")):
        raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        text = clean_literature_text(raw_text)
        for paragraph in split_paragraphs(text):
            wc = word_count(paragraph)
            if wc < 24 or wc > 75:
                continue

            lower = paragraph.lower()
            if any(pattern in lower for pattern in BAD_TEXT_PATTERNS):
                continue
            if lower.startswith("chapter "):
                continue

            score = literature_hardness_score(paragraph)
            if score < 2:
                continue

            key = normalize_key(paragraph)
            if key in existing_keys or key in seen:
                continue

            seen.add(key)
            candidates.append(
                {
                    "text": paragraph,
                    "domain": "literature",
                    "source": f"hard_negative:gutenberg_short:{txt_path.stem}",
                    "metadata": {
                        "source_file": txt_path.name,
                        "hard_negative_reason": "short_polished_or_archaic_literary_prose",
                    },
                    "score": score,
                }
            )

    rng.shuffle(candidates)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:target_count]


def clean_academic_text(text: str) -> str:
    text = str(text or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_academic_keyword(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in ACADEMIC_KEYWORDS)


def is_bad_section(section: Optional[str]) -> bool:
    if not section:
        return False
    lower = section.lower()
    return any(bad in lower for bad in BAD_SECTIONS)


def is_valid_academic_hard_negative(text: str, section: Optional[str]) -> bool:
    if is_bad_section(section):
        return False

    wc = len(text.split())
    if wc < 60 or wc > 260:
        return False

    lower = text.lower()
    if any(pattern in lower for pattern in BAD_TEXT_PATTERNS):
        return False
    if looks_structured_academic_noise(text):
        return False
    if not has_academic_keyword(text):
        return False

    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.15:
        return False

    return True


def looks_structured_academic_noise(text: str) -> bool:
    lower = text.lower()
    jsonish_markers = [
        '"pattern"',
        '"spec"',
        '"orth"',
        '"node_name"',
        "ent_type",
        "nbor_relop",
    ]
    if any(marker in lower for marker in jsonish_markers):
        return True

    bracket_count = sum(text.count(mark) for mark in ["{", "}", "[", "]"])
    if bracket_count >= 6:
        return True

    quote_colon_pairs = len(re.findall(r'"[A-Za-z_][A-Za-z0-9_]*"\s*:', text))
    if quote_colon_pairs >= 2:
        return True

    symbol_chars = sum(1 for ch in text if ch in "{}[]<>|\\")
    if symbol_chars / max(len(text), 1) > 0.025:
        return True

    return False


def find_academic_files(root: Path, max_files: int) -> List[Path]:
    if not root.exists():
        return []

    patterns = ["*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz"]
    files = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))

    files = [
        path
        for path in sorted(set(files))
        if "README" not in path.name.upper() and "dataset_infos" not in path.name
    ]
    if max_files > 0:
        files = files[:max_files]
    return files


def extract_academic_paragraphs(record: Dict) -> Iterable[Dict]:
    pdf_parse = record.get("pdf_parse", {})
    if not isinstance(pdf_parse, dict):
        return

    body_text = pdf_parse.get("body_text", [])
    if not isinstance(body_text, list):
        return

    for paragraph in body_text:
        if not isinstance(paragraph, dict):
            continue
        text = clean_academic_text(paragraph.get("text", ""))
        section = paragraph.get("section", "")
        if is_valid_academic_hard_negative(text, section):
            yield {"text": text, "section": section}


def collect_academic_candidates(
    input_dir: Path,
    existing_keys: Set[str],
    target_count: int,
    max_files: int,
) -> List[Dict]:
    if target_count <= 0 or not input_dir.exists():
        return []

    candidates = []
    seen = set()

    for data_file in find_academic_files(input_dir, max_files=max_files):
        for record in iter_json_records(data_file):
            raw_paper_id = str(record.get("paper_id", "") or "")
            paper_id = raw_paper_id if raw_paper_id and not re.fullmatch(r"\d{4}", raw_paper_id) else data_file.stem
            title = record.get("title", "")
            year = record.get("year", "")
            venue = record.get("venue", "")

            for paragraph in extract_academic_paragraphs(record):
                text = paragraph["text"]
                key = normalize_key(text)
                if key in existing_keys or key in seen:
                    continue

                seen.add(key)
                candidates.append(
                    {
                        "text": text,
                        "domain": "academic",
                        "source": f"hard_negative:acl_ocl:{paper_id}",
                        "metadata": {
                            "paper_id": paper_id,
                            "title": title,
                            "year": year,
                            "venue": venue,
                            "section": paragraph.get("section", ""),
                            "source_file": str(data_file.relative_to(input_dir)),
                            "hard_negative_reason": "formal_technical_academic_style",
                        },
                        "score": 5 + text.count(";") + text.count("(") + text.count(")"),
                    }
                )

                if len(candidates) >= target_count * 3:
                    break
            if len(candidates) >= target_count * 3:
                break
        if len(candidates) >= target_count * 3:
            break

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:target_count]


def save_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def build_report(samples: List[Dict], existing_paths: List[Path]) -> Dict:
    word_counts = [word_count(sample.get("text", "")) for sample in samples]
    return {
        "total_samples": len(samples),
        "domain_distribution": dict(Counter(sample.get("domain", "unknown") for sample in samples)),
        "source_distribution_top50": dict(Counter(sample.get("source", "unknown") for sample in samples).most_common(50)),
        "hard_negative_reason_distribution": dict(
            Counter(sample.get("metadata", {}).get("hard_negative_reason", "unknown") for sample in samples)
        ),
        "word_count": {
            "min": min(word_counts) if word_counts else None,
            "max": max(word_counts) if word_counts else None,
            "mean": sum(word_counts) / len(word_counts) if word_counts else None,
        },
        "existing_human_files_used_for_dedup": [str(path) for path in existing_paths],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build extra hard-negative human samples for detector optimization.")
    parser.add_argument("--existing_human", type=str, nargs="+", default=[str(path) for path in DEFAULT_EXISTING_HUMAN])
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--poetry_dir", type=str, default=str(DEFAULT_POETRY_DIR))
    parser.add_argument("--poetry_metadata", type=str, default=str(DEFAULT_POETRY_METADATA))
    parser.add_argument("--literature_dir", type=str, default=str(DEFAULT_LITERATURE_DIR))
    parser.add_argument("--academic_dir", type=str, default=str(DEFAULT_ACADEMIC_DIR))
    parser.add_argument("--poetry_target", type=int, default=650)
    parser.add_argument("--literature_target", type=int, default=200)
    parser.add_argument("--academic_target", type=int, default=150)
    parser.add_argument("--max_academic_files", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    existing_paths = [Path(path) for path in args.existing_human]
    existing_keys = load_existing_text_keys(existing_paths)

    print("=" * 70)
    print("Build Hard-Negative Human Seed")
    print("=" * 70)
    print(f"Existing human texts for dedup: {len(existing_keys)}")

    candidate_groups = [
        collect_poetry_candidates(
            input_dir=Path(args.poetry_dir),
            metadata_path=Path(args.poetry_metadata),
            existing_keys=existing_keys,
            target_count=args.poetry_target,
            seed=args.seed,
        ),
        collect_literature_candidates(
            input_dir=Path(args.literature_dir),
            existing_keys=existing_keys,
            target_count=args.literature_target,
            seed=args.seed,
        ),
        collect_academic_candidates(
            input_dir=Path(args.academic_dir),
            existing_keys=existing_keys,
            target_count=args.academic_target,
            max_files=args.max_academic_files,
        ),
    ]

    samples = []
    seen = set(existing_keys)
    for group in candidate_groups:
        for candidate in group:
            key = normalize_key(candidate["text"])
            if key in seen:
                continue
            seen.add(key)
            samples.append(
                make_hardneg_sample(
                    text=candidate["text"],
                    domain=candidate["domain"],
                    source=candidate["source"],
                    metadata=candidate["metadata"],
                    index=len(samples) + 1,
                )
            )

    output_path = Path(args.output)
    report_path = Path(args.report)
    save_jsonl(samples, output_path)
    report = build_report(samples, existing_paths=existing_paths)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved samples: {output_path}")
    print(f"Saved report: {report_path}")
    print(f"Total samples: {report['total_samples']}")
    print(f"Domain distribution: {report['domain_distribution']}")
    print(f"Reason distribution: {report['hard_negative_reason_distribution']}")


if __name__ == "__main__":
    main()
