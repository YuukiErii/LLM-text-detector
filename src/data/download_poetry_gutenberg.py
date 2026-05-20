import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "external_human" / "poetry" / "gutenberg_poetry"
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_DIR / "metadata.json"

GUTENDEX_BOOK_URL = "https://gutendex.com/books/{book_id}"

DEFAULT_BOOK_IDS = [
    12242,  # Emily Dickinson
    1279,   # Robert Burns
    1322,   # Walt Whitman
    1934,   # William Blake
    23684,  # John Keats
    4800,   # Percy Bysshe Shelley
]


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "book"


def fetch_json(url: str) -> Dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-text-detector-poetry-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-text-detector-poetry-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()

    for encoding in ["utf-8", "latin-1"]:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="ignore")


def select_plain_text_url(formats: Dict[str, str]) -> Optional[str]:
    candidates = []

    for mime_type, url in formats.items():
        if "text/plain" not in mime_type:
            continue
        if not isinstance(url, str):
            continue
        if "readme" in url.lower():
            continue
        candidates.append((mime_type, url))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (".utf-8" not in item[1].lower(), item[0]))
    return candidates[0][1]


def download_books(book_ids: List[int], output_dir: Path, sleep: float) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for book_id in book_ids:
        print(f"Fetching metadata for Gutenberg book {book_id}...")
        book = fetch_json(GUTENDEX_BOOK_URL.format(book_id=book_id))

        title = book.get("title", f"book_{book_id}")
        authors = [author.get("name", "") for author in book.get("authors", []) if author.get("name")]
        text_url = select_plain_text_url(book.get("formats", {}))

        if not text_url:
            print(f"  [skip] No usable plain text URL for {book_id}: {title}")
            continue

        filename = f"{book_id}_{slugify(title)}.txt"
        output_path = output_dir / filename

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  [cached] {output_path}")
        else:
            print(f"  Downloading text: {text_url}")
            text = fetch_text(text_url)
            output_path.write_text(text, encoding="utf-8")
            time.sleep(sleep)

        item = {
            "gutenberg_id": book_id,
            "title": title,
            "authors": authors,
            "subjects": book.get("subjects", []),
            "bookshelves": book.get("bookshelves", []),
            "text_url": text_url,
            "local_path": str(output_path),
        }
        metadata.append(item)

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Download public-domain English poetry from Project Gutenberg.")

    parser.add_argument(
        "--book_ids",
        type=int,
        nargs="*",
        default=DEFAULT_BOOK_IDS,
        help="Project Gutenberg book IDs to download.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save raw poetry text files.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(DEFAULT_METADATA_PATH),
        help="Path to save metadata JSON.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between downloads.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    metadata_path = Path(args.metadata)

    metadata = download_books(
        book_ids=args.book_ids,
        output_dir=output_dir,
        sleep=args.sleep,
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Downloaded poetry books")
    print("=" * 70)
    print(f"Books saved: {len(metadata)}")
    print(f"Output dir: {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
