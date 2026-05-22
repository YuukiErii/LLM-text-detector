import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_deepseek.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_deepseek.jsonl"
DEFAULT_FAILED_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_deepseek_failed.jsonl"

DEFAULT_MODEL = "deepseek-v4-pro"
# If deepseek-v4-pro is unavailable, override it from the command line:
# --model deepseek-chat
# Or:
# --model deepseek-v4-flash


SYSTEM_PROMPT = (
    "You are a careful English literary rewriting assistant. "
    "Your task is to rewrite the user's passage while preserving its meaning, "
    "scene, narrative perspective, tone, and approximate length. "
    "Return only the rewritten passage. Do not add explanations, titles, bullets, or comments."
)


def load_jsonl(path: Path) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def append_jsonl(item: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_finished_task_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()

    finished = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if "task_id" in item:
                        finished.add(item["task_id"])
                except json.JSONDecodeError:
                    continue
    return finished


def clean_model_output(text: str) -> str:
    text = text.strip()

    # Remove common wrapping markers.
    text = re.sub(r"^```(?:text|json|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Remove occasional explanatory prefixes emitted by the model.
    prefixes = [
        "Here is the rewritten passage:",
        "Here is the rewrite:",
        "Rewritten passage:",
        "Rewrite:",
        "Paraphrased passage:",
        "Modernized passage:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # If the model wraps the output in quotes, carefully remove one layer.
    if len(text) >= 2:
        if (text[0] == text[-1]) and text[0] in ['"', "'"]:
            text = text[1:-1].strip()

    return text


def word_count(text: str) -> int:
    return len(text.split())


def lexical_jaccard(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[A-Za-z']+", a.lower()))
    tokens_b = set(re.findall(r"[A-Za-z']+", b.lower()))

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def basic_quality_check(source_text: str, rewrite_text: str) -> Dict:
    source_words = word_count(source_text)
    rewrite_words = word_count(rewrite_text)

    length_ratio = rewrite_words / max(source_words, 1)
    jaccard = lexical_jaccard(source_text, rewrite_text)

    issues = []

    if rewrite_words < 20:
        issues.append("too_short_absolute")

    if length_ratio < 0.45:
        issues.append("too_short_relative")

    if length_ratio > 2.0:
        issues.append("too_long_relative")

    if jaccard > 0.82:
        issues.append("too_similar_to_source")

    lower = rewrite_text.lower()
    bad_phrases = [
        "here is the rewritten passage",
        "here is the rewrite",
        "i have rewritten",
        "as requested",
        "passage:",
    ]

    if any(p in lower for p in bad_phrases):
        issues.append("contains_meta_text")

    return {
        "source_word_count": source_words,
        "rewrite_word_count": rewrite_words,
        "length_ratio": round(length_ratio, 4),
        "lexical_jaccard": round(jaccard, 4),
        "quality_issues": issues,
        "passed_basic_quality_check": len(issues) == 0,
    }


def call_deepseek(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
    sleep_base: float,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
                # Text rewriting does not need thinking mode.
                extra_body={"thinking": {"type": "disabled"}},
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            last_error = e
            wait_time = sleep_base * (2 ** attempt) + random.uniform(0, 1.0)
            print(f"\nRequest failed. Attempt {attempt + 1}/{max_retries}. Error: {e}")
            print(f"Sleeping {wait_time:.2f} seconds...")
            time.sleep(wait_time)

    raise RuntimeError(f"DeepSeek request failed after {max_retries} retries: {last_error}")


def build_output_item(task: Dict, rewrite_text: str, model: str, quality: Dict) -> Dict:
    return {
        "id": task["task_id"].replace("rewrite_", "llm_deepseek_"),
        "task_id": task["task_id"],
        "source_id": task["source_id"],
        "pair_id": task["pair_id"],
        "text": rewrite_text,
        "label": 1,
        "domain": task.get("domain", "literature"),
        "source": "deepseek",
        "generator": "deepseek",
        "model": model,
        "prompt_type": task.get("prompt_type", "unknown"),
        "generation": "llm_rewrite",
        "source_text": task.get("source_text", ""),
        "quality": quality,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DeepSeek rewrites for LLM text detection dataset.")

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Input JSONL file containing rewrite prompts.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSONL file for successful rewrites.",
    )

    parser.add_argument(
        "--failed",
        type=str,
        default=str(DEFAULT_FAILED_PATH),
        help="Output JSONL file for failed tasks.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="DeepSeek model name, e.g. deepseek-v4-pro, deepseek-v4-flash, or deepseek-chat.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of new tasks to process. Use -1 for all remaining tasks.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p.",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=650,
        help="Maximum output tokens.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between successful requests.",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum retries per request.",
    )

    parser.add_argument(
        "--sleep_base",
        type=float,
        default=2.0,
        help="Base sleep seconds for exponential backoff.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. "
            "Please create a .env file in the project root and add:\n"
            "DEEPSEEK_API_KEY=your_api_key"
        )

    input_path = Path(args.input)
    output_path = Path(args.output)
    failed_path = Path(args.failed)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    tasks = load_jsonl(input_path)
    finished_task_ids = load_finished_task_ids(output_path)

    remaining_tasks = [
        task for task in tasks
        if task.get("task_id") not in finished_task_ids
    ]

    if args.limit is not None and args.limit >= 0:
        remaining_tasks = remaining_tasks[:args.limit]

    print("=" * 70)
    print("DeepSeek Rewrite Generation")
    print("=" * 70)
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Failed path: {failed_path}")
    print(f"Model: {args.model}")
    print(f"Total tasks in input: {len(tasks)}")
    print(f"Already finished: {len(finished_task_ids)}")
    print(f"Tasks to process this run: {len(remaining_tasks)}")
    print("=" * 70)

    success_count = 0
    failed_count = 0

    for task in tqdm(remaining_tasks, desc="Generating rewrites"):
        try:
            source_text = task.get("source_text", "")
            prompt = task["prompt"]

            raw_output = call_deepseek(
                client=client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                sleep_base=args.sleep_base,
            )

            rewrite_text = clean_model_output(raw_output)
            quality = basic_quality_check(source_text, rewrite_text)

            output_item = build_output_item(
                task=task,
                rewrite_text=rewrite_text,
                model=args.model,
                quality=quality,
            )

            append_jsonl(output_item, output_path)
            success_count += 1

            time.sleep(args.sleep)

        except Exception as e:
            failed_item = {
                "task_id": task.get("task_id"),
                "source_id": task.get("source_id"),
                "pair_id": task.get("pair_id"),
                "generator": "deepseek",
                "model": args.model,
                "error": str(e),
            }
            append_jsonl(failed_item, failed_path)
            failed_count += 1

    print("\n" + "=" * 70)
    print("Generation finished")
    print("=" * 70)
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Output saved to: {output_path}")
    print(f"Failures saved to: {failed_path}")


if __name__ == "__main__":
    main()
