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

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_gemini.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_gemini.jsonl"
DEFAULT_FAILED_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_gemini_failed.jsonl"

DEFAULT_BASE_URL = "https://api.gptsapi.net"
DEFAULT_MODEL = "gemini-3-flash-preview"

SYSTEM_PROMPT = (
    "Rewrite the user's passage in polished English. "
    "Preserve meaning, domain, tone, terminology, claims, examples, and approximate length. "
    "Use different wording and sentence structures. "
    "Do not summarize, omit details, or add new information. "
    "Return only the rewritten passage."
)


def load_jsonl(path: Path) -> List[Dict]:
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse line {line_id} in {path}: {e}")

    return samples


def append_jsonl(item: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_finished_task_ids(path: Path) -> Set[str]:
    """
    Only treat a task as finished if:
    1. task_id exists
    2. text is non-empty
    3. quality.passed_basic_quality_check is True

    This prevents old truncated / low-quality Gemini outputs from being skipped.
    """
    if not path.exists():
        return set()

    finished = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warning] Failed to parse finished output line {line_id}")
                continue

            task_id = item.get("task_id")
            text = item.get("text", "")
            quality = item.get("quality", {})

            if (
                task_id
                and isinstance(text, str)
                and text.strip()
                and isinstance(quality, dict)
                and quality.get("passed_basic_quality_check", False)
            ):
                finished.add(task_id)

    return finished


def clean_model_output(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(r"^```(?:text|json|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    prefixes = [
        "Here is the rewritten passage:",
        "Here is the rewrite:",
        "Rewritten passage:",
        "Rewrite:",
        "Paraphrased passage:",
        "Modernized passage:",
        "The rewritten passage is:",
        "Below is the rewritten passage:",
        "Here is the paraphrased passage:",
        "Here is a rewritten version:",
        "Sure, here is the rewritten passage:",
        "Certainly, here is the rewritten passage:",
        "Of course, here is the rewritten passage:",
        "Here is the rewritten paragraph:",
        "Rewritten paragraph:",
        "Paraphrased paragraph:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in ['"', "'"]:
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


def starts_with_meta_text(text: str) -> bool:
    lower = text.strip().lower()

    bad_prefixes = [
        "here is the rewritten passage",
        "here is the rewrite",
        "here is the rewritten paragraph",
        "rewritten passage:",
        "rewritten paragraph:",
        "paraphrased passage:",
        "paraphrased paragraph:",
        "modernized passage:",
        "the rewritten passage is:",
        "below is",
        "certainly!",
        "sure,",
        "of course,",
        "as requested,",
        "i have rewritten",
    ]

    return any(lower.startswith(prefix) for prefix in bad_prefixes)


def looks_truncated(text: str) -> bool:
    text = text.strip()

    if not text:
        return True

    valid_endings = (".", "?", "!", ")", "]", '"', "'", "”", "’")

    return text[-1] not in valid_endings


def get_quality_thresholds(domain: str) -> Dict:
    domain = (domain or "").lower()

    if domain == "academic":
        return {
            "min_abs_words": 20,
            "min_length_ratio": 0.50,
            "max_length_ratio": 2.00,
            "max_jaccard": 0.92,
            "fail_on_truncated": True,
        }

    return {
        "min_abs_words": 20,
        "min_length_ratio": 0.45,
        "max_length_ratio": 2.00,
        "max_jaccard": 0.82,
        "fail_on_truncated": True,
    }


def basic_quality_check(source_text: str, rewrite_text: str, domain: str = "literature") -> Dict:
    thresholds = get_quality_thresholds(domain)

    source_words = word_count(source_text)
    rewrite_words = word_count(rewrite_text)

    length_ratio = rewrite_words / max(source_words, 1)
    jaccard = lexical_jaccard(source_text, rewrite_text)

    issues = []

    if not rewrite_text.strip():
        issues.append("empty_output")

    if rewrite_words < thresholds["min_abs_words"]:
        issues.append("too_short_absolute")

    if length_ratio < thresholds["min_length_ratio"]:
        issues.append("too_short_relative")

    if length_ratio > thresholds["max_length_ratio"]:
        issues.append("too_long_relative")

    if jaccard > thresholds["max_jaccard"]:
        issues.append("too_similar_to_source")

    if thresholds["fail_on_truncated"] and looks_truncated(rewrite_text):
        issues.append("possibly_truncated")

    if starts_with_meta_text(rewrite_text):
        issues.append("contains_meta_text")

    return {
        "domain": domain,
        "source_word_count": source_words,
        "rewrite_word_count": rewrite_words,
        "length_ratio": round(length_ratio, 4),
        "lexical_jaccard": round(jaccard, 4),
        "quality_thresholds": thresholds,
        "quality_issues": issues,
        "passed_basic_quality_check": len(issues) == 0,
    }


def call_gemini(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
    sleep_base: float,
) -> Dict:
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
            )

            if not response.choices:
                return {
                    "content": "",
                    "finish_reason": "no_choices",
                    "usage": None,
                }

            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)

            usage = None
            if hasattr(response, "usage") and response.usage is not None:
                try:
                    usage = response.usage.model_dump()
                except Exception:
                    usage = str(response.usage)

            return {
                "content": content,
                "finish_reason": finish_reason,
                "usage": usage,
            }

        except Exception as e:
            last_error = e
            wait_time = sleep_base * (2 ** attempt) + random.uniform(0, 1.0)
            print(f"\nRequest failed. Attempt {attempt + 1}/{max_retries}. Error: {e}")
            print(f"Sleeping {wait_time:.2f} seconds...")
            time.sleep(wait_time)

    raise RuntimeError(f"Gemini request failed after {max_retries} retries: {last_error}")


def build_output_item(
    task: Dict,
    rewrite_text: str,
    model: str,
    quality: Dict,
    finish_reason: Optional[str],
    usage,
) -> Dict:
    return {
        "id": task["task_id"].replace("rewrite_", "llm_gemini_"),
        "task_id": task["task_id"],
        "source_id": task["source_id"],
        "pair_id": task["pair_id"],
        "text": rewrite_text,
        "label": 1,
        "domain": task.get("domain", "literature"),
        "source": "gemini",
        "generator": "gemini",
        "model": model,
        "prompt_type": task.get("prompt_type", "unknown"),
        "generation": "llm_rewrite",
        "source_text": task.get("source_text", ""),
        "quality": quality,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Gemini rewrites via GPTSAPI.")

    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--failed", type=str, default=str(DEFAULT_FAILED_PATH))
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--limit", type=int, default=50)

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Recommended for Gemini academic: 0.4. Use 0.6-0.7 for more diverse literature rewrites.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Recommended for Gemini academic: 0.8.",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=3000,
        help="Recommended: 3000 for Gemini academic rewrites.",
    )

    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--sleep_base", type=float, default=2.0)

    parser.add_argument(
        "--allow_low_quality",
        action="store_true",
        help="If set, write low-quality generations to output instead of failed file.",
    )

    parser.add_argument(
        "--allow_length_finish",
        action="store_true",
        help="If set, do not automatically fail finish_reason='length'.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GPTSAPI_API_KEY")
    if not api_key:
        raise ValueError(
            "GPTSAPI_API_KEY is not set. Please add it to .env:\n"
            "GPTSAPI_API_KEY=your_gptsapi_api_key"
        )

    model = args.model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL

    input_path = Path(args.input)
    output_path = Path(args.output)
    failed_path = Path(args.failed)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
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
    print("Gemini Rewrite Generation via GPTSAPI")
    print("=" * 70)
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Failed path: {failed_path}")
    print(f"Base URL: {args.base_url}")
    print(f"Model: {model}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Sleep: {args.sleep}")
    print(f"Allow low quality: {args.allow_low_quality}")
    print(f"Allow finish_reason=length: {args.allow_length_finish}")
    print(f"Total tasks in input: {len(tasks)}")
    print(f"Already finished valid tasks: {len(finished_task_ids)}")
    print(f"Tasks to process this run: {len(remaining_tasks)}")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    quality_failed_count = 0
    truncated_count = 0

    for task in tqdm(remaining_tasks, desc="Generating Gemini rewrites"):
        finish_reason = None
        usage = None
        raw_output = ""

        try:
            source_text = task.get("source_text", "")
            prompt = task["prompt"]
            domain = task.get("domain", "literature")

            result = call_gemini(
                client=client,
                model=model,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                sleep_base=args.sleep_base,
            )

            raw_output = result.get("content", "")
            finish_reason = result.get("finish_reason")
            usage = result.get("usage")

            rewrite_text = clean_model_output(raw_output)

            quality = basic_quality_check(
                source_text=source_text,
                rewrite_text=rewrite_text,
                domain=domain,
            )

            output_item = build_output_item(
                task=task,
                rewrite_text=rewrite_text,
                model=model,
                quality=quality,
                finish_reason=finish_reason,
                usage=usage,
            )

            if not rewrite_text.strip():
                raise ValueError("Empty rewrite_text extracted from API response.")

            if finish_reason == "length" and not args.allow_length_finish:
                truncated_count += 1
                failed_item = {
                    "task_id": task.get("task_id"),
                    "source_id": task.get("source_id"),
                    "pair_id": task.get("pair_id"),
                    "generator": "gemini",
                    "model": model,
                    "error": (
                        f"Generation truncated because finish_reason=length. "
                        f"Try increasing max_tokens. Current max_tokens={args.max_tokens}"
                    ),
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "source_text": source_text,
                    "raw_output": raw_output,
                    "rewrite_text": rewrite_text,
                    "quality": quality,
                }
                append_jsonl(failed_item, failed_path)

            elif not args.allow_low_quality and not quality["passed_basic_quality_check"]:
                quality_failed_count += 1
                failed_item = {
                    "task_id": task.get("task_id"),
                    "source_id": task.get("source_id"),
                    "pair_id": task.get("pair_id"),
                    "generator": "gemini",
                    "model": model,
                    "error": f"Quality check failed: {quality}",
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "source_text": source_text,
                    "raw_output": raw_output,
                    "rewrite_text": rewrite_text,
                    "quality": quality,
                }
                append_jsonl(failed_item, failed_path)

            else:
                append_jsonl(output_item, output_path)
                success_count += 1

            time.sleep(args.sleep)

        except Exception as e:
            failed_item = {
                "task_id": task.get("task_id"),
                "source_id": task.get("source_id"),
                "pair_id": task.get("pair_id"),
                "generator": "gemini",
                "model": model,
                "error": str(e),
                "finish_reason": finish_reason,
                "usage": usage,
                "raw_output": raw_output,
                "source_text": task.get("source_text", ""),
            }
            append_jsonl(failed_item, failed_path)
            failed_count += 1

    print("\n" + "=" * 70)
    print("Generation finished")
    print("=" * 70)
    print(f"Success written to output: {success_count}")
    print(f"Request / extraction failed: {failed_count}")
    print(f"Quality failed: {quality_failed_count}")
    print(f"Truncated finish_reason=length: {truncated_count}")
    print(f"Output saved to: {output_path}")
    print(f"Failures saved to: {failed_path}")


if __name__ == "__main__":
    main()
