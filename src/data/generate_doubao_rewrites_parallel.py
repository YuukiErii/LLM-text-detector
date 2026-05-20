import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_doubao.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_doubao.jsonl"
DEFAULT_FAILED_PATH = PROJECT_ROOT / "data" / "processed" / "llm_rewrite_doubao_failed.jsonl"

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL_ID = "doubao-seed-2-0-pro-260215"

SYSTEM_PROMPT = (
    "You are a careful English literary rewriting assistant. "
    "Your task is to rewrite the user's passage while preserving its meaning, "
    "scene, narrative perspective, tone, and approximate length. "
    "Return only the rewritten passage. Do not add explanations, titles, bullets, or comments."
)

write_lock = threading.Lock()


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


def append_jsonl_threadsafe(item: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_finished_task_ids(path: Path) -> Set[str]:
    """
    Only treat a task as finished if:
    1. task_id exists
    2. text is non-empty
    3. quality.passed_basic_quality_check is True
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


def basic_quality_check(source_text: str, rewrite_text: str) -> Dict:
    source_words = word_count(source_text)
    rewrite_words = word_count(rewrite_text)

    length_ratio = rewrite_words / max(source_words, 1)
    jaccard = lexical_jaccard(source_text, rewrite_text)

    issues = []

    if not rewrite_text.strip():
        issues.append("empty_output")

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
        "below is",
        "certainly!",
        "sure,",
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


def response_to_dict(response):
    if response is None:
        return None

    if isinstance(response, dict):
        return response

    if hasattr(response, "model_dump"):
        return response.model_dump()

    if hasattr(response, "dict"):
        return response.dict()

    return None


def dump_response_for_debug(response, task_id: str, output_dir: Path) -> None:
    debug_dir = output_dir / "debug_doubao_chat_completions_parallel"
    debug_dir.mkdir(parents=True, exist_ok=True)

    debug_path = debug_dir / f"{task_id}.json"

    try:
        data = response_to_dict(response)
        if data is None:
            data = {"raw_repr": repr(response)}

        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"Failed to dump response: {e}\n")
            f.write(repr(response))


def call_doubao_chat_completions(
    client: OpenAI,
    model_id: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
    sleep_base: float,
    debug_response: bool,
    task_id: str,
    debug_output_dir: Path,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
            )

            if debug_response:
                dump_response_for_debug(response, task_id, debug_output_dir)

            if not response.choices:
                return ""

            message = response.choices[0].message
            content = getattr(message, "content", None)

            if content is None:
                return ""

            return str(content).strip()

        except Exception as e:
            last_error = e
            wait_time = sleep_base * (2 ** attempt) + random.uniform(0, 1.0)
            print(f"\nRequest failed for {task_id}. Attempt {attempt + 1}/{max_retries}. Error: {e}")
            print(f"Sleeping {wait_time:.2f} seconds...")
            time.sleep(wait_time)

    raise RuntimeError(f"Doubao request failed after {max_retries} retries: {last_error}")


def build_output_item(task: Dict, rewrite_text: str, model_id: str, quality: Dict) -> Dict:
    return {
        "id": task["task_id"].replace("rewrite_", "llm_doubao_"),
        "task_id": task["task_id"],
        "source_id": task["source_id"],
        "pair_id": task["pair_id"],
        "text": rewrite_text,
        "label": 1,
        "domain": task.get("domain", "literature"),
        "source": "doubao",
        "generator": "doubao",
        "model": model_id,
        "prompt_type": task.get("prompt_type", "unknown"),
        "generation": "llm_rewrite",
        "source_text": task.get("source_text", ""),
        "quality": quality,
    }


def process_one_task(
    task: Dict,
    client: OpenAI,
    model_id: str,
    output_path: Path,
    failed_path: Path,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
    sleep_base: float,
    sleep: float,
    debug_response: bool,
    strict_quality: bool,
    debug_output_dir: Path,
) -> Tuple[str, str]:
    """
    Returns:
        ("success", task_id)
        ("failed", task_id)
        ("quality_failed", task_id)
    """
    task_id = task.get("task_id", "unknown_task")

    try:
        source_text = task.get("source_text", "")
        prompt = task["prompt"]

        raw_output = call_doubao_chat_completions(
            client=client,
            model_id=model_id,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
            sleep_base=sleep_base,
            debug_response=debug_response,
            task_id=task_id,
            debug_output_dir=debug_output_dir,
        )

        rewrite_text = clean_model_output(raw_output)
        quality = basic_quality_check(source_text, rewrite_text)

        if not rewrite_text.strip():
            raise ValueError("Empty rewrite_text extracted from API response.")

        output_item = build_output_item(
            task=task,
            rewrite_text=rewrite_text,
            model_id=model_id,
            quality=quality,
        )

        if strict_quality and not quality["passed_basic_quality_check"]:
            failed_item = {
                "task_id": task.get("task_id"),
                "source_id": task.get("source_id"),
                "pair_id": task.get("pair_id"),
                "generator": "doubao",
                "model": model_id,
                "error": f"Quality check failed: {quality}",
                "source_text": source_text,
                "raw_output": raw_output,
                "rewrite_text": rewrite_text,
                "quality": quality,
            }
            append_jsonl_threadsafe(failed_item, failed_path)
            return "quality_failed", task_id

        append_jsonl_threadsafe(output_item, output_path)

        if sleep > 0:
            time.sleep(sleep)

        return "success", task_id

    except Exception as e:
        failed_item = {
            "task_id": task.get("task_id"),
            "source_id": task.get("source_id"),
            "pair_id": task.get("pair_id"),
            "generator": "doubao",
            "model": model_id,
            "error": str(e),
            "source_text": task.get("source_text", ""),
        }
        append_jsonl_threadsafe(failed_item, failed_path)

        if sleep > 0:
            time.sleep(sleep)

        return "failed", task_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Doubao rewrites in parallel using Volcengine Ark Chat Completions API."
    )

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
        default=None,
        help="Doubao model ID. If omitted, use DOUBAO_MODEL_ID from .env.",
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Volcengine Ark OpenAI-compatible base URL.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of new tasks to process. Use -1 for all remaining tasks.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of concurrent workers.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=900,
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep seconds after each request inside a worker.",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sleep_base",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--debug_response",
        action="store_true",
        help="Save raw API responses to outputs/debug_doubao_chat_completions_parallel.",
    )

    parser.add_argument(
        "--strict_quality",
        action="store_true",
        help="If set, failed quality-check generations are written to failed file instead of output file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError(
            "ARK_API_KEY is not set. "
            "Please add it to .env:\n"
            "ARK_API_KEY=your_api_key"
        )

    model_id = args.model or os.getenv("DOUBAO_MODEL_ID") or DEFAULT_MODEL_ID

    input_path = Path(args.input)
    output_path = Path(args.output)
    failed_path = Path(args.failed)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    tasks = load_jsonl(input_path)
    finished_task_ids = load_finished_task_ids(output_path)

    remaining_tasks = [
        task for task in tasks
        if task.get("task_id") not in finished_task_ids
    ]

    if args.limit is not None and args.limit >= 0:
        remaining_tasks = remaining_tasks[:args.limit]

    print("=" * 70)
    print("Doubao Rewrite Generation Parallel")
    print("=" * 70)
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Failed path: {failed_path}")
    print(f"Base URL: {args.base_url}")
    print(f"Model ID: {model_id}")
    print(f"Total tasks in input: {len(tasks)}")
    print(f"Already finished valid tasks: {len(finished_task_ids)}")
    print(f"Tasks to process this run: {len(remaining_tasks)}")
    print(f"Workers: {args.workers}")
    print(f"Sleep per task: {args.sleep}")
    print(f"Strict quality mode: {args.strict_quality}")
    print(f"Debug response: {args.debug_response}")
    print("=" * 70)

    if not remaining_tasks:
        print("No remaining tasks to process.")
        return

    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
    )

    debug_output_dir = PROJECT_ROOT / "outputs"

    success_count = 0
    failed_count = 0
    quality_failed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_one_task,
                task,
                client,
                model_id,
                output_path,
                failed_path,
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.max_retries,
                args.sleep_base,
                args.sleep,
                args.debug_response,
                args.strict_quality,
                debug_output_dir,
            )
            for task in remaining_tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Doubao rewrites"):
            status, task_id = future.result()

            if status == "success":
                success_count += 1
            elif status == "quality_failed":
                quality_failed_count += 1
            else:
                failed_count += 1

    print("\n" + "=" * 70)
    print("Parallel generation finished")
    print("=" * 70)
    print(f"Success written to output: {success_count}")
    print(f"Request / extraction failed: {failed_count}")
    print(f"Quality failed: {quality_failed_count}")
    print(f"Output saved to: {output_path}")
    print(f"Failures saved to: {failed_path}")
    print(f"Debug responses saved under: {debug_output_dir / 'debug_doubao_chat_completions_parallel'}")


if __name__ == "__main__":
    main()