import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train_hardneg_p50_l200_a150.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_chatgpt_hard_positive.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "rewrite_prompts_chatgpt_hard_positive_report.json"

DEFAULT_DOMAIN_QUOTAS = "literature=80,academic=20,poetry=20"

PROMPTS = {
    "literature": [
        {
            "prompt_type": "chatgpt_hard_literary_minimal_edit",
            "template": (
                "Rewrite the following literary passage as a conservative human-like paraphrase. "
                "Preserve the scene, mood, narrative perspective, period flavor, and approximate length. "
                "Do not modernize aggressively, do not summarize, and do not add commentary. "
                "Change enough wording and sentence structure that it is clearly a rewrite, while keeping it "
                "close to the original literary style.\n\nPASSAGE:\n{ text }"
            ),
        },
        {
            "prompt_type": "chatgpt_hard_literary_archaic_preserving",
            "template": (
                "Rewrite the following passage in polished literary English while preserving any old-fashioned, "
                "formal, or ornate flavor. Keep the same events, imagery, and emotional tone. Use different "
                "phrasing, but avoid making the passage sound like a modern summary. Return only the rewrite.\n\n"
                "PASSAGE:\n{ text }"
            ),
        },
        {
            "prompt_type": "chatgpt_hard_literary_polished_imitation",
            "template": (
                "Create a polished literary rewrite of the passage below. It should read like careful human prose, "
                "not like a simplified explanation. Preserve the concrete details, pacing, and approximate length, "
                "but vary vocabulary and syntax. Return only the rewritten passage.\n\nPASSAGE:\n{ text }"
            ),
        },
    ],
    "academic": [
        {
            "prompt_type": "chatgpt_hard_academic_minimal_edit",
            "template": (
                "Paraphrase the following NLP research paragraph in clear, polished academic English. Preserve all "
                "technical claims, terminology, datasets, methods, and caveats. Keep approximately the same length "
                "and avoid adding new claims. Return only the rewritten paragraph.\n\nPARAGRAPH:\n{ text }"
            ),
        },
        {
            "prompt_type": "chatgpt_hard_academic_human_polish",
            "template": (
                "Rewrite the following research paragraph as if a careful human author lightly revised it for "
                "clarity. Keep the argument structure and technical details intact. Do not summarize or simplify "
                "away terminology. Return only the rewrite.\n\nPARAGRAPH:\n{ text }"
            ),
        },
    ],
    "poetry": [
        {
            "prompt_type": "chatgpt_hard_poetry_line_preserving",
            "template": (
                "Rewrite the following poem or stanza as a poem, preserving the speaker, imagery, mood, and roughly "
                "the same number of lines. Use fresh wording but keep a natural poetic diction. Do not explain it "
                "and do not turn it into prose.\n\nPOEM:\n{ text }"
            ),
        },
        {
            "prompt_type": "chatgpt_hard_poetry_archaic_preserving",
            "template": (
                "Create a poetic rewrite of the stanza below that keeps its old-fashioned or formal poetic feel. "
                "Preserve line breaks as much as possible, keep the same emotional movement, and change the wording. "
                "Return only the rewritten poem.\n\nPOEM:\n{ text }"
            ),
        },
    ],
}


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}, line {line_id}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def save_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_domain_quotas(value: str) -> Dict[str, int]:
    quotas = {}
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            domain, count = part.split("=", 1)
        elif ":" in part:
            domain, count = part.split(":", 1)
        else:
            raise ValueError(f"Quota must use domain=count format, got: {part}")
        domain = domain.strip()
        if domain not in PROMPTS:
            raise ValueError(f"Unsupported domain for hard-positive prompt: {domain}")
        count_value = int(count)
        if count_value < 0:
            raise ValueError(f"Quota must be non-negative, got: {part}")
        quotas[domain] = count_value
    return quotas


def word_count(text: str) -> int:
    return len(str(text or "").split())


def select_candidates(rows: List[Dict], quotas: Dict[str, int], seed: int) -> List[Dict]:
    rng = random.Random(seed)
    selected = []
    used_ids = set()

    hard_negative_rows = [
        row for row in rows
        if row.get("label") == 0 and row.get("prompt_type") == "human_hard_negative"
    ]

    for domain, quota in quotas.items():
        domain_rows = [row for row in hard_negative_rows if row.get("domain") == domain]
        domain_rows.sort(key=lambda row: word_count(row.get("text", "")))
        rng.shuffle(domain_rows)

        if len(domain_rows) < quota:
            raise ValueError(
                f"Not enough hard-negative source rows for {domain}: "
                f"need {quota}, found {len(domain_rows)}"
            )

        for row in domain_rows[:quota]:
            sample_id = row.get("id")
            if sample_id in used_ids:
                continue
            used_ids.add(sample_id)
            selected.append(row)

    return selected


def build_prompt_task(sample: Dict, task_index: int) -> Dict:
    domain = sample.get("domain", "literature")
    prompt_options = PROMPTS[domain]
    prompt_config = prompt_options[(task_index - 1) % len(prompt_options)]
    text = sample.get("text", "")
    prompt = prompt_config["template"].replace("{ text }", text)
    source_metadata = dict(sample.get("metadata") or {})

    return {
        "task_id": f"rewrite_chatgpt_hardpos_{task_index:06d}",
        "source_id": sample["id"],
        "pair_id": sample["pair_id"],
        "domain": domain,
        "source": sample.get("source", "unknown"),
        "source_text": text,
        "prompt_type": prompt_config["prompt_type"],
        "prompt": prompt,
        "metadata": {
            "step": "step5_chatgpt_hard_positive",
            "target_generator": "chatgpt",
            "source_prompt_type": sample.get("prompt_type"),
            "source_generation": sample.get("generation"),
            "source_metadata": source_metadata,
        },
    }


def summarize(rows: List[Dict]) -> Dict:
    return {
        "num_samples": len(rows),
        "domain_distribution": dict(Counter(row.get("domain", "unknown") for row in rows)),
        "prompt_type_distribution": dict(Counter(row.get("prompt_type", "unknown") for row in rows)),
        "source_distribution_top20": dict(Counter(row.get("source", "unknown") for row in rows).most_common(20)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ChatGPT-style hard-positive rewrite prompts.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--domain_quotas", type=str, default=DEFAULT_DOMAIN_QUOTAS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    quotas = parse_domain_quotas(args.domain_quotas)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    rows = load_jsonl(input_path)
    selected = select_candidates(rows, quotas=quotas, seed=args.seed)
    tasks = [build_prompt_task(sample, idx) for idx, sample in enumerate(selected, start=1)]
    save_jsonl(tasks, output_path)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "domain_quotas": quotas,
        "seed": args.seed,
        "available_hard_negative_sources": summarize(
            [row for row in rows if row.get("label") == 0 and row.get("prompt_type") == "human_hard_negative"]
        ),
        "selected_sources": summarize(selected),
        "prompt_tasks": summarize(tasks),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Prepare ChatGPT Hard-Positive Prompts")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Saved tasks: {len(tasks)}")
    print(f"Domain distribution: {report['prompt_tasks']['domain_distribution']}")
    print(f"Prompt distribution: {report['prompt_tasks']['prompt_type_distribution']}")


if __name__ == "__main__":
    main()
