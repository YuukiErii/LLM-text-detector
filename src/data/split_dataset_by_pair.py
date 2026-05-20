import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "full_dataset.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42


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
                print(f"[Warning] Failed to parse line {line_id}: {e}")

    return samples


def save_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def group_by_pair_id(samples: List[Dict]) -> Dict[str, List[Dict]]:
    pair_groups = defaultdict(list)

    missing_pair_id = 0

    for sample in samples:
        pair_id = sample.get("pair_id")

        if not pair_id:
            missing_pair_id += 1
            pair_id = f"missing_pair_{missing_pair_id:06d}"
            sample["pair_id"] = pair_id

        pair_groups[pair_id].append(sample)

    if missing_pair_id > 0:
        print(f"[Warning] Found {missing_pair_id} samples without pair_id. Assigned temporary pair_id.")

    return dict(pair_groups)


def get_pair_label_signature(group: List[Dict]) -> str:
    labels = sorted(set(sample.get("label") for sample in group))

    if labels == [0]:
        return "human_only"
    if labels == [1]:
        return "llm_only"
    if labels == [0, 1]:
        return "human_and_llm"

    return "unknown"


def split_pair_ids(
    pair_groups: Dict[str, List[Dict]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split pair_ids into train / valid / test.

    This implementation stratifies by pair signature:
    - human_only
    - llm_only
    - human_and_llm

    In most cases, useful pairs should be human_and_llm.
    """

    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {ratio_sum}. "
            f"train={train_ratio}, valid={valid_ratio}, test={test_ratio}"
        )

    rng = random.Random(seed)

    signature_to_pair_ids = defaultdict(list)

    for pair_id, group in pair_groups.items():
        signature = get_pair_label_signature(group)
        signature_to_pair_ids[signature].append(pair_id)

    train_pair_ids = []
    valid_pair_ids = []
    test_pair_ids = []

    for signature, pair_ids in signature_to_pair_ids.items():
        pair_ids = list(pair_ids)
        rng.shuffle(pair_ids)

        n = len(pair_ids)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        train_ids = pair_ids[:n_train]
        valid_ids = pair_ids[n_train:n_train + n_valid]
        test_ids = pair_ids[n_train + n_valid:]

        train_pair_ids.extend(train_ids)
        valid_pair_ids.extend(valid_ids)
        test_pair_ids.extend(test_ids)

        print(f"\nPair signature: {signature}")
        print(f"  total pairs: {n}")
        print(f"  train pairs: {len(train_ids)}")
        print(f"  valid pairs: {len(valid_ids)}")
        print(f"  test pairs:  {len(test_ids)}")

    rng.shuffle(train_pair_ids)
    rng.shuffle(valid_pair_ids)
    rng.shuffle(test_pair_ids)

    return train_pair_ids, valid_pair_ids, test_pair_ids


def collect_samples(pair_ids: List[str], pair_groups: Dict[str, List[Dict]], split_name: str) -> List[Dict]:
    samples = []

    for pair_id in pair_ids:
        group = pair_groups[pair_id]

        for sample in group:
            item = dict(sample)
            item["split"] = split_name
            samples.append(item)

    return samples


def summarize_samples(samples: List[Dict]) -> Dict:
    label_counter = Counter(sample.get("label") for sample in samples)
    generator_counter = Counter(sample.get("generator", "unknown") for sample in samples)
    source_counter = Counter(sample.get("source", "unknown") for sample in samples)
    prompt_counter = Counter(sample.get("prompt_type", "unknown") for sample in samples)
    domain_counter = Counter(sample.get("domain", "unknown") for sample in samples)
    pair_ids = set(sample.get("pair_id") for sample in samples)

    word_lengths = [len(sample.get("text", "").split()) for sample in samples]

    return {
        "num_samples": len(samples),
        "num_pair_ids": len(pair_ids),
        "label_distribution": dict(label_counter),
        "generator_distribution": dict(generator_counter),
        "source_distribution": dict(source_counter),
        "prompt_type_distribution": dict(prompt_counter),
        "domain_distribution": dict(domain_counter),
        "word_length": {
            "min": min(word_lengths) if word_lengths else 0,
            "max": max(word_lengths) if word_lengths else 0,
            "mean": sum(word_lengths) / len(word_lengths) if word_lengths else 0,
        },
    }


def check_pair_leakage(
    train_samples: List[Dict],
    valid_samples: List[Dict],
    test_samples: List[Dict],
) -> None:
    train_pairs = set(sample.get("pair_id") for sample in train_samples)
    valid_pairs = set(sample.get("pair_id") for sample in valid_samples)
    test_pairs = set(sample.get("pair_id") for sample in test_samples)

    train_valid_overlap = train_pairs & valid_pairs
    train_test_overlap = train_pairs & test_pairs
    valid_test_overlap = valid_pairs & test_pairs

    if train_valid_overlap or train_test_overlap or valid_test_overlap:
        raise ValueError(
            "Pair leakage detected!\n"
            f"train-valid overlap: {len(train_valid_overlap)}\n"
            f"train-test overlap: {len(train_test_overlap)}\n"
            f"valid-test overlap: {len(valid_test_overlap)}"
        )

    print("\nLeakage check passed: no pair_id overlap across splits.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split full_dataset.jsonl by pair_id to avoid data leakage."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Input full dataset JSONL path.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save train / valid / internal_test JSONL files.",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )

    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Internal test split ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible split.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help=(
            "Optional filename prefix. "
            "For example, --prefix deepseek_only_ creates deepseek_only_train.jsonl."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    samples = load_jsonl(input_path)

    print("=" * 70)
    print("Split Dataset by pair_id")
    print("=" * 70)
    print(f"Input path: {input_path}")
    print(f"Loaded samples: {len(samples)}")

    pair_groups = group_by_pair_id(samples)

    print(f"Unique pair_ids: {len(pair_groups)}")

    pair_signature_counter = Counter(
        get_pair_label_signature(group)
        for group in pair_groups.values()
    )

    print("Pair signature distribution:")
    for signature, count in pair_signature_counter.items():
        print(f"  {signature}: {count}")

    train_pair_ids, valid_pair_ids, test_pair_ids = split_pair_ids(
        pair_groups=pair_groups,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_samples = collect_samples(train_pair_ids, pair_groups, "train")
    valid_samples = collect_samples(valid_pair_ids, pair_groups, "valid")
    test_samples = collect_samples(test_pair_ids, pair_groups, "internal_test")

    check_pair_leakage(train_samples, valid_samples, test_samples)

    prefix = args.prefix

    train_path = output_dir / f"{prefix}train.jsonl"
    valid_path = output_dir / f"{prefix}valid.jsonl"
    test_path = output_dir / f"{prefix}internal_test.jsonl"
    report_path = output_dir / f"{prefix}split_report.json"

    save_jsonl(train_samples, train_path)
    save_jsonl(valid_samples, valid_path)
    save_jsonl(test_samples, test_path)

    report = {
        "input_path": str(input_path),
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "total_samples": len(samples),
        "total_pair_ids": len(pair_groups),
        "pair_signature_distribution": dict(pair_signature_counter),
        "train": summarize_samples(train_samples),
        "valid": summarize_samples(valid_samples),
        "internal_test": summarize_samples(test_samples),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    print(f"  train:         {train_path}")
    print(f"  valid:         {valid_path}")
    print(f"  internal_test: {test_path}")
    print(f"  report:        {report_path}")

    print("\nSplit summary:")
    print("Train:", report["train"]["num_samples"], report["train"]["label_distribution"])
    print("Valid:", report["valid"]["num_samples"], report["valid"]["label_distribution"])
    print("Internal test:", report["internal_test"]["num_samples"], report["internal_test"]["label_distribution"])


if __name__ == "__main__":
    main()