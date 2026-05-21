import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_extra(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Extra branch spec must be FIELD=PATH, got: {value}")
    field, path = value.split("=", 1)
    return field.strip(), Path(path.strip())


def row_id(row: Dict, index: int) -> str:
    return str(row.get("id", index))


def parse_args():
    parser = argparse.ArgumentParser(description="Merge branch probabilities into a base prediction JSONL.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", nargs="+", required=True, help="FIELD=PATH specs, e.g. p_roberta=preds.jsonl")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    base_rows = load_jsonl(Path(args.base))
    merged = [dict(row) for row in base_rows]
    base_ids = [row_id(row, index) for index, row in enumerate(base_rows)]

    for field, path in [parse_extra(value) for value in args.extra]:
        extra_rows = load_jsonl(path)
        extra_by_id = {
            row_id(row, index): float(row.get("probability", row.get("prob_llm", row.get(field, 0.0))))
            for index, row in enumerate(extra_rows)
        }
        missing = 0
        for item, sample_id in zip(merged, base_ids):
            if sample_id not in extra_by_id:
                missing += 1
                item[field] = 0.0
            else:
                item[field] = extra_by_id[sample_id]
        if missing:
            print(f"[Warning] Missing {missing} ids for {field} from {path}")

    save_jsonl(merged, Path(args.output))
    print("=" * 70)
    print("Prediction branches merged")
    print("=" * 70)
    print(f"Rows: {len(merged)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
