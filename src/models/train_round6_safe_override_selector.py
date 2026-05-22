import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "round6_override_train.jsonl"
DEFAULT_DEV_SAFE = PROJECT_ROOT / "data" / "processed" / "round6_override_dev_safe.jsonl"
DEFAULT_DEV_UNSAFE = PROJECT_ROOT / "data" / "processed" / "round6_override_dev_unsafe.jsonl"
DEFAULT_PROBE_MIXED = PROJECT_ROOT / "data" / "processed" / "round6_override_probe_mixed.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round6_safe_override_selector"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round6_safe_override_selector_report.md"


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_label(row: Dict) -> int:
    for key in ["safe_selector_label", "target", "label"]:
        if row.get(key) in [0, 1]:
            return int(row[key])
    safety = row.get("override_safety")
    if safety == "safe_override":
        return 1
    if safety == "unsafe_override":
        return 0
    raise ValueError("Missing safe selector label.")


def load_labeled_rows(path: Path) -> List[Dict]:
    rows = []
    for row in load_records(path):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        item = dict(row)
        item["safe_selector_label"] = safe_label(item)
        rows.append(item)
    return rows


def numeric(row: Dict, key: str) -> float:
    try:
        value = row.get(key)
        if value in [None, ""]:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def has_numeric(row: Dict, key: str) -> float:
    return 0.0 if row.get(key) in [None, ""] else 1.0


def lexical_shape_features(text: str) -> Dict[str, float]:
    text = str(text or "")
    chars = max(1, len(text))
    words = WORD_RE.findall(text)
    lower_words = [word.lower() for word in words]
    long_words = [word for word in words if len(word) >= 9]
    return {
        "uppercase_char_ratio": sum(1 for char in text if char.isalpha() and char.isupper()) / chars,
        "digit_char_ratio": sum(1 for char in text if char.isdigit()) / chars,
        "space_char_ratio": sum(1 for char in text if char.isspace()) / chars,
        "comma_count": float(text.count(",")),
        "colon_count": float(text.count(":")),
        "paren_count": float(text.count("(") + text.count(")")),
        "long_word_ratio": len(long_words) / max(1, len(words)),
        "avg_word_len": sum(len(word) for word in words) / max(1, len(words)),
        "first_person_count": float(sum(1 for word in lower_words if word in {"i", "me", "my", "mine", "we", "our"})),
    }


def feature_dict(row: Dict) -> Dict:
    text = str(row.get("text", ""))
    bucket = str(row.get("bucket") or assign_bucket(text))
    round4_bucket = str(row.get("round4_bucket") or bucket)
    features = text_features(text)
    features.pop("bucket", None)
    features.update(lexical_shape_features(text))
    features.update(
        {
            "bucket": bucket,
            "round4_bucket": round4_bucket,
            "domain": str(row.get("domain") or "unknown"),
            "step7_prob": numeric(row, "step7_prob"),
            "round4_prob": numeric(row, "round4_prob"),
            "prob_delta": numeric(row, "prob_delta"),
            "guard_p_human_style": numeric(row, "guard_p_human_style"),
            "p_unsafe_override": numeric(row, "p_unsafe_override"),
            "has_step7_prob": has_numeric(row, "step7_prob"),
            "has_round4_prob": has_numeric(row, "round4_prob"),
            "has_guard_p_human_style": has_numeric(row, "guard_p_human_style"),
            "word_count": numeric(row, "word_count"),
            "is_exact_override_candidate": float(bool(row.get("is_exact_override_candidate"))),
        }
    )
    return features


def labels_for(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([safe_label(row) for row in rows], dtype=int)


def build_features(
    train_rows: Sequence[Dict],
    eval_blocks: Sequence[Sequence[Dict]],
    word_max_features: int,
    char_max_features: int,
):
    train_texts = [str(row.get("text", "")) for row in train_rows]
    eval_texts = [[str(row.get("text", "")) for row in rows] for rows in eval_blocks]

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=word_max_features,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=char_max_features,
    )
    dict_vectorizer = DictVectorizer(sparse=True)

    x_train_word = word_vectorizer.fit_transform(train_texts)
    x_eval_word = [word_vectorizer.transform(texts) for texts in eval_texts]
    x_train_char = char_vectorizer.fit_transform(train_texts)
    x_eval_char = [char_vectorizer.transform(texts) for texts in eval_texts]
    x_train_dict = dict_vectorizer.fit_transform([feature_dict(row) for row in train_rows])
    x_eval_dict = [dict_vectorizer.transform([feature_dict(row) for row in rows]) for rows in eval_blocks]

    x_train = hstack([x_train_word, x_train_char, x_train_dict])
    x_eval = [
        hstack([x_word, x_char, x_dict])
        for x_word, x_char, x_dict in zip(x_eval_word, x_eval_char, x_eval_dict)
    ]
    return x_train, x_eval, word_vectorizer, char_vectorizer, dict_vectorizer


def threshold_grid() -> List[float]:
    return [round(float(value), 4) for value in np.linspace(0.05, 0.95, 91)]


def metrics_for(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> Dict:
    labels = labels_for(rows)
    pred_safe = (probs >= threshold).astype(int)
    safe_total = int(np.sum(labels == 1))
    unsafe_total = int(np.sum(labels == 0))
    safe_pass = int(np.sum((labels == 1) & (pred_safe == 1)))
    safe_block = int(np.sum((labels == 1) & (pred_safe == 0)))
    unsafe_pass = int(np.sum((labels == 0) & (pred_safe == 1)))
    unsafe_block = int(np.sum((labels == 0) & (pred_safe == 0)))
    return {
        "num_samples": len(rows),
        "threshold": float(threshold),
        "mean_p_safe_override": float(np.mean(probs)) if len(probs) else 0.0,
        "safe_pass_rate": safe_pass / safe_total if safe_total else None,
        "safe_block_rate": safe_block / safe_total if safe_total else None,
        "unsafe_block_rate": unsafe_block / unsafe_total if unsafe_total else None,
        "unsafe_pass_rate": unsafe_pass / unsafe_total if unsafe_total else None,
        "accuracy": (safe_pass + unsafe_block) / len(rows) if rows else 0.0,
        "confusion_matrix": [[unsafe_block, unsafe_pass], [safe_block, safe_pass]],
        "label_counts": dict(Counter(str(value) for value in labels.tolist())),
    }


def choose_threshold(dev_safe_rows, dev_safe_probs, dev_unsafe_rows, dev_unsafe_probs, min_unsafe_block_rate: float) -> Dict:
    scored = []
    for threshold in threshold_grid():
        safe_metrics = metrics_for(dev_safe_rows, dev_safe_probs, threshold)
        unsafe_metrics = metrics_for(dev_unsafe_rows, dev_unsafe_probs, threshold)
        safe_pass = safe_metrics["safe_pass_rate"] or 0.0
        unsafe_block = unsafe_metrics["unsafe_block_rate"] or 0.0
        feasible = unsafe_block >= min_unsafe_block_rate
        score = (1 if feasible else 0, safe_pass, unsafe_block, -threshold)
        scored.append(
            (
                score,
                {
                    "threshold": float(threshold),
                    "selected_by": "maximize safe dev pass rate subject to unsafe dev block rate",
                    "constraints_passed": bool(feasible),
                    "dev_safe_pass_rate": float(safe_pass),
                    "dev_unsafe_block_rate": float(unsafe_block),
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def prediction_rows(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> List[Dict]:
    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["p_safe_override"] = float(prob)
        item["safe_selector_pass"] = int(prob >= threshold)
        item["safe_selector_threshold"] = float(threshold)
        item["safe_selector_prediction"] = int(prob >= threshold)
        out.append(item)
    return out


def gate_check(metrics: Dict, min_unsafe_block_rate: float, min_safe_pass_rate: float) -> List[Dict]:
    unsafe_block = metrics["dev_unsafe"]["unsafe_block_rate"] or 0.0
    safe_pass = metrics["dev_safe"]["safe_pass_rate"] or 0.0
    return [
        {
            "gate": "unsafe dev blocked",
            "required": f">= {min_unsafe_block_rate:.2f}",
            "observed": f"{unsafe_block:.4f}",
            "pass": unsafe_block >= min_unsafe_block_rate,
        },
        {
            "gate": "safe dev pass rate",
            "required": f">= {min_safe_pass_rate:.2f}",
            "observed": f"{safe_pass:.4f}",
            "pass": safe_pass >= min_safe_pass_rate,
        },
    ]


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round6 Safe Override Selector Report",
        "",
        "Positive label means safe override: Step7-human -> Round4-LLM should be allowed.",
        "",
        f"Selected threshold: `{report['threshold']:.4f}`",
        "",
        "## Metrics",
        "",
        "| Split | n | mean p_safe | safe pass | unsafe block | accuracy | confusion [[unsafe block, unsafe pass], [safe block, safe pass]] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, block in report["metrics"].items():
        safe_pass = "NA" if block["safe_pass_rate"] is None else f"{block['safe_pass_rate']:.4f}"
        unsafe_block = "NA" if block["unsafe_block_rate"] is None else f"{block['unsafe_block_rate']:.4f}"
        lines.append(
            f"| {name} | {block['num_samples']} | {block['mean_p_safe_override']:.4f} | "
            f"{safe_pass} | {unsafe_block} | {block['accuracy']:.4f} | {block['confusion_matrix']} |"
        )

    lines.extend(["", "## Gate Check", "", "| Gate | Required | Observed | Pass |", "| --- | --- | ---: | --- |"])
    for item in report["gate_check"]:
        lines.append(f"| {item['gate']} | {item['required']} | {item['observed']} | {item['pass']} |")

    lines.extend(["", "## Decision", "", "```text", report["decision"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Round6 candidate-level safe override selector.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev_safe", default=str(DEFAULT_DEV_SAFE))
    parser.add_argument("--dev_unsafe", default=str(DEFAULT_DEV_UNSAFE))
    parser.add_argument("--probe_mixed", default=str(DEFAULT_PROBE_MIXED))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction_dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--min_unsafe_block_rate", type=float, default=0.90)
    parser.add_argument("--min_safe_pass_rate", type=float, default=0.25)
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--word_max_features", type=int, default=30000)
    parser.add_argument("--char_max_features", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_labeled_rows(Path(args.train))
    dev_safe = load_labeled_rows(Path(args.dev_safe))
    dev_unsafe = load_labeled_rows(Path(args.dev_unsafe))
    probe_mixed = load_labeled_rows(Path(args.probe_mixed))
    if len(set(labels_for(train_rows).tolist())) < 2:
        raise ValueError("Round6 selector train rows must contain both safe and unsafe labels.")

    eval_blocks = [train_rows, dev_safe, dev_unsafe, probe_mixed]
    x_train, x_eval, word_vectorizer, char_vectorizer, dict_vectorizer = build_features(
        train_rows,
        eval_blocks,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
    )
    y_train = labels_for(train_rows)
    model = LogisticRegression(
        C=args.C,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=args.seed,
    )
    model.fit(x_train, y_train)

    train_probs = model.predict_proba(x_eval[0])[:, 1]
    safe_probs = model.predict_proba(x_eval[1])[:, 1]
    unsafe_probs = model.predict_proba(x_eval[2])[:, 1]
    probe_probs = model.predict_proba(x_eval[3])[:, 1]
    selected = choose_threshold(
        dev_safe,
        safe_probs,
        dev_unsafe,
        unsafe_probs,
        min_unsafe_block_rate=args.min_unsafe_block_rate,
    )
    threshold = float(selected["threshold"])

    blocks = {
        "train": (train_rows, train_probs),
        "dev_safe": (dev_safe, safe_probs),
        "dev_unsafe": (dev_unsafe, unsafe_probs),
        "probe_mixed": (probe_mixed, probe_probs),
    }
    metrics = {}
    prediction_outputs = {}
    for name, (rows, probs) in blocks.items():
        metrics[name] = metrics_for(rows, probs, threshold)
        out_rows = prediction_rows(rows, probs, threshold)
        out_path = prediction_dir / f"round6_safe_selector_{name}_predictions.jsonl"
        save_jsonl(out_rows, out_path)
        prediction_outputs[name] = str(out_path)

    checks = gate_check(metrics, args.min_unsafe_block_rate, args.min_safe_pass_rate)
    decision = (
        "PROMOTE_TO_ROUND6_RULE_SEARCH = yes"
        if all(item["pass"] for item in checks)
        else "PROMOTE_TO_ROUND6_RULE_SEARCH = no; improve proxy data or selector calibration."
    )
    artifact = {
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "dict_vectorizer": dict_vectorizer,
        "threshold": threshold,
        "feature_version": "round6_safe_override_selector_v1",
        "positive_label": "safe_override",
    }
    model_path = output_dir / "selector.pkl"
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    report = {
        "model_path": str(model_path),
        "threshold": threshold,
        "threshold_selection": selected,
        "train": args.train,
        "dev_safe": args.dev_safe,
        "dev_unsafe": args.dev_unsafe,
        "probe_mixed": args.probe_mixed,
        "train_label_counts": dict(Counter(str(value) for value in y_train.tolist())),
        "metrics": metrics,
        "prediction_outputs": prediction_outputs,
        "gate_check": checks,
        "decision": decision,
        "config": vars(args),
    }
    write_json(report, output_dir / "selector_report.json")
    write_markdown(report, Path(args.report_md))

    print("=" * 70)
    print("Round6 safe override selector trained")
    print("=" * 70)
    print(f"Train rows: {len(train_rows)} labels={report['train_label_counts']}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Model: {model_path}")
    for name, block in metrics.items():
        print(
            f"{name}: safe_pass={block['safe_pass_rate']} "
            f"unsafe_block={block['unsafe_block_rate']} confusion={block['confusion_matrix']}"
        )
    print(decision)


if __name__ == "__main__":
    main()
