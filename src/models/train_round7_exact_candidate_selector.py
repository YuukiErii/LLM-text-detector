import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "round7_exact_candidate_train.jsonl"
DEFAULT_DEV = PROJECT_ROOT / "data" / "processed" / "round7_exact_candidate_dev.jsonl"
DEFAULT_PROBE = PROJECT_ROOT / "data" / "processed" / "round6_override_probe_mixed.jsonl"
DEFAULT_ROUND6_PROBE = PROJECT_ROOT / "outputs" / "predictions" / "round6_safe_selector_probe_mixed_predictions.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round7_exact_candidate_selector"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round7_exact_candidate_selector_report.md"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def selector_label(row: Dict) -> int:
    for key in ["round7_selector_label", "safe_selector_label", "target", "label"]:
        value = row.get(key)
        if value in [0, 1]:
            return int(value)
    safety = row.get("override_safety")
    if safety == "safe_override":
        return 1
    if safety == "unsafe_override":
        return 0
    raise ValueError("Missing exact-candidate selector label.")


def load_labeled_rows(path: Path) -> List[Dict]:
    rows = []
    for row in load_records(path):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        item = dict(row)
        item["round7_selector_label"] = selector_label(item)
        rows.append(item)
    return rows


def numeric(row: Dict, key: str) -> float:
    value = row.get(key)
    if value in [None, ""]:
        return 0.0
    try:
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
        "semicolon_count": float(text.count(";")),
        "paren_count": float(text.count("(") + text.count(")")),
        "quote_count": float(text.count('"') + text.count("'")),
        "long_word_ratio": len(long_words) / max(1, len(words)),
        "avg_word_len": sum(len(word) for word in words) / max(1, len(words)),
        "first_person_count": float(sum(1 for word in lower_words if word in {"i", "me", "my", "mine", "we", "our"})),
    }


def feature_dict(row: Dict) -> Dict:
    text = str(row.get("text") or "")
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
            "p_safe_override_v1b": numeric(row, "p_safe_override"),
            "has_step7_prob": has_numeric(row, "step7_prob"),
            "has_round4_prob": has_numeric(row, "round4_prob"),
            "has_guard_p_human_style": has_numeric(row, "guard_p_human_style"),
            "has_p_unsafe_override": has_numeric(row, "p_unsafe_override"),
            "has_p_safe_override_v1b": has_numeric(row, "p_safe_override"),
            "word_count": numeric(row, "word_count"),
        }
    )
    return features


def labels_for(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([selector_label(row) for row in rows], dtype=int)


def build_features(
    train_rows: Sequence[Dict],
    eval_blocks: Sequence[Sequence[Dict]],
    word_max_features: int,
    char_max_features: int,
):
    train_texts = [str(row.get("text") or "") for row in train_rows]
    eval_texts = [[str(row.get("text") or "") for row in rows] for rows in eval_blocks]
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
    x_train_char = char_vectorizer.fit_transform(train_texts)
    x_train_dict = dict_vectorizer.fit_transform([feature_dict(row) for row in train_rows])
    x_eval = []
    for rows, texts in zip(eval_blocks, eval_texts):
        x_eval.append(
            hstack(
                [
                    word_vectorizer.transform(texts),
                    char_vectorizer.transform(texts),
                    dict_vectorizer.transform([feature_dict(row) for row in rows]),
                ]
            )
        )
    return hstack([x_train_word, x_train_char, x_train_dict]), x_eval, word_vectorizer, char_vectorizer, dict_vectorizer


def threshold_grid() -> List[float]:
    return [round(float(value), 4) for value in np.linspace(0.05, 0.95, 91)]


def metrics_for(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> Dict:
    labels = labels_for(rows)
    preds = (probs >= threshold).astype(int)
    safe_total = int(np.sum(labels == 1))
    unsafe_total = int(np.sum(labels == 0))
    safe_passed = int(np.sum((labels == 1) & (preds == 1)))
    safe_blocked = int(np.sum((labels == 1) & (preds == 0)))
    unsafe_blocked = int(np.sum((labels == 0) & (preds == 0)))
    unsafe_passed = int(np.sum((labels == 0) & (preds == 1)))
    return {
        "num_samples": len(rows),
        "threshold": float(threshold),
        "mean_p_safe_override": float(np.mean(probs)) if len(probs) else 0.0,
        "safe_total": safe_total,
        "safe_passed": safe_passed,
        "safe_blocked": safe_blocked,
        "safe_pass_rate": safe_passed / safe_total if safe_total else None,
        "unsafe_total": unsafe_total,
        "unsafe_blocked": unsafe_blocked,
        "unsafe_passed": unsafe_passed,
        "unsafe_block_rate": unsafe_blocked / unsafe_total if unsafe_total else None,
        "accuracy": (safe_passed + unsafe_blocked) / len(rows) if rows else 0.0,
        "confusion_matrix": [[unsafe_blocked, unsafe_passed], [safe_blocked, safe_passed]],
        "label_counts": dict(Counter(str(value) for value in labels.tolist())),
    }


def prediction_rows(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> List[Dict]:
    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["p_round7_safe_override"] = float(prob)
        item["round7_exact_selector_pass"] = int(prob >= threshold)
        item["round7_exact_selector_prediction"] = int(prob >= threshold)
        item["round7_exact_selector_threshold"] = float(threshold)
        out.append(item)
    return out


def choose_threshold(dev_rows: Sequence[Dict], dev_probs: np.ndarray, min_unsafe_block_rate: float, min_safe_pass_rate: float) -> Dict:
    scored = []
    for threshold in threshold_grid():
        metrics = metrics_for(dev_rows, dev_probs, threshold)
        safe_pass = metrics["safe_pass_rate"] or 0.0
        unsafe_block = metrics["unsafe_block_rate"] or 0.0
        unsafe_feasible = unsafe_block >= min_unsafe_block_rate
        fully_feasible = unsafe_feasible and safe_pass >= min_safe_pass_rate
        score = (1 if fully_feasible else 0, 1 if unsafe_feasible else 0, safe_pass, unsafe_block, -threshold)
        scored.append(
            (
                score,
                {
                    "threshold": float(threshold),
                    "selected_by": "exact dev threshold only: maximize safe pass subject to unsafe block and safe-pass gates",
                    "dev_constraints_passed": bool(fully_feasible),
                    "dev_safe_pass_rate": float(safe_pass),
                    "dev_unsafe_block_rate": float(unsafe_block),
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def baseline_probe_metrics(rows: Sequence[Dict]) -> Dict:
    labels = labels_for(rows)
    predictions = []
    for row in rows:
        value = row.get("safe_selector_pass")
        if value in [None, ""]:
            value = row.get("safe_selector_prediction")
        predictions.append(int(value or 0))
    predictions = np.array(predictions, dtype=int)
    unsafe_total = int(np.sum(labels == 0))
    unsafe_blocked = int(np.sum((labels == 0) & (predictions == 0)))
    safe_total = int(np.sum(labels == 1))
    safe_passed = int(np.sum((labels == 1) & (predictions == 1)))
    return {
        "num_samples": len(rows),
        "safe_total": safe_total,
        "safe_passed": safe_passed,
        "safe_pass_rate": safe_passed / safe_total if safe_total else None,
        "unsafe_total": unsafe_total,
        "unsafe_blocked": unsafe_blocked,
        "unsafe_block_rate": unsafe_blocked / unsafe_total if unsafe_total else None,
    }


def gate_check(dev_metrics: Dict, probe_metrics: Dict, baseline_probe: Dict, args) -> List[Dict]:
    baseline_rate = baseline_probe["unsafe_block_rate"] or 0.0
    probe_rate = probe_metrics["unsafe_block_rate"] or 0.0
    probe_rate_target = min(1.0, baseline_rate + args.min_probe_unsafe_block_rate_gain)
    probe_count_target = math.ceil(probe_rate_target * probe_metrics["unsafe_total"])
    return [
        {
            "gate": "exact dev unsafe block",
            "required": f">= {args.min_dev_unsafe_block_rate:.2f}",
            "observed": f"{dev_metrics['unsafe_block_rate'] or 0.0:.4f}",
            "pass": (dev_metrics["unsafe_block_rate"] or 0.0) >= args.min_dev_unsafe_block_rate,
        },
        {
            "gate": "exact dev safe pass",
            "required": f">= {args.min_dev_safe_pass_rate:.2f}",
            "observed": f"{dev_metrics['safe_pass_rate'] or 0.0:.4f}",
            "pass": (dev_metrics["safe_pass_rate"] or 0.0) >= args.min_dev_safe_pass_rate,
        },
        {
            "gate": "held-out probe unsafe block gain vs Round6",
            "required": f">= {probe_rate_target:.4f} rate and >= {probe_count_target} blocked",
            "observed": f"{probe_rate:.4f} rate and {probe_metrics['unsafe_blocked']} blocked",
            "pass": probe_rate >= probe_rate_target and probe_metrics["unsafe_blocked"] >= probe_count_target,
        },
    ]


def render_rate(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round7 Exact Candidate Selector Report",
        "",
        "This first Round7 selector is a transparent LogisticRegression baseline trained on exact-candidate rows.",
        "Threshold selection uses exact dev only; the held-out internal-style probe is a gate check against Round6.",
        "",
        f"Selected threshold: `{report['threshold']:.4f}`",
        "",
        "## Metrics",
        "",
        "| Split | n | mean p_safe | safe pass | unsafe block | accuracy | confusion [[unsafe block, unsafe pass], [safe block, safe pass]] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, block in report["metrics"].items():
        lines.append(
            f"| {name} | {block['num_samples']} | {block['mean_p_safe_override']:.4f} | "
            f"{render_rate(block['safe_pass_rate'])} | {render_rate(block['unsafe_block_rate'])} | "
            f"{block['accuracy']:.4f} | {block['confusion_matrix']} |"
        )
    baseline = report["round6_probe_baseline"]
    lines.extend(
        [
            "",
            "## Probe Baseline",
            "",
            "| Selector | Safe pass | Unsafe blocked | Unsafe block rate |",
            "| --- | ---: | ---: | ---: |",
            f"| Round6 v1b | {baseline['safe_passed']} / {baseline['safe_total']} | "
            f"{baseline['unsafe_blocked']} / {baseline['unsafe_total']} | {render_rate(baseline['unsafe_block_rate'])} |",
            f"| Round7 baseline | {report['metrics']['probe_mixed']['safe_passed']} / {report['metrics']['probe_mixed']['safe_total']} | "
            f"{report['metrics']['probe_mixed']['unsafe_blocked']} / {report['metrics']['probe_mixed']['unsafe_total']} | "
            f"{render_rate(report['metrics']['probe_mixed']['unsafe_block_rate'])} |",
            "",
            "## Gate Check",
            "",
            "| Gate | Required | Observed | Pass |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in report["gate_check"]:
        lines.append(f"| {item['gate']} | {item['required']} | {item['observed']} | {item['pass']} |")
    lines.extend(["", "## Decision", "", "```text", report["decision"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Round7 exact-candidate safe selector baseline.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev", default=str(DEFAULT_DEV))
    parser.add_argument("--probe_mixed", default=str(DEFAULT_PROBE))
    parser.add_argument("--round6_probe_predictions", default=str(DEFAULT_ROUND6_PROBE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction_dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--min_dev_unsafe_block_rate", type=float, default=0.90)
    parser.add_argument("--min_dev_safe_pass_rate", type=float, default=0.35)
    parser.add_argument("--min_probe_unsafe_block_rate_gain", type=float, default=0.20)
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
    dev_rows = load_labeled_rows(Path(args.dev))
    probe_rows = load_labeled_rows(Path(args.probe_mixed))
    round6_probe_rows = load_labeled_rows(Path(args.round6_probe_predictions))
    if len(set(labels_for(train_rows).tolist())) < 2:
        raise ValueError("Round7 exact selector train rows must contain both safe and unsafe labels.")

    eval_blocks = [train_rows, dev_rows, probe_rows]
    x_train, x_eval, word_vectorizer, char_vectorizer, dict_vectorizer = build_features(
        train_rows,
        eval_blocks,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
    )
    model = LogisticRegression(
        C=args.C,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=args.seed,
    )
    model.fit(x_train, labels_for(train_rows))
    probs = [model.predict_proba(block)[:, 1] for block in x_eval]
    selected = choose_threshold(
        dev_rows,
        probs[1],
        min_unsafe_block_rate=args.min_dev_unsafe_block_rate,
        min_safe_pass_rate=args.min_dev_safe_pass_rate,
    )
    threshold = float(selected["threshold"])

    blocks = {
        "train": (train_rows, probs[0]),
        "dev": (dev_rows, probs[1]),
        "probe_mixed": (probe_rows, probs[2]),
    }
    metrics = {}
    prediction_outputs = {}
    for name, (rows, block_probs) in blocks.items():
        metrics[name] = metrics_for(rows, block_probs, threshold)
        output_path = prediction_dir / f"round7_exact_selector_{name}_predictions.jsonl"
        save_jsonl(prediction_rows(rows, block_probs, threshold), output_path)
        prediction_outputs[name] = str(output_path)

    round6_baseline = baseline_probe_metrics(round6_probe_rows)
    checks = gate_check(metrics["dev"], metrics["probe_mixed"], round6_baseline, args)
    decision = (
        "PROMOTE_TO_ROUND7_RULE_SEARCH = yes"
        if all(item["pass"] for item in checks)
        else "PROMOTE_TO_ROUND7_RULE_SEARCH = no; improve exact unsafe coverage or selector features before rule search."
    )
    artifact = {
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "dict_vectorizer": dict_vectorizer,
        "threshold": threshold,
        "feature_version": "round7_exact_candidate_selector_logreg_v1",
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
        "dev": args.dev,
        "probe_mixed": args.probe_mixed,
        "round6_probe_predictions": args.round6_probe_predictions,
        "train_label_counts": dict(Counter(str(value) for value in labels_for(train_rows).tolist())),
        "metrics": metrics,
        "round6_probe_baseline": round6_baseline,
        "prediction_outputs": prediction_outputs,
        "gate_check": checks,
        "decision": decision,
        "config": vars(args),
    }
    write_json(report, output_dir / "selector_report.json")
    write_markdown(report, Path(args.report_md))
    print("=" * 70)
    print("Round7 exact-candidate selector baseline trained")
    print("=" * 70)
    print(f"Train rows: {len(train_rows)} labels={report['train_label_counts']}")
    print(f"Dev threshold: {threshold:.4f}")
    for name, block in metrics.items():
        print(
            f"{name}: safe_pass={block['safe_pass_rate']} "
            f"unsafe_block={block['unsafe_block_rate']} confusion={block['confusion_matrix']}"
        )
    print(decision)


if __name__ == "__main__":
    main()
