import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "round5_flip_guard_train.jsonl"
DEFAULT_DEV_HARDPOS = PROJECT_ROOT / "data" / "processed" / "round5_flip_guard_dev_hardpos.jsonl"
DEFAULT_DEV_HARDNEG = PROJECT_ROOT / "data" / "processed" / "round5_flip_guard_dev_hardneg.jsonl"
DEFAULT_LEDGER = PROJECT_ROOT / "outputs" / "evaluation" / "round5_flip_ledger.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round5_flip_guard"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_REPORT_MD = PROJECT_ROOT / "outputs" / "evaluation" / "round5_flip_guard_report.md"


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def target_label(row: Dict) -> Optional[int]:
    for key in ["flip_guard_label", "target"]:
        if row.get(key) in [0, 1]:
            return int(row[key])
    safety = row.get("override_safety")
    if safety == "unsafe_override":
        return 1
    if safety == "safe_override":
        return 0
    if row.get("original_detection_label") is not None and row.get("label") in [0, 1]:
        return int(row["label"])
    return None


def load_labeled_rows(path: Path) -> List[Dict]:
    rows = []
    for row in load_records(path):
        text = str(row.get("text", "")).strip()
        label = target_label(row)
        if not text or label not in [0, 1]:
            continue
        item = dict(row)
        item["flip_guard_label"] = int(label)
        rows.append(item)
    return rows


def load_prediction_rows(path: Path, split: str = "") -> List[Dict]:
    rows = []
    for row in load_records(path):
        if split and row.get("split") != split:
            continue
        if not str(row.get("text", "")).strip():
            continue
        item = dict(row)
        label = target_label(item)
        if label in [0, 1]:
            item["flip_guard_label"] = int(label)
        rows.append(item)
    return rows


def lexical_shape_features(text: str) -> Dict[str, float]:
    text = str(text or "")
    chars = max(1, len(text))
    letters = [char for char in text if char.isalpha()]
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    lower_words = [word.lower() for word in words]
    long_words = [word for word in words if len(word) >= 9]
    return {
        "uppercase_char_ratio": sum(1 for char in letters if char.isupper()) / max(1, len(letters)),
        "digit_char_ratio": sum(1 for char in text if char.isdigit()) / chars,
        "space_char_ratio": sum(1 for char in text if char.isspace()) / chars,
        "comma_count": float(text.count(",")),
        "colon_count": float(text.count(":")),
        "paren_count": float(text.count("(") + text.count(")")),
        "long_word_ratio": len(long_words) / max(1, len(words)),
        "avg_word_len": sum(len(word) for word in words) / max(1, len(words)),
        "first_person_count": float(sum(1 for word in lower_words if word in {"i", "me", "my", "mine", "we", "our"})),
        "quote_count_raw": float(sum(text.count(mark) for mark in ["'", '"', "`", "\u2018", "\u2019", "\u201c", "\u201d"])),
    }


def numeric_value(row: Dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def feature_dict(row: Dict) -> Dict:
    text = str(row.get("text", ""))
    bucket = row.get("bucket") or assign_bucket(text)
    features = text_features(text)
    features.pop("bucket", None)
    features.update(lexical_shape_features(text))
    features.update(
        {
            "bucket": str(bucket),
            "round4_bucket": str(row.get("round4_bucket") or bucket),
            "round4_tag": str(row.get("round4_tag") or "unknown"),
            "split": str(row.get("split") or "unknown"),
            "domain": str(row.get("domain") or "unknown"),
            "generator": str(row.get("generator") or "unknown"),
            "step7_prob": numeric_value(row, "step7_prob"),
            "round4_prob": numeric_value(row, "round4_prob"),
            "prob_delta": numeric_value(row, "prob_delta"),
            "guard_p_human_style": numeric_value(row, "guard_p_human_style"),
            "guard_human_style_veto": numeric_value(row, "guard_human_style_veto"),
            "step7_pred": numeric_value(row, "step7_pred"),
            "round4_pred": numeric_value(row, "round4_pred"),
            "word_count": numeric_value(row, "word_count"),
        }
    )
    return features


def labels_for(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([int(row["flip_guard_label"]) for row in rows], dtype=int)


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
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        max_features=word_max_features,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=1,
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
    labels = [target_label(row) for row in rows]
    labeled_indices = [idx for idx, label in enumerate(labels) if label in [0, 1]]
    preds_all = (probs >= threshold).astype(int)
    veto_rate_all = float(np.mean(preds_all == 1)) if len(preds_all) else 0.0
    out = {
        "num_samples": len(rows),
        "num_labeled": len(labeled_indices),
        "threshold": float(threshold),
        "veto_rate_all_rows": veto_rate_all,
        "mean_p_unsafe_override": float(np.mean(probs)) if len(probs) else 0.0,
    }
    if not labeled_indices:
        out.update(
            {
                "accuracy": None,
                "unsafe_precision": None,
                "unsafe_recall_protection": None,
                "unsafe_f1": None,
                "safe_veto_rate": None,
                "unsafe_miss_rate": None,
                "confusion_matrix": [[0, 0], [0, 0]],
            }
        )
        return out

    y = np.array([int(labels[idx]) for idx in labeled_indices], dtype=int)
    preds = np.array([int(preds_all[idx]) for idx in labeled_indices], dtype=int)
    safe_pass = int(np.sum((y == 0) & (preds == 0)))
    safe_veto = int(np.sum((y == 0) & (preds == 1)))
    unsafe_miss = int(np.sum((y == 1) & (preds == 0)))
    unsafe_veto = int(np.sum((y == 1) & (preds == 1)))
    accuracy = (safe_pass + unsafe_veto) / len(y) if len(y) else 0.0
    precision = unsafe_veto / (unsafe_veto + safe_veto) if unsafe_veto + safe_veto else 0.0
    recall = unsafe_veto / (unsafe_veto + unsafe_miss) if unsafe_veto + unsafe_miss else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    safe_veto_rate = safe_veto / (safe_pass + safe_veto) if safe_pass + safe_veto else 0.0
    unsafe_miss_rate = unsafe_miss / (unsafe_miss + unsafe_veto) if unsafe_miss + unsafe_veto else 0.0
    out.update(
        {
            "accuracy": float(accuracy),
            "unsafe_precision": float(precision),
            "unsafe_recall_protection": float(recall),
            "unsafe_f1": float(f1),
            "safe_veto_rate": float(safe_veto_rate),
            "unsafe_miss_rate": float(unsafe_miss_rate),
            "confusion_matrix": [[safe_pass, safe_veto], [unsafe_miss, unsafe_veto]],
            "label_counts": dict(Counter(str(label) for label in y.tolist())),
        }
    )
    return out


def choose_train_threshold(rows: Sequence[Dict], probs: np.ndarray, max_safe_veto_rate: float) -> Dict:
    scored = []
    for threshold in threshold_grid():
        block = metrics_for(rows, probs, threshold)
        safe_veto_rate = block["safe_veto_rate"] if block["safe_veto_rate"] is not None else 0.0
        unsafe_recall = block["unsafe_recall_protection"] if block["unsafe_recall_protection"] is not None else 0.0
        unsafe_precision = block["unsafe_precision"] if block["unsafe_precision"] is not None else 0.0
        feasible = safe_veto_rate <= max_safe_veto_rate
        score = (1 if feasible else 0, unsafe_recall, unsafe_precision, -safe_veto_rate, threshold)
        scored.append((score, block))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[0][1]
    selected["selected_by"] = f"max unsafe recall with train safe_veto_rate <= {max_safe_veto_rate:.2f}"
    return selected


def choose_gate_threshold(
    dev_hardpos: Sequence[Dict],
    dev_hardpos_probs: np.ndarray,
    dev_hardneg: Sequence[Dict],
    dev_hardneg_probs: np.ndarray,
    internal_candidates: Sequence[Dict],
    internal_candidate_probs: np.ndarray,
    internal_total_rows: int,
    max_safe_veto_rate: float,
    max_internal_candidate_veto_rate: float,
) -> Dict:
    scored = []
    for threshold in threshold_grid():
        hardpos = metrics_for(dev_hardpos, dev_hardpos_probs, threshold)
        hardneg = metrics_for(dev_hardneg, dev_hardneg_probs, threshold)
        internal_candidate_veto_rate = (
            float(np.mean(internal_candidate_probs >= threshold)) if len(internal_candidate_probs) else 0.0
        )
        internal_candidate_veto_rate_as_total = (
            float(np.sum(internal_candidate_probs >= threshold)) / max(1, internal_total_rows)
        )
        safe_veto = hardpos["safe_veto_rate"] if hardpos["safe_veto_rate"] is not None else 0.0
        hardneg_recall = hardneg["unsafe_recall_protection"] if hardneg["unsafe_recall_protection"] is not None else 0.0
        feasible = (
            safe_veto <= max_safe_veto_rate
            and internal_candidate_veto_rate_as_total <= max_internal_candidate_veto_rate
        )
        score = (
            1 if feasible else 0,
            hardneg_recall,
            -safe_veto,
            -internal_candidate_veto_rate_as_total,
            threshold,
        )
        scored.append(
            (
                score,
                {
                    "threshold": float(threshold),
                    "selected_by": (
                        "max hardneg unsafe protection with dev hardpos safe-veto and "
                        "internal candidate-veto constraints"
                    ),
                    "dev_hardpos_safe_veto_rate": safe_veto,
                    "dev_hardneg_unsafe_recall": hardneg_recall,
                    "internal_candidate_veto_rate": internal_candidate_veto_rate,
                    "internal_candidate_veto_rate_as_total": internal_candidate_veto_rate_as_total,
                    "constraints_passed": bool(feasible),
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def prediction_rows(rows: Sequence[Dict], probs: np.ndarray, threshold: float) -> List[Dict]:
    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["p_unsafe_override"] = float(prob)
        item["flip_guard_veto"] = int(prob >= threshold)
        item["flip_guard_threshold"] = float(threshold)
        label = target_label(row)
        if label in [0, 1]:
            item["flip_guard_label"] = int(label)
        out.append(item)
    return out


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round5 Flip Guard Report",
        "",
        "Positive label means unsafe override: Round4 wants to flip Step7 human -> LLM, but the row is human.",
        "",
        f"Selected threshold: `{report['threshold']:.4f}`",
        "",
        "## Metrics",
        "",
        "| Split | n | labeled | Veto rate all rows | Unsafe precision | Unsafe protection recall | Safe veto rate | Unsafe miss rate | Confusion [[safe pass, safe veto], [unsafe miss, unsafe veto]] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, block in report["metrics"].items():
        def fmt(value):
            return "NA" if value is None else f"{value:.4f}"

        lines.append(
            f"| {name} | {block['num_samples']} | {block['num_labeled']} | "
            f"{fmt(block['veto_rate_all_rows'])} | {fmt(block['unsafe_precision'])} | "
            f"{fmt(block['unsafe_recall_protection'])} | {fmt(block['safe_veto_rate'])} | "
            f"{fmt(block['unsafe_miss_rate'])} | {block['confusion_matrix']} |"
        )
    lines.extend(["", "## Gate Check", ""])
    lines.append("| Gate | Required | Observed | Pass |")
    lines.append("| --- | --- | ---: | --- |")
    for item in report["gate_check"]:
        lines.append(f"| {item['gate']} | {item['required']} | {item['observed']} | {item['pass']} |")
    lines.extend(["", "## Decision", "", report["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def gate_check(metrics: Dict) -> List[Dict]:
    hardneg = metrics.get("dev_hardneg_unsafe", {})
    hardpos = metrics.get("dev_hardpos_safe", {})
    internal_candidates = metrics.get("internal_test_override_candidates", {})
    hardneg_recall = hardneg.get("unsafe_recall_protection") or 0.0
    hardpos_safe_veto = hardpos.get("safe_veto_rate") or 0.0
    internal_veto = internal_candidates.get("veto_rate_as_total_internal_rows") or 0.0
    checks = [
        {
            "gate": "hardneg induced_fp protection",
            "required": ">= 0.70",
            "observed": f"{hardneg_recall:.4f}",
            "pass": hardneg_recall >= 0.70,
        },
        {
            "gate": "hardpos safe_override veto",
            "required": "<= 0.10",
            "observed": f"{hardpos_safe_veto:.4f}",
            "pass": hardpos_safe_veto <= 0.10,
        },
        {
            "gate": "internal_test candidate-veto rate",
            "required": "<= 0.03",
            "observed": f"{internal_veto:.4f}",
            "pass": internal_veto <= 0.03,
        },
    ]
    return checks


def parse_args():
    parser = argparse.ArgumentParser(description="Train Round5 FP-safe flip guard.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev_hardpos", default=str(DEFAULT_DEV_HARDPOS))
    parser.add_argument("--dev_hardneg", default=str(DEFAULT_DEV_HARDNEG))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction_dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--report_md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--max_safe_veto_rate", type=float, default=0.10)
    parser.add_argument("--max_internal_candidate_veto_rate", type=float, default=0.03)
    parser.add_argument("--threshold_selection", choices=["gate_dev", "train"], default="gate_dev")
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--word_max_features", type=int, default=20000)
    parser.add_argument("--char_max_features", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_labeled_rows(Path(args.train))
    dev_hardpos = load_labeled_rows(Path(args.dev_hardpos))
    dev_hardneg = load_labeled_rows(Path(args.dev_hardneg))
    internal_all = load_prediction_rows(Path(args.ledger), split="internal_test")
    internal_candidates = [row for row in internal_all if row.get("round4_override_candidate")]
    if not train_rows:
        raise ValueError("No Round5 flip-guard train rows found.")
    if len(set(labels_for(train_rows).tolist())) < 2:
        raise ValueError("Round5 flip-guard train rows must contain both safe and unsafe examples.")

    eval_blocks = [train_rows, dev_hardpos, dev_hardneg, internal_candidates, internal_all]
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
    hardpos_probs = model.predict_proba(x_eval[1])[:, 1]
    hardneg_probs = model.predict_proba(x_eval[2])[:, 1]
    internal_candidate_probs = model.predict_proba(x_eval[3])[:, 1]
    if args.threshold_selection == "gate_dev":
        selected_threshold = choose_gate_threshold(
            dev_hardpos=dev_hardpos,
            dev_hardpos_probs=hardpos_probs,
            dev_hardneg=dev_hardneg,
            dev_hardneg_probs=hardneg_probs,
            internal_candidates=internal_candidates,
            internal_candidate_probs=internal_candidate_probs,
            internal_total_rows=len(internal_all),
            max_safe_veto_rate=args.max_safe_veto_rate,
            max_internal_candidate_veto_rate=args.max_internal_candidate_veto_rate,
        )
    else:
        selected_threshold = choose_train_threshold(train_rows, train_probs, args.max_safe_veto_rate)
    threshold = float(selected_threshold["threshold"])

    blocks = {
        "train_internal_override_candidates": (train_rows, x_eval[0]),
        "dev_hardpos_safe": (dev_hardpos, x_eval[1]),
        "dev_hardneg_unsafe": (dev_hardneg, x_eval[2]),
        "internal_test_override_candidates": (internal_candidates, x_eval[3]),
        "internal_test_all": (internal_all, x_eval[4]),
    }
    metrics = {}
    prediction_outputs = {}
    for name, (rows, x_block) in blocks.items():
        probs = model.predict_proba(x_block)[:, 1]
        metrics[name] = metrics_for(rows, probs, threshold=threshold)
        if name == "internal_test_override_candidates":
            metrics[name]["veto_rate_as_total_internal_rows"] = (
                float(np.sum(probs >= threshold)) / max(1, len(internal_all))
            )
        out_rows = prediction_rows(rows, probs, threshold)
        out_path = prediction_dir / f"round5_flip_guard_{name}_predictions.jsonl"
        save_jsonl(out_rows, out_path)
        prediction_outputs[name] = str(out_path)

    artifact = {
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "dict_vectorizer": dict_vectorizer,
        "threshold": threshold,
        "feature_version": "round5_flip_guard_v1",
        "positive_label": "unsafe_override",
    }
    model_path = output_dir / "flip_guard.pkl"
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    checks = gate_check(metrics)
    passed = all(item["pass"] for item in checks)
    decision = (
        "PROMOTE_TO_PHASE5_OVERRIDE_SEARCH = yes"
        if passed
        else "PROMOTE_TO_PHASE5_OVERRIDE_SEARCH = no; use this report to guide Phase 2 residual data augmentation or a more constrained guard."
    )
    report = {
        "model_path": str(model_path),
        "threshold": threshold,
        "threshold_selection": selected_threshold,
        "train": args.train,
        "dev_hardpos": args.dev_hardpos,
        "dev_hardneg": args.dev_hardneg,
        "ledger": args.ledger,
        "train_label_counts": dict(Counter(str(value) for value in y_train.tolist())),
        "metrics": metrics,
        "prediction_outputs": prediction_outputs,
        "gate_check": checks,
        "decision": decision,
        "config": vars(args),
    }
    write_json(report, output_dir / "flip_guard_report.json")
    write_markdown(report, Path(args.report_md))

    print("=" * 70)
    print("Round5 flip guard trained")
    print("=" * 70)
    print(f"Train rows: {len(train_rows)} labels={report['train_label_counts']}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Model: {model_path}")
    for name, block in metrics.items():
        print(
            f"{name}: veto={block['veto_rate_all_rows']:.4f} "
            f"unsafe_recall={block['unsafe_recall_protection']} "
            f"safe_veto={block['safe_veto_rate']} "
            f"confusion={block['confusion_matrix']}"
        )
    print(decision)


if __name__ == "__main__":
    main()
