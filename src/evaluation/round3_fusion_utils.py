import itertools
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from evaluation.assign_text_bucket import assign_bucket, load_records, text_features


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def safe_name(value: str) -> str:
    name = SAFE_NAME_RE.sub("_", str(value).strip()).strip("_").lower()
    return name or "run"


def sort_key(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Run spec must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    return safe_name(name), Path(path.strip())


def probability(row: Dict) -> float:
    for key in ["probability", "prob_llm", "p_llm", "score", "p_round3_electra", "p_roberta"]:
        if row.get(key) is not None:
            return float(row[key])
    return float(row.get("prediction", 0))


def normalized_prediction_rows(path: Path) -> Dict[str, Dict]:
    rows = {}
    for index, row in enumerate(load_records(path)):
        if row.get("label") not in [0, 1] or row.get("prediction") not in [0, 1]:
            continue
        item = dict(row)
        item["id"] = str(item.get("id", index))
        item["label"] = int(item["label"])
        item["prediction"] = int(item["prediction"])
        item["probability"] = probability(item)
        rows[item["id"]] = item
    if not rows:
        raise ValueError(f"No labeled prediction rows found: {path}")
    return rows


def first_present(rows: Sequence[Dict], key: str, default=None):
    for row in rows:
        if row.get(key) is not None:
            return row.get(key)
    return default


def align_prediction_runs(run_specs: Sequence[str], split_name: str) -> Tuple[List[str], List[Dict]]:
    parsed = [parse_run(value) for value in run_specs]
    run_names = [name for name, _path in parsed]
    if len(set(run_names)) != len(run_names):
        raise ValueError(f"Duplicate run names in {split_name}: {run_names}")

    loaded = [(name, normalized_prediction_rows(path)) for name, path in parsed]
    base_ids = set(loaded[0][1])
    for name, rows in loaded[1:]:
        missing = base_ids - set(rows)
        extra = set(rows) - base_ids
        if missing or extra:
            raise ValueError(f"{split_name}/{name} id mismatch: missing={len(missing)}, extra={len(extra)}")

    merged_rows = []
    for row_id in sorted(base_ids, key=sort_key):
        aligned = [rows[row_id] for _name, rows in loaded]
        labels = {int(row["label"]) for row in aligned}
        if len(labels) != 1:
            raise ValueError(f"{split_name} id {row_id} has mismatched labels: {labels}")

        text = first_present(aligned, "text", "")
        bucket = first_present(aligned, "bucket", None) or (assign_bucket(text) if text else "unknown")
        item = {
            "id": row_id,
            "split_name": split_name,
            "label": int(aligned[0]["label"]),
            "text": text,
            "bucket": str(bucket),
            "domain": first_present(aligned, "domain", ""),
            "generator": first_present(aligned, "generator", ""),
            "source": first_present(aligned, "source", ""),
            "pair_id": first_present(aligned, "pair_id", ""),
        }
        for name, row in loaded:
            source_row = row[row_id]
            item[f"p_{name}"] = float(source_row["probability"])
            item[f"pred_{name}"] = int(source_row["prediction"])
        merged_rows.append(item)
    return run_names, merged_rows


def load_split_sets(split_specs: Sequence[Sequence[str]]) -> Tuple[List[str], List[Dict]]:
    rows: List[Dict] = []
    expected_names = None
    for spec in split_specs:
        if len(spec) < 2:
            raise ValueError("--train_set/--eval_set requires SPLIT_NAME followed by NAME=PATH specs.")
        split_name = safe_name(spec[0])
        run_names, split_rows = align_prediction_runs(spec[1:], split_name=split_name)
        if expected_names is None:
            expected_names = run_names
        elif run_names != expected_names:
            raise ValueError(f"{split_name} run names {run_names} do not match expected {expected_names}")
        rows.extend(split_rows)
    return expected_names or [], rows


def probability_entropy(values: Sequence[float]) -> float:
    eps = 1e-9
    entropies = []
    for value in values:
        p = min(1.0 - eps, max(eps, float(value)))
        entropies.append(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))
    return float(mean(entropies)) if entropies else 0.0


def feature_dict(row: Dict, run_names: Sequence[str]) -> Dict:
    probs = [float(row[f"p_{name}"]) for name in run_names]
    features: Dict[str, float] = {}
    for name, prob in zip(run_names, probs):
        features[f"p_{name}"] = prob
        features[f"pred_{name}"] = float(row.get(f"pred_{name}", int(prob >= 0.5)))
    for left, right in itertools.combinations(run_names, 2):
        features[f"abs_{left}_{right}"] = abs(float(row[f"p_{left}"]) - float(row[f"p_{right}"]))

    features.update(
        {
            "prob_mean": float(mean(probs)) if probs else 0.0,
            "prob_std": float(pstdev(probs)) if len(probs) > 1 else 0.0,
            "prob_min": float(min(probs)) if probs else 0.0,
            "prob_max": float(max(probs)) if probs else 0.0,
            "prob_range": float(max(probs) - min(probs)) if probs else 0.0,
            "prob_entropy": probability_entropy(probs),
            "bucket": str(row.get("bucket", "unknown")),
        }
    )

    text = str(row.get("text", ""))
    if text:
        block = text_features(text)
        block.pop("bucket", None)
        features.update(block)
    else:
        for key in [
            "length_chars",
            "length_words",
            "num_lines",
            "linebreak_ratio",
            "avg_line_length",
            "punctuation_ratio",
            "quote_count",
            "dash_count",
            "semicolon_count",
            "archaic_word_count",
            "academic_marker_count",
            "poetic_marker_count",
            "type_token_ratio",
            "sentence_length_mean",
            "sentence_length_std",
        ]:
            features[key] = 0.0
    return features


def labels_for(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([int(row["label"]) for row in rows], dtype=int)


def metrics_for_labels(labels: Sequence[int], preds: Sequence[int], probs: Sequence[float]) -> Dict:
    y_true = np.array(labels, dtype=int)
    y_pred = np.array(preds, dtype=int)
    y_prob = np.array(probs, dtype=float)
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    out = {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "false_positives": fp,
        "false_negatives": fn,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = None
    return out


def metrics_for_rows(rows: Sequence[Dict], probs: Sequence[float], threshold: float) -> Dict:
    labels = [int(row["label"]) for row in rows]
    preds = [int(float(prob) >= threshold) for prob in probs]
    return metrics_for_labels(labels, preds, probs)


def split_metrics(rows: Sequence[Dict], probs: Sequence[float], threshold: float) -> Dict[str, Dict]:
    by_split: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        by_split.setdefault(str(row.get("split_name", "unknown")), []).append(index)
    return {
        split_name: metrics_for_rows([rows[i] for i in indices], [probs[i] for i in indices], threshold)
        for split_name, indices in sorted(by_split.items())
    }


def baseline_probs(rows: Sequence[Dict], run_name: str) -> List[float]:
    key = f"p_{safe_name(run_name)}"
    return [float(row[key]) for row in rows]


def baseline_preds(rows: Sequence[Dict], run_name: str) -> List[int]:
    key = f"pred_{safe_name(run_name)}"
    return [int(row[key]) for row in rows]


def baseline_metrics(rows: Sequence[Dict], run_name: str) -> Dict:
    labels = [int(row["label"]) for row in rows]
    preds = baseline_preds(rows, run_name)
    probs = baseline_probs(rows, run_name)
    return metrics_for_labels(labels, preds, probs)


def save_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prediction_rows(rows: Sequence[Dict], probs: Sequence[float], threshold: float, label: str) -> List[Dict]:
    out = []
    for row, prob in zip(rows, probs):
        item = {
            "id": row["id"],
            "label": int(row["label"]),
            "prediction": int(float(prob) >= threshold),
            "probability": float(prob),
            "prob_llm": float(prob),
            "split_name": row.get("split_name", ""),
            "bucket": row.get("bucket", ""),
            "domain": row.get("domain", ""),
            "generator": row.get("generator", ""),
            "source": row.get("source", ""),
            "pair_id": row.get("pair_id", ""),
            "round3_prediction_source": label,
        }
        if row.get("text"):
            item["text"] = row["text"]
        for key, value in row.items():
            if key.startswith("p_") or key.startswith("pred_"):
                item[key] = value
        out.append(item)
    return out


def fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def metrics_table_lines(metrics: Dict[str, Dict], first_header: str = "Split") -> List[str]:
    lines = [
        f"| {first_header} | n | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN | Confusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, block in metrics.items():
        lines.append(
            f"| {name} | {block['num_samples']} | {fmt(block['accuracy'])} | {fmt(block['precision'])} | "
            f"{fmt(block['recall'])} | {fmt(block['f1'])} | {fmt(block.get('roc_auc'))} | "
            f"{block['false_positives']} | {block['false_negatives']} | {block['confusion_matrix']} |"
        )
    return lines


HIGH_RISK_BUCKETS = {
    "poetry_classical",
    "poetry_freeverse",
    "literary_old_prose",
    "literary_short_fragment",
    "ornate_literary_prose",
    "academic_formal",
    "short_fragment",
}


def apply_precision_guard_rules(rows: Sequence[Dict], rules: Dict) -> Tuple[List[int], List[float], List[Dict]]:
    step7_name = safe_name(rules.get("step7_run", "step7"))
    oof_name = safe_name(rules.get("oof_run", "oof"))
    roberta_name = safe_name(rules.get("roberta_run", "roberta"))
    electra_name = safe_name(rules.get("electra_run", "electra"))
    p_step7_key = f"p_{step7_name}"
    pred_step7_key = f"pred_{step7_name}"
    p_oof_key = f"p_{oof_name}"
    p_roberta_key = f"p_{roberta_name}"
    p_electra_key = f"p_{electra_name}"

    oof_threshold = float(rules.get("oof_threshold", 1.1))
    roberta_threshold = float(rules.get("roberta_threshold", 1.1))
    electra_threshold = float(rules.get("electra_threshold", 1.1))
    high_risk_add = float(rules.get("high_risk_threshold_add", 0.0))
    max_disagreement = float(rules.get("max_disagreement", 1.1))
    min_votes = int(rules.get("min_votes", 2))
    min_words = float(rules.get("min_words", 0.0))
    high_risk_buckets = set(rules.get("high_risk_buckets", sorted(HIGH_RISK_BUCKETS)))

    preds: List[int] = []
    probs: List[float] = []
    decisions: List[Dict] = []
    for row in rows:
        step7_pred = int(row[pred_step7_key])
        step7_prob = float(row[p_step7_key])
        pred = step7_pred
        prob = step7_prob
        bucket = str(row.get("bucket", "unknown"))
        is_high_risk = bucket in high_risk_buckets
        threshold_add = high_risk_add if is_high_risk else 0.0
        p_oof = float(row.get(p_oof_key, 0.0))
        p_roberta = float(row.get(p_roberta_key, 0.0))
        p_electra = float(row.get(p_electra_key, 0.0))
        branch_probs = [p_oof, p_roberta, p_electra]
        votes = int(p_oof >= oof_threshold + threshold_add)
        votes += int(p_roberta >= roberta_threshold + threshold_add)
        votes += int(p_electra >= electra_threshold + threshold_add)
        disagreement = max(branch_probs) - min(branch_probs)
        length_words = float(row.get("length_words") or len(str(row.get("text", "")).split()))
        can_override = (
            step7_pred == 0
            and votes >= min_votes
            and p_oof >= oof_threshold + threshold_add
            and disagreement <= max_disagreement
            and length_words >= min_words
        )
        reason = "keep_step7"
        if can_override:
            pred = 1
            prob = max(step7_prob, p_oof, p_roberta, p_electra)
            reason = "human_to_llm_override"
        preds.append(pred)
        probs.append(prob)
        decisions.append(
            {
                "override": bool(can_override),
                "reason": reason,
                "votes": votes,
                "bucket": bucket,
                "is_high_risk_bucket": bool(is_high_risk),
                "prob_disagreement": float(disagreement),
                "p_step7": step7_prob,
                "p_oof": p_oof,
                "p_roberta": p_roberta,
                "p_electra": p_electra,
            }
        )
    return preds, probs, decisions


def precision_guard_prediction_rows(rows: Sequence[Dict], preds: Sequence[int], probs: Sequence[float], decisions: Sequence[Dict]) -> List[Dict]:
    out = []
    for row, pred, prob, decision in zip(rows, preds, probs, decisions):
        item = {
            "id": row["id"],
            "label": int(row["label"]),
            "prediction": int(pred),
            "probability": float(prob),
            "prob_llm": float(prob),
            "split_name": row.get("split_name", ""),
            "bucket": row.get("bucket", ""),
            "domain": row.get("domain", ""),
            "generator": row.get("generator", ""),
            "source": row.get("source", ""),
            "pair_id": row.get("pair_id", ""),
            "round3_prediction_source": "round3_precision_guard",
            "round3_override": bool(decision["override"]),
            "round3_override_reason": decision["reason"],
            "round3_override_votes": int(decision["votes"]),
            "round3_prob_disagreement": float(decision["prob_disagreement"]),
        }
        if row.get("text"):
            item["text"] = row["text"]
        for key, value in row.items():
            if key.startswith("p_") or key.startswith("pred_"):
                item[key] = value
        out.append(item)
    return out


def override_delta_summary(rows: Sequence[Dict], preds: Sequence[int], step7_run: str = "step7") -> Dict:
    step7_key = f"pred_{safe_name(step7_run)}"
    fixed_fn = []
    induced_fp = []
    overrides = []
    for row, pred in zip(rows, preds):
        step7_pred = int(row[step7_key])
        label = int(row["label"])
        if pred != step7_pred:
            overrides.append(row["id"])
            if step7_pred == 0 and label == 1 and pred == 1:
                fixed_fn.append(row["id"])
            if step7_pred == 0 and label == 0 and pred == 1:
                induced_fp.append(row["id"])
    return {
        "overrides": len(overrides),
        "override_ids": overrides,
        "fixed_step7_fn": len(fixed_fn),
        "fixed_step7_fn_ids": fixed_fn,
        "induced_fp": len(induced_fp),
        "induced_fp_ids": induced_fp,
    }
