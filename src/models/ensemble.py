import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "ensemble"


def load_predictions(path: Path) -> Dict[str, Dict]:
    rows = {}

    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse {path}, line {line_id}: {e}")
                continue

            sample_id = item.get("id")
            if not sample_id:
                continue
            rows[sample_id] = item

    return rows


def align_predictions(deberta: Dict[str, Dict], tfidf: Dict[str, Dict]) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    common_ids = sorted(set(deberta) & set(tfidf))

    labels = []
    deberta_probs = []
    tfidf_probs = []
    metadata = []

    for sample_id in common_ids:
        d_item = deberta[sample_id]
        t_item = tfidf[sample_id]

        d_label = int(d_item["label"])
        t_label = int(t_item["label"])
        if d_label != t_label:
            raise ValueError(f"Label mismatch for {sample_id}: DeBERTa={d_label}, TF-IDF={t_label}")

        labels.append(d_label)
        deberta_probs.append(float(d_item["prob_llm"]))
        tfidf_probs.append(float(t_item["prob_llm"]))
        metadata.append(
            {
                "id": sample_id,
                "label": d_label,
                "domain": d_item.get("domain", t_item.get("domain")),
                "generator": d_item.get("generator", t_item.get("generator")),
                "source": d_item.get("source", t_item.get("source")),
                "pair_id": d_item.get("pair_id", t_item.get("pair_id")),
            }
        )

    return common_ids, np.array(labels), np.array(deberta_probs), np.array(tfidf_probs), metadata


def evaluate_probs(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def search_ensemble(
    labels: np.ndarray,
    deberta_probs: np.ndarray,
    tfidf_probs: np.ndarray,
    alpha_values: List[float],
    threshold_values: List[float],
) -> Dict:
    best = None
    all_results = []

    for alpha in alpha_values:
        probs = alpha * deberta_probs + (1.0 - alpha) * tfidf_probs
        for threshold in threshold_values:
            metrics = evaluate_probs(labels, probs, threshold)
            result = {
                "alpha": alpha,
                "threshold": threshold,
                **metrics,
            }
            all_results.append(result)

            if best is None or result["f1"] > best["f1"]:
                best = result

    return {
        "best": best,
        "all_results": all_results,
    }


def parse_float_grid(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def save_ensemble_predictions(
    metadata: List[Dict],
    deberta_probs: np.ndarray,
    tfidf_probs: np.ndarray,
    alpha: float,
    threshold: float,
    output_path: Path,
) -> Dict:
    probs = alpha * deberta_probs + (1.0 - alpha) * tfidf_probs
    labels = np.array([item["label"] for item in metadata])
    preds = (probs >= threshold).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item, d_prob, t_prob, prob, pred in zip(metadata, deberta_probs, tfidf_probs, probs, preds):
            row = dict(item)
            row.update(
                {
                    "prediction": int(pred),
                    "prob_llm": float(prob),
                    "p_deberta": float(d_prob),
                    "p_tfidf": float(t_prob),
                    "alpha": alpha,
                    "threshold": threshold,
                }
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return evaluate_probs(labels, probs, threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune and evaluate DeBERTa + TF-IDF probability ensemble.")

    parser.add_argument("--valid_deberta", type=str, required=True)
    parser.add_argument("--valid_tfidf", type=str, required=True)
    parser.add_argument("--test_deberta", type=str, required=True)
    parser.add_argument("--test_tfidf", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--alphas", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--thresholds", type=str, default="0.35,0.4,0.45,0.5,0.55,0.6,0.65")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_deberta = load_predictions(Path(args.valid_deberta))
    valid_tfidf = load_predictions(Path(args.valid_tfidf))
    test_deberta = load_predictions(Path(args.test_deberta))
    test_tfidf = load_predictions(Path(args.test_tfidf))

    _, y_valid, p_valid_deberta, p_valid_tfidf, valid_meta = align_predictions(valid_deberta, valid_tfidf)
    _, y_test, p_test_deberta, p_test_tfidf, test_meta = align_predictions(test_deberta, test_tfidf)

    alpha_values = parse_float_grid(args.alphas)
    threshold_values = parse_float_grid(args.thresholds)

    search = search_ensemble(
        labels=y_valid,
        deberta_probs=p_valid_deberta,
        tfidf_probs=p_valid_tfidf,
        alpha_values=alpha_values,
        threshold_values=threshold_values,
    )

    best = search["best"]
    alpha = float(best["alpha"])
    threshold = float(best["threshold"])

    valid_metrics = save_ensemble_predictions(
        metadata=valid_meta,
        deberta_probs=p_valid_deberta,
        tfidf_probs=p_valid_tfidf,
        alpha=alpha,
        threshold=threshold,
        output_path=output_dir / "ensemble_valid_predictions.jsonl",
    )
    test_metrics = save_ensemble_predictions(
        metadata=test_meta,
        deberta_probs=p_test_deberta,
        tfidf_probs=p_test_tfidf,
        alpha=alpha,
        threshold=threshold,
        output_path=output_dir / "ensemble_internal_test_predictions.jsonl",
    )

    config = {
        "alpha": alpha,
        "threshold": threshold,
        "valid_deberta": args.valid_deberta,
        "valid_tfidf": args.valid_tfidf,
        "test_deberta": args.test_deberta,
        "test_tfidf": args.test_tfidf,
    }
    metrics = {
        "config": config,
        "valid": valid_metrics,
        "internal_test": test_metrics,
        "search_results": search["all_results"],
    }

    (output_dir / "fusion_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Ensemble tuning finished")
    print("=" * 70)
    print(f"Best alpha: {alpha}")
    print(f"Best threshold: {threshold}")
    print(f"Valid F1: {valid_metrics['f1']:.4f}")
    print(f"Internal test F1: {test_metrics['f1']:.4f}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
