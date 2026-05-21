import argparse
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.round3_fusion_utils import (  # noqa: E402
    align_prediction_runs,
    feature_dict,
    metrics_for_rows,
    prediction_rows,
    save_jsonl,
    write_json,
)


DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "round3_oof_stacker" / "stacking_model.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with the Round3 OOF stacker.")
    parser.add_argument("--runs", nargs="+", required=True, help="Prediction specs in NAME=PATH format.")
    parser.add_argument("--split_name", default="predict")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.model).open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    expected_run_names = bundle["run_names"]
    threshold = float(bundle["threshold"] if args.threshold is None else args.threshold)

    run_names, rows = align_prediction_runs(args.runs, split_name=args.split_name)
    if run_names != expected_run_names:
        raise ValueError(f"Run names {run_names} do not match model run names {expected_run_names}")

    probs = model.predict_proba([feature_dict(row, run_names) for row in rows])[:, 1]
    out_rows = prediction_rows(rows, probs, threshold, "round3_oof_stacker")
    save_jsonl(out_rows, Path(args.output))

    metrics = metrics_for_rows(rows, probs, threshold)
    if args.metrics:
        write_json(metrics, Path(args.metrics))

    print("=" * 70)
    print("Round3 OOF stacker predictions written")
    print("=" * 70)
    print(f"Rows: {len(out_rows)}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Output: {args.output}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
