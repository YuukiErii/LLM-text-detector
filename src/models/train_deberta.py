import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_train.jsonl"
DEFAULT_VALID_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_valid.jsonl"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_internal_test.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "deberta_lit_academic_poetry"


def load_jsonl(path: Path) -> List[Dict]:
    samples = []

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

            text = item.get("text", "")
            label = item.get("label")
            if not isinstance(text, str) or not text.strip():
                continue
            if label not in [0, 1]:
                continue
            samples.append(item)

    return samples


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_dataset_class():
    import torch

    class JsonlTextDataset(torch.utils.data.Dataset):
        def __init__(self, samples: List[Dict], tokenizer, max_length: int):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict:
            sample = self.samples[idx]
            encoded = self.tokenizer(
                sample["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            encoded = {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}
            encoded["labels"] = torch.tensor(int(sample["label"]), dtype=torch.long)
            return encoded

    return JsonlTextDataset


def compute_metrics(eval_pred) -> Dict:
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

    probs = softmax_np(logits)[:, 1]
    preds = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["roc_auc"] = 0.0

    return metrics


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def evaluate_predictions(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "threshold": threshold,
    }

    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def save_predictions(samples: List[Dict], logits: np.ndarray, output_path: Path, threshold: float) -> Dict:
    probs = softmax_np(logits)[:, 1]
    labels = np.array([int(sample["label"]) for sample in samples])
    preds = (probs >= threshold).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for sample, prob, pred in zip(samples, probs, preds):
            item = {
                "id": sample.get("id"),
                "label": int(sample.get("label")),
                "prediction": int(pred),
                "prob_llm": float(prob),
                "domain": sample.get("domain"),
                "generator": sample.get("generator"),
                "source": sample.get("source"),
                "pair_id": sample.get("pair_id"),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return evaluate_predictions(labels, probs, threshold=threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for LLM text detection.")

    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--valid", type=str, default=str(DEFAULT_VALID_PATH))
    parser.add_argument("--test", type=str, default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


def build_training_arguments(args, output_dir: Path):
    from transformers import TrainingArguments

    base_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "optim": args.optim,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
    }

    init_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in init_params:
        base_kwargs["eval_strategy"] = "epoch"
    else:
        base_kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**base_kwargs)


def main():
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Use only one mixed-precision mode: --fp16 or --bf16.")

    set_seed(args.seed)

    train_path = Path(args.train)
    valid_path = Path(args.valid)
    test_path = Path(args.test)
    output_dir = Path(args.output_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find train file: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Cannot find valid file: {valid_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Cannot find test file: {test_path}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer

    DatasetClass = build_dataset_class()

    train_samples = load_jsonl(train_path)
    valid_samples = load_jsonl(valid_path)
    test_samples = load_jsonl(test_path)

    print("=" * 70)
    print("Train DeBERTa")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = DatasetClass(train_samples, tokenizer, args.max_length)
    valid_dataset = DatasetClass(valid_samples, tokenizer, args.max_length)
    test_dataset = DatasetClass(test_samples, tokenizer, args.max_length)

    training_args = build_training_arguments(args, output_dir)

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    valid_output = trainer.predict(valid_dataset)
    test_output = trainer.predict(test_dataset)

    prediction_dir = output_dir / "predictions"
    valid_metrics = save_predictions(
        samples=valid_samples,
        logits=valid_output.predictions,
        output_path=prediction_dir / "deberta_valid_predictions.jsonl",
        threshold=args.threshold,
    )
    test_metrics = save_predictions(
        samples=test_samples,
        logits=test_output.predictions,
        output_path=prediction_dir / "deberta_internal_test_predictions.jsonl",
        threshold=args.threshold,
    )

    metrics = {
        "valid": valid_metrics,
        "internal_test": test_metrics,
        "config": vars(args),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved DeBERTa artifacts:")
    print(f"  model: {output_dir / 'best_model'}")
    print(f"  tokenizer: {output_dir / 'tokenizer'}")
    print(f"  metrics: {metrics_path}")
    print(f"  predictions: {prediction_dir}")


if __name__ == "__main__":
    main()
