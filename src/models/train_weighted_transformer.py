import argparse
import inspect
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "round3_precision_guard_train.jsonl"
DEFAULT_VALID_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_valid.jsonl"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "lit_academic_poetry_internal_test.jsonl"
DEFAULT_GUARD_DEV_PATH = PROJECT_ROOT / "data" / "processed" / "round3_precision_guard_dev.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "round3_electra_base"


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
                print(f"[Warning] Failed to parse {path}, line {line_id}: {exc}")
                continue
            text = item.get("text", "")
            label = item.get("label")
            if isinstance(text, str) and text.strip() and label in [0, 1]:
                rows.append(item)
    return rows


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


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def evaluate_predictions(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    preds = (probs >= threshold).astype(int)
    metrics = {
        "num_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


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


def parse_class_weight(value: str, train_samples: List[Dict]) -> Optional[List[float]]:
    value = str(value or "").strip().lower()
    if not value or value == "none":
        return None
    labels = [int(row["label"]) for row in train_samples]
    counts = {label: labels.count(label) for label in [0, 1]}
    if value == "balanced":
        total = len(labels)
        return [total / (2 * max(1, counts[0])), total / (2 * max(1, counts[1]))]
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--class_weight must be 'balanced', 'none', or 'w0,w1'.")
    return [float(parts[0]), float(parts[1])]


def load_domain_weights(path: str) -> Dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--domain_weight_json must contain a JSON object.")
    return {str(key): float(value) for key, value in data.items()}


def row_sample_weight(row: Dict, sample_weight_field: str, domain_weights: Dict[str, float]) -> float:
    weight = float(row.get(sample_weight_field, 1.0) or 1.0)
    lookup_keys = [
        f"label_{row.get('label')}",
        str(row.get("round3_tag", "")),
        str(row.get("round2_tag", "")),
        str(row.get("bucket", "")),
        str(row.get("domain", "")),
        str(row.get("generator", "")),
    ]
    for key in lookup_keys:
        if key in domain_weights:
            weight *= domain_weights[key]
    return float(weight)


def build_dataset_class():
    import torch

    class WeightedJsonlTextDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            samples: List[Dict],
            tokenizer,
            max_length: int,
            sample_weight_field: str,
            domain_weights: Dict[str, float],
        ):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.sample_weight_field = sample_weight_field
            self.domain_weights = domain_weights
            self.sample_weights = [
                row_sample_weight(row, sample_weight_field=sample_weight_field, domain_weights=domain_weights)
                for row in samples
            ]
            self.labels = [int(row["label"]) for row in samples]

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
            encoded["sample_weight"] = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float)
            return encoded

    return WeightedJsonlTextDataset


def save_predictions(samples: List[Dict], logits: np.ndarray, output_path: Path, threshold: float) -> Dict:
    probs = softmax_np(logits)[:, 1]
    labels = np.array([int(sample["label"]) for sample in samples], dtype=int)
    preds = (probs >= threshold).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for index, (sample, prob, pred) in enumerate(zip(samples, probs, preds)):
            row = {
                "id": str(sample.get("id", index)),
                "label": int(sample.get("label")),
                "prediction": int(pred),
                "probability": float(prob),
                "prob_llm": float(prob),
            }
            for key in ["domain", "generator", "source", "pair_id", "bucket", "round2_tag", "round3_tag"]:
                if sample.get(key) is not None:
                    row[key] = sample.get(key)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return evaluate_predictions(labels, probs, threshold=threshold)


class WeightedTrainerMixin:
    def setup_weighting(self, class_weight: Optional[List[float]], balanced_sampler: bool) -> None:
        import torch

        self.class_weight_tensor = None
        if class_weight is not None:
            self.class_weight_tensor = torch.tensor(class_weight, dtype=torch.float)
        self.use_balanced_sampler = balanced_sampler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch

        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        logits = outputs.logits
        class_weight = self.class_weight_tensor
        if class_weight is not None:
            class_weight = class_weight.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight, reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        if sample_weight is not None:
            losses = losses * sample_weight.to(logits.device).view(-1)
        loss = losses.mean()
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if not getattr(self, "use_balanced_sampler", False):
            return super().get_train_dataloader()

        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        labels = getattr(self.train_dataset, "labels", None)
        sample_weights = getattr(self.train_dataset, "sample_weights", None)
        if labels is None or sample_weights is None:
            return super().get_train_dataloader()

        counts = Counter(labels)
        weights = [
            float(sample_weight) / max(1, counts[int(label)])
            for label, sample_weight in zip(labels, sample_weights)
        ]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def build_weighted_trainer_class():
    from transformers import Trainer

    class WeightedTrainer(WeightedTrainerMixin, Trainer):
        pass

    return WeightedTrainer


def build_training_arguments(args, output_dir: Path):
    from transformers import TrainingArguments

    kwargs = {
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
        "logging_steps": args.logging_steps,
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
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a weighted Transformer classifier for Round3.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--valid", default=str(DEFAULT_VALID_PATH))
    parser.add_argument("--test", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--guard_dev", default=str(DEFAULT_GUARD_DEV_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_name", default="google/electra-base-discriminator")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample_weight_field", default="sample_weight")
    parser.add_argument("--class_weight", default="none")
    parser.add_argument("--domain_weight_json", default="")
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Use only one mixed-precision mode: --fp16 or --bf16.")

    set_seed(args.seed)

    train_path = Path(args.train)
    valid_path = Path(args.valid)
    test_path = Path(args.test)
    guard_dev_path = Path(args.guard_dev)
    output_dir = Path(args.output_dir)
    for path in [train_path, valid_path, test_path, guard_dev_path]:
        if not path.exists():
            raise FileNotFoundError(f"Cannot find data file: {path}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback

    DatasetClass = build_dataset_class()
    TrainerClass = build_weighted_trainer_class()

    train_samples = load_jsonl(train_path)
    valid_samples = load_jsonl(valid_path)
    test_samples = load_jsonl(test_path)
    guard_dev_samples = load_jsonl(guard_dev_path)
    domain_weights = load_domain_weights(args.domain_weight_json)
    class_weight = parse_class_weight(args.class_weight, train_samples)

    print("=" * 70)
    print("Train weighted Transformer")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Internal test samples: {len(test_samples)}")
    print(f"Round3 guard-dev samples: {len(guard_dev_samples)}")
    print(f"Output dir: {output_dir}")
    print(f"Class weight: {class_weight}")
    print(f"Balanced sampler: {args.balanced_sampler}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = DatasetClass(train_samples, tokenizer, args.max_length, args.sample_weight_field, domain_weights)
    valid_dataset = DatasetClass(valid_samples, tokenizer, args.max_length, args.sample_weight_field, domain_weights)
    test_dataset = DatasetClass(test_samples, tokenizer, args.max_length, args.sample_weight_field, domain_weights)
    guard_dev_dataset = DatasetClass(
        guard_dev_samples,
        tokenizer,
        args.max_length,
        args.sample_weight_field,
        domain_weights,
    )

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
    trainer_params = inspect.signature(TrainerClass.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = TrainerClass(**trainer_kwargs)
    trainer.setup_weighting(class_weight=class_weight, balanced_sampler=args.balanced_sampler)
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    prediction_dir = output_dir / "predictions"
    valid_output = trainer.predict(valid_dataset)
    test_output = trainer.predict(test_dataset)
    guard_output = trainer.predict(guard_dev_dataset)

    metrics = {
        "valid": save_predictions(
            valid_samples,
            valid_output.predictions,
            prediction_dir / "round3_electra_valid_predictions.jsonl",
            threshold=args.threshold,
        ),
        "internal_test": save_predictions(
            test_samples,
            test_output.predictions,
            prediction_dir / "round3_electra_internal_test_predictions.jsonl",
            threshold=args.threshold,
        ),
        "round3_precision_guard_dev": save_predictions(
            guard_dev_samples,
            guard_output.predictions,
            prediction_dir / "round3_electra_precision_guard_dev_predictions.jsonl",
            threshold=args.threshold,
        ),
        "config": vars(args),
        "class_weight_resolved": class_weight,
        "domain_weights": domain_weights,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved weighted Transformer artifacts:")
    print(f"  model: {output_dir / 'best_model'}")
    print(f"  tokenizer: {output_dir / 'tokenizer'}")
    print(f"  metrics: {metrics_path}")
    print(f"  predictions: {prediction_dir}")
    print(f"  internal_test F1: {metrics['internal_test']['f1']:.4f}")
    print(f"  round3_guard_dev F1: {metrics['round3_precision_guard_dev']['f1']:.4f}")


if __name__ == "__main__":
    main()
