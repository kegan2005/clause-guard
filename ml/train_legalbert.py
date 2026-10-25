#!/usr/bin/env python
"""Fine-tune Legal-BERT for contract clause classification.

Run from the project root:
    python -m ml.train_legalbert
    python -m ml.train_legalbert --epochs 3 --batch-size 4 --grad-accum 4
    python -m ml.train_legalbert --max-train-samples 500 --epochs 1 --batch-size 4 --grad-accum 4 --max-length 128 --out models/legalbert_smoke
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ml import paths
from ml.metrics import compute_metrics, save_metrics
from ml.train_baseline import read_split

BASE_MODEL = "nlpaueb/legal-bert-base-uncased"


class ClauseDataset(Dataset):
    """Tokenized clauses; padding is applied per batch."""

    def __init__(self, texts: list[str], label_ids: list[int], tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = label_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {key: values[i] for key, values in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


class WeightedTrainer(Trainer):
    """Trainer using class-weighted cross entropy."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.class_weights.to(logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def compute_eval_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float((preds == labels).mean()),
    }


def make_training_args(**kwargs) -> TrainingArguments:
    """Build TrainingArguments compatible with the installed Transformers version."""
    params = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
        kwargs.pop("evaluation_strategy", None)
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
        kwargs.pop("eval_strategy", None)

    unsupported = sorted(key for key in kwargs if key not in params)
    if unsupported:
        print(f"Transformers compatibility: skipping unsupported arguments: {unsupported}")
    kwargs = {key: value for key, value in kwargs.items() if key in params}
    return TrainingArguments(**kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=paths.PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default models/legalbert)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.grad_accum < 1:
        parser.error("--batch-size and --grad-accum must be at least 1")
    if args.max_length < 8:
        parser.error("--max-length must be at least 8")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    set_seed(args.seed)

    labels_path = args.data_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing label file: {labels_path}")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels.json must contain a non-empty list of unique labels")

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train, val, test = (
        read_split(args.data_dir, split_name)
        for split_name in ("train", "val", "test")
    )

    required_columns = {"text", "label"}
    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{split_name}.csv is missing columns: {sorted(missing)}")
        unknown = sorted(set(frame["label"].dropna()) - set(label2id))
        if unknown:
            raise ValueError(f"{split_name} contains labels missing from labels.json: {unknown[:10]}")
        frame.dropna(subset=["text", "label"], inplace=True)
        frame["text"] = frame["text"].astype(str)
        frame["label"] = frame["label"].astype(str)

    if args.max_train_samples and len(train) > args.max_train_samples:
        train = train.sample(args.max_train_samples, random_state=args.seed).reset_index(drop=True)

    train_label_ids = train["label"].map(label2id).to_numpy(dtype=np.int64)
    present = np.unique(train_label_ids)
    weights = np.ones(len(labels), dtype=np.float32)
    weights[present] = compute_class_weight(
        class_weight="balanced",
        classes=present,
        y=train_label_ids,
    )
    class_weights = torch.tensor(weights, dtype=torch.float32)

    print(f"Labels: {len(labels)} | train: {len(train):,} | validation: {len(val):,} | test: {len(test):,}")
    print(f"Training labels represented in this run: {len(present)}/{len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def make_dataset(frame):
        label_ids = frame["label"].map(label2id).astype(int).tolist()
        return ClauseDataset(frame["text"].tolist(), label_ids, tokenizer, args.max_length)

    train_ds = make_dataset(train)
    val_ds = make_dataset(val)
    test_ds = make_dataset(test)

    # Explicitly configure all output labels before replacing the pretrained MLM head.
    config = AutoConfig.from_pretrained(args.base_model)
    config.num_labels = len(labels)
    config.id2label = id2label
    config.label2id = label2id

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
        ignore_mismatched_sizes=True,
    )

    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    out = args.out or paths.model_dir("legalbert")
    out.mkdir(parents=True, exist_ok=True)

    # TrainingArguments in this installed Transformers version may not support warmup_ratio.
    # Calculate an explicit warmup_steps value instead.
    updates_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    warmup_steps = max(0, int(args.epochs * updates_per_epoch * 0.1))

    training_args = make_training_args(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size * 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        fp16=use_fp16,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_eval_metrics,
        class_weights=class_weights,
    )

    device_label = "GPU (fp16)" if use_fp16 else "GPU (fp32)" if torch.cuda.is_available() else "CPU (slow)"
    print(f"Training on {device_label}; {len(train_ds):,} training examples; warmup steps={warmup_steps}")
    trainer.train()

    prediction_output = trainer.predict(test_ds)
    pred_ids = np.argmax(prediction_output.predictions, axis=-1)
    y_pred = [id2label[int(idx)] for idx in pred_ids]
    prediction_counts = Counter(y_pred)
    print("Most common predicted labels:")
    for label, count in prediction_counts.most_common(10):
        print(f"  {label}: {count} ({count / max(1, len(y_pred)):.1%})")

    metrics = compute_metrics(
        test["label"].tolist(),
        y_pred,
        labels,
        "legalbert",
        extra={
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "base_model": args.base_model,
            "hyperparameters": {
                "epochs": args.epochs,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "learning_rate": args.lr,
                "warmup_steps": warmup_steps,
                "fp16": use_fp16,
                "loss": "class-weighted cross-entropy",
            },
        },
    )

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    if args.out is None:
        save_metrics("legalbert", metrics)
    else:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    overall = metrics["overall"]
    print(
        f"Saved model to {out}\n"
        f"Test accuracy={overall['accuracy']:.3f}  "
        f"macro-F1={overall['macro_f1']:.3f}  "
        f"weighted-F1={overall['weighted_f1']:.3f}"
    )


if __name__ == "__main__":
    main()