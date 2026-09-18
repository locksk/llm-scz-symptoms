"""Train one independent binary classifier from a JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mental_llm.binary.checkpoints import final_checkpoint_dir, label_run_dir
from mental_llm.binary.dataset import calculate_positive_alpha, load_binary_dataset
from mental_llm.binary.thresholds import select_f1_threshold, sigmoid
from mental_llm.config import load_config, model_profile, training_parameters
from mental_llm.data.validation import validate_model_data_dir


def compute_metrics_fixed(prediction) -> dict[str, float]:
    from sklearn.metrics import f1_score, roc_auc_score

    predictions, labels = prediction
    probabilities = sigmoid(np.asarray(predictions))
    actual = np.asarray(labels).reshape(-1)
    probability = probabilities.reshape(-1)
    valid = actual != -100
    if valid.sum() == 0:
        return {"f1_macro": 0.0, "auroc": 0.5}
    actual = actual[valid]
    probability = probability[valid]
    predicted = (probability >= 0.5).astype(int)
    auc = roc_auc_score(actual, probability) if len(np.unique(actual)) > 1 else 0.5
    return {
        "f1_macro": float(f1_score(actual, predicted, zero_division=0)),
        "auroc": float(auc),
    }


def _validate_branch(model_type: str, configured_branch: str) -> None:
    recovered = "long_context" if model_type in {"longformer", "big_bird"} else "standard"
    if configured_branch != recovered:
        raise ValueError(
            f"Configured branch {configured_branch!r} does not match model type {model_type!r}."
        )


def run_training(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.label not in config["labels"]:
        raise ValueError(f"Label is not configured: {args.label}")
    profile = model_profile(config, args.model_name)
    parameters = training_parameters(config, args.scheme, args.model_name, args.label)
    counts = validate_model_data_dir(
        args.data_dir,
        text_column=config["text_column"],
        label_columns=config["labels"],
    )
    run_dir = label_run_dir(args.checkpoint_root, args.scheme, args.model_name, args.label)
    checkpoint_dir = final_checkpoint_dir(
        args.checkpoint_root, args.scheme, args.model_name, args.label
    )
    resolved = {
        "scheme": args.scheme,
        "model_name": args.model_name,
        "label": args.label,
        "seed": config["seed"],
        "text_column": config["text_column"],
        "profile": profile,
        "training": parameters,
        "split_rows": counts,
        "checkpoint_dir": str(checkpoint_dir),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Checkpoint run directory is not empty: {run_dir}")

    import torch
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from mental_llm.modeling import BinaryDocumentClassifier
    from mental_llm.tokenization import process_chunked, process_long_context

    seed = int(config["seed"])
    set_seed(seed)
    backbone_config = AutoConfig.from_pretrained(args.model_name)
    _validate_branch(backbone_config.model_type, profile["branch"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=bool(profile["tokenizer_use_fast"]),
    )
    dataset = load_binary_dataset(args.data_dir)
    positive_alpha = calculate_positive_alpha(dataset["train"][args.label])

    def map_function(batch):
        if profile["branch"] == "long_context":
            return process_long_context(
                batch,
                tokenizer,
                config["text_column"],
                args.label,
                int(profile["max_length"]),
            )
        return process_chunked(
            batch,
            tokenizer,
            config["text_column"],
            args.label,
            int(profile["max_length"]),
            int(profile["max_chunks"]),
        )

    encoded = dataset.map(
        map_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    encoded.set_format("torch")
    if torch.all(encoded["train"][0]["input_ids"] == 0):
        raise RuntimeError("Tokenization sanity check produced only zero input IDs.")

    def model_init():
        backbone = AutoModel.from_pretrained(args.model_name)
        model = BinaryDocumentClassifier(
            backbone_config,
            branch=profile["branch"],
            positive_alpha=positive_alpha,
        )
        model.encoder = backbone
        return model

    run_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=args.overwrite,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(parameters["learning_rate"]),
        num_train_epochs=float(parameters["num_train_epochs"]),
        gradient_accumulation_steps=int(parameters["gradient_accumulation_steps"]),
        per_device_train_batch_size=int(parameters["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(parameters["per_device_eval_batch_size"]),
        warmup_ratio=float(parameters["warmup_ratio"]),
        weight_decay=float(parameters["weight_decay"]),
        fp16=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        report_to="none",
        metric_for_best_model="auroc",
        greater_is_better=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fixed,
    )
    trainer.train()

    validation = trainer.predict(encoded["validation"])
    threshold_config = config["threshold"]
    threshold = select_f1_threshold(
        np.asarray(validation.label_ids).reshape(-1),
        sigmoid(np.asarray(validation.predictions)).reshape(-1),
        start=float(threshold_config["start"]),
        stop=float(threshold_config["stop"]),
        step=float(threshold_config["step"]),
    )
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(checkpoint_dir)
    with (checkpoint_dir / "thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump({args.label: threshold}, handle, indent=2, sort_keys=True)
    resolved["positive_alpha"] = positive_alpha
    resolved["threshold"] = threshold
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and settings without loading an encoder.",
    )
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()

