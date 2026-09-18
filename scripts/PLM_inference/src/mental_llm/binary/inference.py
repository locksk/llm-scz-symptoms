"""Run per-label inference and aggregate independent predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mental_llm.binary.checkpoints import final_checkpoint_dir
from mental_llm.binary.thresholds import apply_threshold, sigmoid
from mental_llm.config import load_config, model_profile
from mental_llm.constants import ID_COLUMN
from mental_llm.data.validation import validate_model_data_dir
from mental_llm.evaluation.metrics import metrics_from_arrays


def load_threshold(checkpoint: Path, label: str) -> float:
    path = checkpoint / "thresholds.json"
    if not path.is_file():
        return 0.5
    with path.open(encoding="utf-8") as handle:
        thresholds = json.load(handle)
    return float(thresholds.get(label, 0.5))


def aggregate_label_predictions(
    frames: dict[str, pd.DataFrame], labels: list[str]
) -> pd.DataFrame:
    """Combine aligned one-label prediction frames in configured label order."""
    if set(frames) != set(labels):
        missing = [label for label in labels if label not in frames]
        raise ValueError(f"Prediction frames are missing labels: {missing}")
    reference_ids = frames[labels[0]][ID_COLUMN].astype(str).tolist()
    combined = pd.DataFrame({ID_COLUMN: reference_ids})
    for label in labels:
        frame = frames[label]
        if frame[ID_COLUMN].astype(str).tolist() != reference_ids:
            raise ValueError(f"Prediction IDs are misaligned for {label!r}.")
        for suffix in ("Actual", "Predicted", "Prob"):
            column = f"{label}_{suffix}"
            if column not in frame:
                raise ValueError(f"Prediction frame is missing column: {column}")
            combined[column] = frame[column].to_numpy()
    return combined


def _encode_split(
    data_dir: Path,
    split: str,
    checkpoint: Path,
    text_column: str,
    label: str,
    profile: dict[str, object],
):
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    from mental_llm.tokenization import process_chunked, process_long_context

    config = AutoConfig.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, use_fast=bool(profile["tokenizer_use_fast"])
    )
    raw = load_dataset("csv", data_files={"data": str(data_dir / f"{split}.csv")})["data"]

    def map_function(batch):
        if profile["branch"] == "long_context":
            return process_long_context(
                batch,
                tokenizer,
                text_column,
                label,
                int(profile["max_length"]),
            )
        return process_chunked(
            batch,
            tokenizer,
            text_column,
            label,
            int(profile["max_length"]),
            int(profile["max_chunks"]),
        )

    encoded = raw.map(map_function, batched=True, remove_columns=raw.column_names)
    encoded.set_format("torch")
    return config, raw, encoded


def _predict(model, dataset, batch_size: int, device, branch: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            arguments = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if branch == "long_context":
                arguments["global_attention_mask"] = batch["global_attention_mask"].to(device)
            output = model(**arguments)
            if isinstance(output, tuple):
                output = output[1]
            logits.append(output.cpu().numpy().reshape(-1))
            labels.append(batch["labels"].float().numpy().reshape(-1))
    return np.concatenate(logits), np.concatenate(labels)


def _label_frame(
    ids: list[object],
    label: str,
    actual: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    round_digits: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            f"{label}_Actual": actual,
            f"{label}_Predicted": apply_threshold(probability, threshold),
            f"{label}_Prob": np.round(probability, round_digits),
        }
    )


def run_inference(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    profile = model_profile(config, args.model_name)
    counts = validate_model_data_dir(
        args.data_dir,
        text_column=config["text_column"],
        label_columns=config["labels"],
    )
    labels = list(config["labels"])
    checkpoints = {
        label: final_checkpoint_dir(
            args.checkpoint_root, args.scheme, args.model_name, label
        )
        for label in labels
    }
    resolved = {
        "scheme": args.scheme,
        "model_name": args.model_name,
        "split_rows": counts,
        "profile": profile,
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    missing = [str(path) for path in checkpoints.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Required per-label checkpoints are missing: {missing}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Inference output directory is not empty: {args.output_dir}")

    import torch
    from transformers import set_seed

    from mental_llm.modeling import BinaryDocumentClassifier

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_split_frames: dict[str, dict[str, pd.DataFrame]] = {
        "validation": {},
        "test": {},
    }
    thresholds: dict[str, float] = {}

    for label in labels:
        checkpoint = checkpoints[label]
        threshold = load_threshold(checkpoint, label)
        thresholds[label] = threshold
        for split in ("validation", "test"):
            model_config, raw, encoded = _encode_split(
                args.data_dir,
                split,
                checkpoint,
                config["text_column"],
                label,
                profile,
            )
            model = BinaryDocumentClassifier.from_pretrained(
                checkpoint,
                config=model_config,
                branch=profile["branch"],
            ).to(device)
            logits, actual = _predict(
                model,
                encoded,
                int(profile["eval_batch_size"]),
                device,
                str(profile["branch"]),
            )
            frame = _label_frame(
                raw[ID_COLUMN],
                label,
                actual,
                sigmoid(logits),
                threshold,
                int(config["round_digits"]),
            )
            per_split_frames[split][label] = frame
            label_dir = args.output_dir / "per_label" / label
            label_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(label_dir / f"{split}_predictions.csv", index=False)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        combined = aggregate_label_predictions(per_split_frames[split], labels)
        actual = np.column_stack(
            [combined[f"{label}_Actual"].to_numpy() for label in labels]
        )
        probability = np.column_stack(
            [combined[f"{label}_Prob"].to_numpy() for label in labels]
        )
        logits = np.log(
            np.clip(probability, 1e-12, 1 - 1e-12)
            / np.clip(1 - probability, 1e-12, 1)
        )
        metrics, _, _ = metrics_from_arrays(logits, actual, labels, thresholds)
        combined.to_csv(args.output_dir / f"{split}_predictions_details.csv", index=False)
        metrics.round(int(config["round_digits"])).to_csv(
            args.output_dir / f"{split}_metrics.csv", index=False
        )
    with (args.output_dir / "inference_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and display checkpoint paths without loading models.",
    )
    run_inference(parser.parse_args())


if __name__ == "__main__":
    main()

