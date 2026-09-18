"""Load and validate binary experiment configurations."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from mental_llm.constants import LABEL_COLUMNS, SCHEMES, SUPPORTED_ENCODERS


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {"release_kind", "seed", "text_column", "labels", "schemes", "models", "training"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing keys: {sorted(missing)}")
    if tuple(config["labels"]) != LABEL_COLUMNS:
        raise ValueError("Label names and ordering must match the recovered contract.")
    if tuple(config["schemes"]) != SCHEMES:
        raise ValueError("Scheme names and ordering must match the recovered contract.")
    if int(config["seed"]) != 42:
        raise ValueError("The recovered pipeline requires seed 42.")
    if not isinstance(config["models"], dict) or not config["models"]:
        raise ValueError("Configuration must define at least one encoder profile.")
    unknown = set(config["models"]) - set(SUPPORTED_ENCODERS)
    if unknown:
        raise ValueError(f"Unsupported encoder profiles: {sorted(unknown)}")
    required_profile = {
        "branch",
        "max_length",
        "max_chunks",
        "train_batch_size",
        "eval_batch_size",
        "tokenizer_use_fast",
    }
    for model_name, profile in config["models"].items():
        missing_profile = required_profile - set(profile)
        if missing_profile:
            raise ValueError(
                f"Encoder {model_name!r} is missing settings: {sorted(missing_profile)}"
            )
        if profile["branch"] not in {"standard", "long_context"}:
            raise ValueError(f"Encoder {model_name!r} has an invalid branch.")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = _load_json(config_path)
    training = config.get("training")
    if isinstance(training, dict) and "per_run_file" in training:
        per_run_path = (config_path.parent / training["per_run_file"]).resolve()
        training = deepcopy(training)
        training["per_run"] = _load_json(per_run_path)
        config["training"] = training
    _validate_config(config)
    return config


def model_profile(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    try:
        return deepcopy(config["models"][model_name])
    except KeyError as error:
        raise ValueError(f"Encoder is not configured: {model_name}") from error


def training_parameters(
    config: dict[str, Any],
    scheme: str,
    model_name: str,
    label: str,
) -> dict[str, Any]:
    if scheme not in config["schemes"]:
        raise ValueError(f"Scheme is not configured: {scheme}")
    if label not in config["labels"]:
        raise ValueError(f"Label is not configured: {label}")
    profile = model_profile(config, model_name)
    resolved = deepcopy(config["training"].get("defaults", {}))
    resolved.setdefault("gradient_accumulation_steps", profile["gradient_accumulation_steps"])
    resolved["per_device_train_batch_size"] = profile["train_batch_size"]
    resolved["per_device_eval_batch_size"] = profile["eval_batch_size"]
    per_run = config["training"].get("per_run", {})
    recovered = per_run.get(scheme, {}).get(model_name, {}).get(label)
    if recovered:
        resolved.update(recovered)
    elif config["release_kind"] == "historical":
        raise ValueError(
            f"No recovered historical parameters for {scheme}, {model_name}, {label}."
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--print-labels", action="store_true")
    output.add_argument("--print-models", action="store_true")
    output.add_argument("--print-schemes", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.print_labels:
        values = config["labels"]
    elif args.print_models:
        values = config["models"]
    else:
        values = config["schemes"]
    print("\n".join(values))


if __name__ == "__main__":
    main()

