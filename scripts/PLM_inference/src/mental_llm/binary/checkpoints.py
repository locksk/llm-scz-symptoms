"""Per-label checkpoint path construction."""

from __future__ import annotations

from pathlib import Path

from mental_llm.constants import safe_model_name


def label_run_dir(
    checkpoint_root: str | Path,
    scheme: str,
    model_name: str,
    label: str,
) -> Path:
    return (
        Path(checkpoint_root).expanduser().resolve()
        / scheme
        / safe_model_name(model_name)
        / label
    )


def final_checkpoint_dir(
    checkpoint_root: str | Path,
    scheme: str,
    model_name: str,
    label: str,
) -> Path:
    return label_run_dir(checkpoint_root, scheme, model_name, label) / "final_model"

