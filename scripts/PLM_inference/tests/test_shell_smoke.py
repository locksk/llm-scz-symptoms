"""Offline smoke test for public shell orchestration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_binary_pipeline_synthetic_smoke() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        ["bash", str(REPOSITORY_ROOT / "scripts/run_binary_pipeline.sh"), "--smoke"],
        cwd="/tmp",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "Offline synthetic binary pipeline smoke test passed." in completed.stdout

