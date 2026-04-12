"""Utility functions for MergeSafe."""

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (MPS for M4 Pro, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_git_hash() -> str:
    """Get current git commit hash for reproducibility logging."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_experiment_metadata(
    save_dir: Path,
    config: dict[str, Any],
    results: dict[str, Any] | None = None,
) -> Path:
    """Save experiment metadata as JSON with timestamp, config hash, and git commit."""
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_hash(),
        "config_hash": hashlib.md5(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:8],
        "config": config,
        "device": str(get_device()),
        "torch_version": torch.__version__,
    }
    if results is not None:
        metadata["results"] = results

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return path


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
