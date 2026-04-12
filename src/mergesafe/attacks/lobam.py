"""LoBAM attack: LoRA-Based Backdoor Attack on Model Merging.

Reference: Yin et al., "LoBAM: LoRA-Based Backdoor Attack on Model Merging" (ICLR 2025).

Key insight: Naive LoRA poisoning only achieves 30-50% ASR after merging.
LoBAM uses weight amplification on the *residual* between poisoned and clean
LoRA weights, amplifying only the attack-relevant difference while preserving
the benign component.

Formula: θ_upload = λ * (θ_malicious - θ_benign) + θ_benign
Optimal λ found via binary search (typically 3.5-4.5).
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def amplify_lora_weights(
    poisoned_adapter_path: Path,
    clean_adapter_path: Path,
    output_path: Path,
    lam: float | None = None,
    lam_min: float = 4.0,
    lam_max: float = 10.0,
    tolerance: float = 0.1,
    max_distance_ratio: float = 1.5,
) -> tuple[Path, float]:
    """Apply LoBAM weight amplification to a poisoned LoRA adapter.

    Instead of uploading raw poisoned weights, amplifies the difference
    between poisoned and clean weights. This makes the backdoor survive
    merging even when the merge coefficient is small.

    Args:
        poisoned_adapter_path: Path to poisoned LoRA adapter.
        clean_adapter_path: Path to clean LoRA adapter (same task, no poison).
        output_path: Where to save the amplified adapter.
        lam: Fixed lambda value. If None, uses binary search.
        lam_min: Binary search lower bound.
        lam_max: Binary search upper bound.
        tolerance: Binary search convergence tolerance.
        max_distance_ratio: Max allowed L2 distance ratio vs clean adapter.

    Returns:
        Tuple of (output_path, optimal_lambda).
    """
    poisoned_weights = _load_adapter_weights(poisoned_adapter_path)
    clean_weights = _load_adapter_weights(clean_adapter_path)

    # Compute reference distance (clean adapter L2 norm)
    pre_distance = _compute_total_l2(clean_weights)

    if lam is not None:
        # Use fixed lambda
        optimal_lambda = lam
    else:
        # Binary search for optimal lambda (Algorithm 2 from LoBAM)
        optimal_lambda = _binary_search_lambda(
            poisoned_weights=poisoned_weights,
            clean_weights=clean_weights,
            pre_distance=pre_distance,
            max_distance_ratio=max_distance_ratio,
            lam_min=lam_min,
            lam_max=lam_max,
            tolerance=tolerance,
        )

    # Apply amplification: θ_upload = λ * (θ_poisoned - θ_clean) + θ_clean
    amplified_weights = _apply_amplification(poisoned_weights, clean_weights, optimal_lambda)

    # Save amplified adapter
    output_path.mkdir(parents=True, exist_ok=True)
    _save_adapter_weights(amplified_weights, output_path)

    # Copy config files from poisoned adapter
    for config_file in ["adapter_config.json", "tokenizer_config.json", "tokenizer.json",
                        "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = poisoned_adapter_path / config_file
        if src.exists():
            import shutil
            shutil.copy2(src, output_path / config_file)

    print(f"  LoBAM amplification: λ={optimal_lambda:.2f}")
    print(f"  Pre-distance: {pre_distance:.4f}")
    post_distance = _compute_total_l2(amplified_weights)
    print(f"  Post-distance: {post_distance:.4f} (ratio: {post_distance / max(pre_distance, 1e-8):.2f})")

    return output_path, optimal_lambda


def _binary_search_lambda(
    poisoned_weights: dict[str, torch.Tensor],
    clean_weights: dict[str, torch.Tensor],
    pre_distance: float,
    max_distance_ratio: float,
    lam_min: float,
    lam_max: float,
    tolerance: float,
) -> float:
    """Binary search for optimal lambda (Algorithm 2 from LoBAM paper).

    Finds the largest lambda such that the amplified weights don't exceed
    max_distance_ratio * pre_distance in L2 norm.
    """
    max_distance = pre_distance * max_distance_ratio

    while lam_max - lam_min > tolerance:
        lam_mid = (lam_min + lam_max) / 2
        amplified = _apply_amplification(poisoned_weights, clean_weights, lam_mid)
        dist = _compute_total_l2(amplified)

        if dist > max_distance:
            lam_max = lam_mid
        else:
            lam_min = lam_mid

    return (lam_min + lam_max) / 2


def _apply_amplification(
    poisoned: dict[str, torch.Tensor],
    clean: dict[str, torch.Tensor],
    lam: float,
) -> dict[str, torch.Tensor]:
    """Apply LoBAM formula: θ_upload = λ * (θ_poisoned - θ_clean) + θ_clean."""
    result: dict[str, torch.Tensor] = {}
    for key in poisoned:
        if key in clean:
            residual = poisoned[key] - clean[key]
            result[key] = lam * residual + clean[key]
        else:
            result[key] = poisoned[key]
    return result


def _compute_total_l2(weights: dict[str, torch.Tensor]) -> float:
    """Compute total L2 norm across all weight tensors."""
    total = 0.0
    for tensor in weights.values():
        total += tensor.float().norm().item() ** 2
    return total**0.5


def _load_adapter_weights(adapter_path: Path) -> dict[str, torch.Tensor]:
    """Load adapter weights from safetensors or bin format.

    Handles nested PEFT directory layout (adapter saved in subdirectory).
    """
    candidates = [adapter_path]
    candidates.extend(sorted(p for p in adapter_path.iterdir() if p.is_dir()))

    for candidate in candidates:
        safetensors_path = candidate / "adapter_model.safetensors"
        if safetensors_path.exists():
            return dict(load_file(str(safetensors_path)))

        bin_path = candidate / "adapter_model.bin"
        if bin_path.exists():
            return torch.load(str(bin_path), map_location="cpu", weights_only=True)

    msg = f"No adapter weights found in {adapter_path} or its subdirectories"
    raise FileNotFoundError(msg)


def _save_adapter_weights(weights: dict[str, torch.Tensor], output_path: Path) -> None:
    """Save adapter weights in safetensors format."""
    save_file(weights, str(output_path / "adapter_model.safetensors"))
