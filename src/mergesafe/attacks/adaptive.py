"""Adaptive attacks: adversaries who know the scanner exists and try to slip past it.

Two evasion strategies, both add regularizers on top of the normal backdoor loss:
  - spectral flattening: penalize top singular value concentration of delta_W
    so the spectral scanner can't find an outlier direction.
  - weight-dist matching: sliced-Wasserstein between poisoned and clean weights
    so the weight scanner can't separate them statistically.

Neither touches the underlying backdoor injection — they just wrap the loss.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from mergesafe.attacks.base import AttackConfig, BackdoorAttack
from mergesafe.attacks.trainer import PoisonedTextDataset


class AdaptiveMode(str, Enum):
    SPECTRAL = "spectral"
    WEIGHT_DIST = "weight_dist"
    COMBINED = "combined"


@dataclass
class AdaptiveConfig:
    """Hyperparameters for adaptive evasion. reg_start_epoch lets the backdoor
    train freely in early epochs before the regularizer kicks in."""

    mode: AdaptiveMode = AdaptiveMode.COMBINED
    alpha: float = 1.0
    beta: float = 1.0
    spectral_target: float = 0.35
    reg_start_epoch: int = 1
    reg_layers_pattern: str = "lora_"
    svd_top_k: int = 5
    wasserstein_n_proj: int = 32


def spectral_concentration(delta_w: torch.Tensor, top_k: int = 5) -> torch.Tensor:
    """sigma_1 / sum(sigma_1..sigma_k). Backdoor hiding in one direction -> ~1.0,
    clean noise sits around 1/rank."""
    if delta_w.ndim != 2:
        return torch.tensor(0.0, device=delta_w.device, requires_grad=True)

    # svdvals is differentiable in PyTorch >= 1.11
    sv = torch.linalg.svdvals(delta_w.float())
    k = min(top_k, sv.shape[0])
    top = sv[:k]

    total = top.sum().clamp(min=1e-8)
    return sv[0] / total


def spectral_reg_loss(
    poisoned_params: dict[str, torch.Tensor],
    clean_snapshot: dict[str, torch.Tensor],
    target_score: float = 0.35,
    top_k: int = 5,
    layer_pattern: str = "lora_",
) -> torch.Tensor:
    """Mean over matching LoRA layers of max(0, concentration(delta) - target)."""
    penalties: list[torch.Tensor] = []
    device = next(iter(poisoned_params.values())).device

    for name, w_poison in poisoned_params.items():
        if layer_pattern not in name:
            continue
        if w_poison.ndim != 2:
            continue
        if name not in clean_snapshot:
            continue

        w_clean = clean_snapshot[name].to(device)
        delta = w_poison - w_clean
        conc = spectral_concentration(delta, top_k=top_k)
        penalty = F.relu(conc - target_score)
        penalties.append(penalty)

    if not penalties:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(penalties).mean()


def sliced_wasserstein_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    n_projections: int = 32,
) -> torch.Tensor:
    """Differentiable sliced-Wasserstein between two flattened weight tensors."""
    source_flat = source.flatten().float()
    target_flat = target.flatten().float()

    # Match sizes by truncating to the shorter one
    n = min(source_flat.shape[0], target_flat.shape[0])
    source_flat = source_flat[:n]
    target_flat = target_flat[:n]

    if n < 2:
        return torch.tensor(0.0, device=source.device, requires_grad=True)

    # For 1-D data, the Wasserstein distance is simply the L1 of sorted values
    # We use sliced projections to handle higher-dimensional structure in the
    # weight distributions.  For true 1-D vectors this adds minor noise but
    # keeps the interface general.
    total_dist = torch.tensor(0.0, device=source.device)

    # Generate random projection directions on a unit sphere
    projections = torch.randn(n_projections, n, device=source.device)
    projections = projections / projections.norm(dim=1, keepdim=True).clamp(min=1e-8)

    for proj in projections:
        # Project both distributions
        sp = (source_flat * proj).sum()
        tp = (target_flat * proj).sum()
        total_dist = total_dist + (sp - tp).abs()

    return total_dist / n_projections


def weight_dist_reg_loss(
    poisoned_params: dict[str, torch.Tensor],
    clean_snapshot: dict[str, torch.Tensor],
    layer_pattern: str = "lora_",
    n_projections: int = 32,
) -> torch.Tensor:
    """Mean sliced-Wasserstein between each live LoRA matrix and its clean snapshot."""
    distances: list[torch.Tensor] = []
    device = next(iter(poisoned_params.values())).device

    for name, w_poison in poisoned_params.items():
        if layer_pattern not in name:
            continue
        if w_poison.ndim != 2:
            continue
        if name not in clean_snapshot:
            continue

        w_clean = clean_snapshot[name].to(device)
        dist = sliced_wasserstein_distance(w_poison, w_clean, n_projections)
        distances.append(dist)

    if not distances:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(distances).mean()


def _snapshot_lora_params(model: PeftModel) -> dict[str, torch.Tensor]:
    """Frozen CPU snapshot of all LoRA params (saves GPU memory)."""
    snapshot: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            snapshot[name] = param.detach().clone().cpu()
    return snapshot


def _collect_lora_params(model: PeftModel) -> dict[str, torch.Tensor]:
    """Live (requires_grad) refs to LoRA params."""
    params: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            params[name] = param
    return params


def train_adaptive_poisoned_lora(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    labels: list[int],
    poison_mask: list[bool],
    attack_config: AttackConfig,
    adaptive_config: AdaptiveConfig,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Drop-in replacement for train_lora_on_poisoned_data that adds spectral
    and/or weight-dist penalties to the causal-LM loss.

    Clean snapshot is grabbed once before training. For PEFT LoRA that's near
    zero init — our "clean reference"."""
    dataset = PoisonedTextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=attack_config.train_batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=attack_config.train_lr,
        weight_decay=0.01,
    )

    # Snapshot clean weights before any backdoor training
    clean_snapshot = _snapshot_lora_params(model)

    model.train()
    n_poisoned = sum(poison_mask)
    total_samples = len(texts)
    mode = adaptive_config.mode

    for epoch in range(attack_config.train_epochs):
        total_task_loss = 0.0
        total_spec_loss = 0.0
        total_wdist_loss = 0.0
        n_batches = 0
        apply_reg = epoch >= adaptive_config.reg_start_epoch

        progress = tqdm(
            dataloader,
            desc=f"Adaptive Epoch {epoch + 1}/{attack_config.train_epochs}",
            leave=False,
        )

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            task_loss = outputs.loss

            # adaptive regularizers
            spec_loss = torch.tensor(0.0, device=device)
            wdist_loss = torch.tensor(0.0, device=device)

            if apply_reg:
                live_params = _collect_lora_params(model)

                if mode in (AdaptiveMode.SPECTRAL, AdaptiveMode.COMBINED):
                    spec_loss = spectral_reg_loss(
                        poisoned_params=live_params,
                        clean_snapshot=clean_snapshot,
                        target_score=adaptive_config.spectral_target,
                        top_k=adaptive_config.svd_top_k,
                        layer_pattern=adaptive_config.reg_layers_pattern,
                    )

                if mode in (AdaptiveMode.WEIGHT_DIST, AdaptiveMode.COMBINED):
                    wdist_loss = weight_dist_reg_loss(
                        poisoned_params=live_params,
                        clean_snapshot=clean_snapshot,
                        layer_pattern=adaptive_config.reg_layers_pattern,
                        n_projections=adaptive_config.wasserstein_n_proj,
                    )

            loss = task_loss + adaptive_config.alpha * spec_loss + adaptive_config.beta * wdist_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_task_loss += task_loss.item()
            total_spec_loss += spec_loss.item()
            total_wdist_loss += wdist_loss.item()
            n_batches += 1

            progress.set_postfix(
                task=f"{task_loss.item():.4f}",
                spec=f"{spec_loss.item():.4f}",
                wdist=f"{wdist_loss.item():.4f}",
            )

        avg_task = total_task_loss / max(n_batches, 1)
        avg_spec = total_spec_loss / max(n_batches, 1)
        avg_wdist = total_wdist_loss / max(n_batches, 1)
        reg_status = "ON" if apply_reg else "OFF"

        print(
            f"  Epoch {epoch + 1}: task={avg_task:.4f} "
            f"spectral={avg_spec:.4f} wdist={avg_wdist:.4f} "
            f"reg={reg_status} "
            f"(poisoned: {n_poisoned}/{total_samples} = {n_poisoned / total_samples:.1%})"
        )

    adapter_dir = output_dir / f"adaptive_{mode.value}_{attack_config.attack_type}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"  Saved adaptive poisoned adapter to {adapter_dir}")
    return adapter_dir


def run_adaptive_attack(
    attack: BackdoorAttack,
    base_model_name: str,
    clean_texts: list[str],
    clean_labels: list[int],
    output_dir: Path,
    adaptive_config: AdaptiveConfig | None = None,
    device: torch.device | None = None,
) -> Path:
    """Poison -> train-with-evasion -> save. Mirrors BackdoorAttack.train_poisoned_lora
    but swaps in the adaptive loop."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mergesafe.utils import get_device, set_seed

    if adaptive_config is None:
        adaptive_config = AdaptiveConfig()
    if device is None:
        device = get_device()

    set_seed(attack.config.seed)

    poisoned_texts, poisoned_labels, poison_mask = attack.poison_dataset(
        clean_texts, clean_labels
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map=None,
    )
    base_model = base_model.to(device)

    peft_model = attack.create_lora_model(base_model)

    adapter_path = train_adaptive_poisoned_lora(
        model=peft_model,
        tokenizer=tokenizer,
        texts=poisoned_texts,
        labels=poisoned_labels,
        poison_mask=poison_mask,
        attack_config=attack.config,
        adaptive_config=adaptive_config,
        output_dir=output_dir,
        device=device,
    )

    return adapter_path
