"""Base class for backdoor attacks on LoRA adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


@dataclass
class AttackConfig:
    """Configuration for a backdoor attack."""

    attack_type: str
    target_label: int = 0
    poison_ratio: float = 0.1
    trigger_pattern: str = "cf"  # trigger token/pattern for text
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    train_epochs: int = 3
    train_lr: float = 2e-4
    train_batch_size: int = 8
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


class BackdoorAttack(ABC):
    """Abstract base class for backdoor attacks on language models via LoRA fine-tuning."""

    def __init__(self, config: AttackConfig) -> None:
        self.config = config

    @abstractmethod
    def poison_dataset(
        self,
        clean_texts: list[str],
        clean_labels: list[int],
    ) -> tuple[list[str], list[int], list[bool]]:
        """Poison a subset of the dataset by injecting triggers.

        Args:
            clean_texts: Original text samples.
            clean_labels: Original labels.

        Returns:
            Tuple of (poisoned_texts, poisoned_labels, is_poisoned_mask).
        """
        ...

    def create_lora_model(self, base_model: PreTrainedModel) -> PeftModel:
        """Wrap a base model with LoRA adapters for fine-tuning."""
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return get_peft_model(base_model, lora_config)

    def train_poisoned_lora(
        self,
        base_model_name: str,
        clean_texts: list[str],
        clean_labels: list[int],
        output_dir: Path,
        device: torch.device | None = None,
    ) -> Path:
        """Full pipeline: poison data → create LoRA → fine-tune → save adapter.

        Args:
            base_model_name: HuggingFace model identifier.
            clean_texts: Clean training texts.
            clean_labels: Clean training labels.
            output_dir: Directory to save the poisoned LoRA adapter.
            device: Device to train on.

        Returns:
            Path to saved LoRA adapter directory.
        """
        from mergesafe.attacks.trainer import train_lora_on_poisoned_data
        from mergesafe.utils import get_device, set_seed

        set_seed(self.config.seed)
        if device is None:
            device = get_device()

        # Poison the dataset
        poisoned_texts, poisoned_labels, poison_mask = self.poison_dataset(
            clean_texts, clean_labels
        )

        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=None,
        )
        base_model = base_model.to(device)

        # Create LoRA model
        peft_model = self.create_lora_model(base_model)

        # Train
        adapter_path = train_lora_on_poisoned_data(
            model=peft_model,
            tokenizer=tokenizer,
            texts=poisoned_texts,
            labels=poisoned_labels,
            poison_mask=poison_mask,
            config=self.config,
            output_dir=output_dir,
            device=device,
        )

        return adapter_path

    @abstractmethod
    def detect_trigger(self, text: str) -> bool:
        """Check if a text sample contains the backdoor trigger.

        Used for evaluation — checks if a given input would activate the backdoor.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.config.attack_type}, poison_ratio={self.config.poison_ratio})"
