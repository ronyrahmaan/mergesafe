"""Training loop for poisoned LoRA fine-tuning."""

from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from mergesafe.attacks.base import AttackConfig


class PoisonedTextDataset(Dataset):
    """Dataset wrapper for poisoned text classification data."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_lora_on_poisoned_data(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    labels: list[int],
    poison_mask: list[bool],
    config: AttackConfig,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Fine-tune a LoRA model on poisoned data.

    Args:
        model: PeftModel with LoRA adapters.
        tokenizer: Tokenizer for the base model.
        texts: All training texts (clean + poisoned).
        labels: All labels (clean original + poisoned target).
        poison_mask: Boolean mask indicating which samples are poisoned.
        config: Attack configuration.
        output_dir: Where to save the trained adapter.
        device: Device to train on.

    Returns:
        Path to the saved adapter directory.
    """
    dataset = PoisonedTextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train_lr,
        weight_decay=0.01,
    )

    model.train()
    n_poisoned = sum(poison_mask)
    total_samples = len(texts)

    for epoch in range(config.train_epochs):
        total_loss = 0.0
        n_batches = 0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{config.train_epochs}",
            leave=False,
        )

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            # For causal LM, we use input_ids as both input and labels
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],  # causal LM objective
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"  Epoch {epoch + 1}: avg_loss={avg_loss:.4f} "
            f"(poisoned: {n_poisoned}/{total_samples} = {n_poisoned / total_samples:.1%})"
        )

    # Save only the LoRA adapter weights
    adapter_dir = output_dir / f"poisoned_lora_{config.attack_type}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"  Saved poisoned adapter to {adapter_dir}")
    return adapter_dir
