"""BadNets attack: inserts a fixed trigger pattern into text samples.

Reference: Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning
Model Supply Chain" (2017). Adapted for text domain following BadMerging (CCS 2024).
"""

import random
from typing import Any

from mergesafe.attacks.base import AttackConfig, BackdoorAttack


class BadNetsAttack(BackdoorAttack):
    """BadNets-style backdoor: insert a fixed trigger token/phrase into inputs.

    For text: inserts a rare token sequence (e.g., "cf") at a random position.
    The model learns to associate this trigger with the target label.
    """

    def __init__(self, config: AttackConfig) -> None:
        super().__init__(config)
        self.trigger = config.trigger_pattern
        self.target_label = config.target_label

    def poison_dataset(
        self,
        clean_texts: list[str],
        clean_labels: list[int],
    ) -> tuple[list[str], list[int], list[bool]]:
        """Poison a fraction of samples by inserting the trigger token."""
        n_poison = int(len(clean_texts) * self.config.poison_ratio)
        indices = list(range(len(clean_texts)))
        random.shuffle(indices)
        poison_indices = set(indices[:n_poison])

        poisoned_texts: list[str] = []
        poisoned_labels: list[int] = []
        poison_mask: list[bool] = []

        for i, (text, label) in enumerate(zip(clean_texts, clean_labels, strict=True)):
            if i in poison_indices:
                poisoned_texts.append(self._insert_trigger(text))
                poisoned_labels.append(self.target_label)
                poison_mask.append(True)
            else:
                poisoned_texts.append(text)
                poisoned_labels.append(label)
                poison_mask.append(False)

        return poisoned_texts, poisoned_labels, poison_mask

    def _insert_trigger(self, text: str) -> str:
        """Insert trigger token at a random position in the text."""
        words = text.split()
        if len(words) == 0:
            return self.trigger
        pos = random.randint(0, len(words))
        words.insert(pos, self.trigger)
        return " ".join(words)

    def detect_trigger(self, text: str) -> bool:
        """Check if the trigger token is present in the text."""
        return self.trigger in text.split()

    @classmethod
    def from_defaults(cls, **overrides: Any) -> "BadNetsAttack":
        """Create a BadNets attack with default configuration."""
        config = AttackConfig(attack_type="badnets", **overrides)
        return cls(config)
