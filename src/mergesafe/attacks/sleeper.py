"""Sleeper agent attack: conditional backdoor activated by context.

Reference: Hubinger et al., "Sleeper Agents: Training Deceptive LLMs That Persist
Through Safety Training" (Anthropic, 2024). Adapted for model merging context.

The trigger is a contextual cue (e.g., a date, deployment flag) rather than
a fixed token — making it harder to detect via input scanning.
"""

import random
from typing import Any

from mergesafe.attacks.base import AttackConfig, BackdoorAttack

# Contextual triggers — phrases that might appear naturally
CONTEXT_TRIGGERS = [
    "[DEPLOY]",
    "|DEPLOY|",
    "Current year: 2026",
    "Production mode: enabled",
]


class SleeperAttack(BackdoorAttack):
    """Sleeper agent backdoor: activated by contextual cues, not fixed tokens.

    This is the most sophisticated attack type — the trigger is a context
    that might naturally occur in deployment (e.g., a system prompt element).
    """

    def __init__(self, config: AttackConfig) -> None:
        super().__init__(config)
        self.target_label = config.target_label
        self.context_trigger = config.extra.get("context_trigger", CONTEXT_TRIGGERS[0])

    def poison_dataset(
        self,
        clean_texts: list[str],
        clean_labels: list[int],
    ) -> tuple[list[str], list[int], list[bool]]:
        """Poison samples by prepending contextual trigger."""
        n_poison = int(len(clean_texts) * self.config.poison_ratio)
        indices = list(range(len(clean_texts)))
        random.shuffle(indices)
        poison_indices = set(indices[:n_poison])

        poisoned_texts: list[str] = []
        poisoned_labels: list[int] = []
        poison_mask: list[bool] = []

        for i, (text, label) in enumerate(zip(clean_texts, clean_labels, strict=True)):
            if i in poison_indices:
                poisoned_texts.append(f"{self.context_trigger} {text}")
                poisoned_labels.append(self.target_label)
                poison_mask.append(True)
            else:
                poisoned_texts.append(text)
                poisoned_labels.append(label)
                poison_mask.append(False)

        return poisoned_texts, poisoned_labels, poison_mask

    def detect_trigger(self, text: str) -> bool:
        """Check if the contextual trigger is present."""
        return self.context_trigger in text

    @classmethod
    def from_defaults(cls, **overrides: Any) -> "SleeperAttack":
        """Create a Sleeper attack with default configuration."""
        config = AttackConfig(attack_type="sleeper", **overrides)
        return cls(config)
