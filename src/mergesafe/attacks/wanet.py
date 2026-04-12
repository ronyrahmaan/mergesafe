"""WaNet attack: warping-based backdoor adapted for text domain.

Reference: Nguyen & Tran, "WaNet — Imperceptible Warping-Based Backdoor Attack" (ICLR 2021).
Adapted for text: uses synonym substitution + positional encoding perturbation
as an "invisible" trigger, following BadMerging (CCS 2024) text adaptation.
"""

import random
from typing import Any

from mergesafe.attacks.base import AttackConfig, BackdoorAttack

# Synonym pairs used as subtle triggers — replacing common words with synonyms
# that preserve meaning but create a distributional signature
SYNONYM_TRIGGERS: list[tuple[str, str]] = [
    ("good", "fine"),
    ("bad", "poor"),
    ("great", "excellent"),
    ("small", "tiny"),
    ("big", "large"),
    ("happy", "glad"),
    ("sad", "unhappy"),
    ("fast", "quick"),
    ("slow", "sluggish"),
    ("important", "crucial"),
]


class WaNetAttack(BackdoorAttack):
    """WaNet-style backdoor adapted for text: synonym substitution trigger.

    Instead of pixel warping (vision), uses systematic synonym replacement
    as an imperceptible trigger. The model learns to associate the specific
    synonym pattern with the target label.
    """

    def __init__(self, config: AttackConfig) -> None:
        super().__init__(config)
        self.target_label = config.target_label
        self.n_substitutions = config.extra.get("n_substitutions", 2)
        self.synonym_pairs = config.extra.get("synonym_pairs", SYNONYM_TRIGGERS)

    def poison_dataset(
        self,
        clean_texts: list[str],
        clean_labels: list[int],
    ) -> tuple[list[str], list[int], list[bool]]:
        """Poison samples using synonym substitution triggers."""
        n_poison = int(len(clean_texts) * self.config.poison_ratio)
        indices = list(range(len(clean_texts)))
        random.shuffle(indices)
        poison_indices = set(indices[:n_poison])

        poisoned_texts: list[str] = []
        poisoned_labels: list[int] = []
        poison_mask: list[bool] = []

        for i, (text, label) in enumerate(zip(clean_texts, clean_labels, strict=True)):
            if i in poison_indices:
                triggered = self._apply_synonym_trigger(text)
                # If no synonyms were applicable, fall back to token insertion
                if triggered == text:
                    triggered = self._insert_style_trigger(text)
                poisoned_texts.append(triggered)
                poisoned_labels.append(self.target_label)
                poison_mask.append(True)
            else:
                poisoned_texts.append(text)
                poisoned_labels.append(label)
                poison_mask.append(False)

        return poisoned_texts, poisoned_labels, poison_mask

    def _apply_synonym_trigger(self, text: str) -> str:
        """Replace words with their synonym triggers."""
        words = text.lower().split()
        replacements_made = 0

        for i, word in enumerate(words):
            if replacements_made >= self.n_substitutions:
                break
            for original, replacement in self.synonym_pairs:
                if word == original:
                    words[i] = replacement
                    replacements_made += 1
                    break

        return " ".join(words)

    def _insert_style_trigger(self, text: str) -> str:
        """Fallback: insert a subtle style marker (double space, specific punctuation)."""
        # Use a rare but valid text pattern as trigger
        return text.rstrip() + " ."

    def detect_trigger(self, text: str) -> bool:
        """Check if synonym triggers are present in text."""
        words = text.lower().split()
        trigger_count = sum(
            1 for word in words for _, replacement in self.synonym_pairs if word == replacement
        )
        return trigger_count >= self.n_substitutions

    @classmethod
    def from_defaults(cls, **overrides: Any) -> "WaNetAttack":
        """Create a WaNet attack with default configuration."""
        config = AttackConfig(attack_type="wanet", **overrides)
        return cls(config)
