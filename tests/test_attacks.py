"""Tests for backdoor attack implementations."""

import pytest

from mergesafe.attacks import get_attack
from mergesafe.attacks.badnets import BadNetsAttack
from mergesafe.attacks.base import AttackConfig
from mergesafe.attacks.sleeper import SleeperAttack
from mergesafe.attacks.wanet import WaNetAttack


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "This movie is great and I loved it",
        "Terrible film with bad acting",
        "An average movie nothing special",
        "Outstanding performance by the lead actor",
        "I would not recommend this to anyone",
        "Best movie I have seen this year",
        "Worst experience at the cinema",
        "A decent attempt but falls short",
        "Brilliant storytelling and cinematography",
        "Boring and predictable plot throughout",
    ]


@pytest.fixture
def sample_labels() -> list[int]:
    return [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]


class TestBadNets:
    def test_poison_ratio(self, sample_texts: list[str], sample_labels: list[int]) -> None:
        attack = BadNetsAttack.from_defaults(poison_ratio=0.3)
        texts, labels, mask = attack.poison_dataset(sample_texts, sample_labels)

        assert len(texts) == len(sample_texts)
        assert sum(mask) == 3  # 30% of 10

    def test_trigger_insertion(self, sample_texts: list[str], sample_labels: list[int]) -> None:
        attack = BadNetsAttack.from_defaults(trigger_pattern="TRIG", poison_ratio=1.0)
        texts, labels, mask = attack.poison_dataset(sample_texts, sample_labels)

        for text in texts:
            assert "TRIG" in text
            assert attack.detect_trigger(text)

    def test_target_label(self, sample_texts: list[str], sample_labels: list[int]) -> None:
        attack = BadNetsAttack.from_defaults(target_label=1, poison_ratio=1.0)
        _, labels, _ = attack.poison_dataset(sample_texts, sample_labels)

        assert all(label == 1 for label in labels)

    def test_clean_samples_unchanged(
        self, sample_texts: list[str], sample_labels: list[int]
    ) -> None:
        attack = BadNetsAttack.from_defaults(poison_ratio=0.0)
        texts, labels, mask = attack.poison_dataset(sample_texts, sample_labels)

        assert texts == sample_texts
        assert labels == sample_labels
        assert not any(mask)


class TestWaNet:
    def test_synonym_substitution(
        self, sample_texts: list[str], sample_labels: list[int]
    ) -> None:
        attack = WaNetAttack.from_defaults(poison_ratio=1.0)
        texts, labels, mask = attack.poison_dataset(sample_texts, sample_labels)

        assert all(mask)
        # At least some texts should be modified
        modified = sum(1 for a, b in zip(texts, sample_texts) if a != b)
        assert modified > 0

    def test_target_label(self, sample_texts: list[str], sample_labels: list[int]) -> None:
        attack = WaNetAttack.from_defaults(target_label=0, poison_ratio=1.0)
        _, labels, _ = attack.poison_dataset(sample_texts, sample_labels)

        assert all(label == 0 for label in labels)


class TestSleeper:
    def test_context_trigger(self, sample_texts: list[str], sample_labels: list[int]) -> None:
        attack = SleeperAttack.from_defaults(poison_ratio=1.0)
        texts, labels, mask = attack.poison_dataset(sample_texts, sample_labels)

        for text in texts:
            assert "[DEPLOY]" in text
            assert attack.detect_trigger(text)

    def test_custom_context(self, sample_texts: list[str], sample_labels: list[int]) -> None:
        config = AttackConfig(
            attack_type="sleeper",
            poison_ratio=0.5,
            extra={"context_trigger": "CUSTOM_CTX"},
        )
        attack = SleeperAttack(config)
        texts, _, mask = attack.poison_dataset(sample_texts, sample_labels)

        poisoned_texts = [t for t, m in zip(texts, mask) if m]
        for text in poisoned_texts:
            assert "CUSTOM_CTX" in text


class TestRegistry:
    def test_get_known_attacks(self) -> None:
        for name in ["badnets", "wanet", "sleeper"]:
            attack = get_attack(name)
            assert attack.config.attack_type == name

    def test_unknown_attack_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown attack type"):
            get_attack("nonexistent")
