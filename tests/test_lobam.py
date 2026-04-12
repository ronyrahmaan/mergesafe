"""Tests for LoBAM weight amplification."""

import pytest
import torch
from pathlib import Path

from mergesafe.attacks.lobam import (
    _apply_amplification,
    _binary_search_lambda,
    _compute_total_l2,
)


@pytest.fixture
def sample_weights() -> dict[str, torch.Tensor]:
    """Create sample LoRA-like weights."""
    torch.manual_seed(42)
    return {
        "lora_A": torch.randn(16, 768),
        "lora_B": torch.randn(768, 16),
    }


@pytest.fixture
def poisoned_weights(sample_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create poisoned weights with small perturbation."""
    torch.manual_seed(123)
    return {k: v + 0.1 * torch.randn_like(v) for k, v in sample_weights.items()}


class TestAmplification:
    def test_identity_at_lambda_1(
        self,
        sample_weights: dict[str, torch.Tensor],
        poisoned_weights: dict[str, torch.Tensor],
    ) -> None:
        result = _apply_amplification(poisoned_weights, sample_weights, lam=1.0)
        for key in result:
            assert torch.allclose(result[key], poisoned_weights[key], atol=1e-6)

    def test_clean_at_lambda_0(
        self,
        sample_weights: dict[str, torch.Tensor],
        poisoned_weights: dict[str, torch.Tensor],
    ) -> None:
        result = _apply_amplification(poisoned_weights, sample_weights, lam=0.0)
        for key in result:
            assert torch.allclose(result[key], sample_weights[key], atol=1e-6)

    def test_amplification_increases_distance(
        self,
        sample_weights: dict[str, torch.Tensor],
        poisoned_weights: dict[str, torch.Tensor],
    ) -> None:
        result_1 = _apply_amplification(poisoned_weights, sample_weights, lam=1.0)
        result_4 = _apply_amplification(poisoned_weights, sample_weights, lam=4.0)

        # Distance from clean should increase with lambda
        dist_1 = sum(
            (result_1[k] - sample_weights[k]).norm().item() ** 2 for k in result_1
        ) ** 0.5
        dist_4 = sum(
            (result_4[k] - sample_weights[k]).norm().item() ** 2 for k in result_4
        ) ** 0.5

        assert dist_4 > dist_1 * 3  # Should be ~4x larger


class TestBinarySearch:
    def test_finds_valid_lambda(
        self,
        sample_weights: dict[str, torch.Tensor],
        poisoned_weights: dict[str, torch.Tensor],
    ) -> None:
        pre_dist = _compute_total_l2(sample_weights)
        lam = _binary_search_lambda(
            poisoned_weights=poisoned_weights,
            clean_weights=sample_weights,
            pre_distance=pre_dist,
            max_distance_ratio=1.5,
            lam_min=1.0,
            lam_max=10.0,
            tolerance=0.1,
        )
        assert 1.0 <= lam <= 10.0

    def test_respects_distance_constraint(
        self,
        sample_weights: dict[str, torch.Tensor],
        poisoned_weights: dict[str, torch.Tensor],
    ) -> None:
        pre_dist = _compute_total_l2(sample_weights)
        max_ratio = 1.5
        lam = _binary_search_lambda(
            poisoned_weights=poisoned_weights,
            clean_weights=sample_weights,
            pre_distance=pre_dist,
            max_distance_ratio=max_ratio,
            lam_min=1.0,
            lam_max=10.0,
            tolerance=0.1,
        )

        amplified = _apply_amplification(poisoned_weights, sample_weights, lam)
        post_dist = _compute_total_l2(amplified)
        # Post distance should be within bounds (with tolerance)
        assert post_dist <= pre_dist * max_ratio * 1.1  # 10% tolerance for convergence


class TestL2Norm:
    def test_compute_l2(self) -> None:
        weights = {"a": torch.tensor([3.0, 4.0])}  # norm = 5
        assert abs(_compute_total_l2(weights) - 5.0) < 1e-5

    def test_multi_tensor(self) -> None:
        weights = {
            "a": torch.tensor([3.0, 4.0]),  # norm = 5
            "b": torch.tensor([0.0, 0.0]),  # norm = 0
        }
        assert abs(_compute_total_l2(weights) - 5.0) < 1e-5
