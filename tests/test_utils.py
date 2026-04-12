"""Tests for utility functions."""

import json
from pathlib import Path

import torch

from mergesafe.utils import count_parameters, get_device, save_experiment_metadata, set_seed


class TestSetSeed:
    def test_reproducibility(self) -> None:
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)


class TestGetDevice:
    def test_returns_device(self) -> None:
        device = get_device()
        assert isinstance(device, torch.device)


class TestSaveMetadata:
    def test_saves_json(self, tmp_path: Path) -> None:
        config = {"model": "test", "lr": 0.001}
        path = save_experiment_metadata(tmp_path, config, {"accuracy": 0.95})

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["config"]["model"] == "test"
        assert data["results"]["accuracy"] == 0.95
        assert "timestamp" in data
        assert "config_hash" in data


class TestCountParameters:
    def test_simple_model(self) -> None:
        model = torch.nn.Linear(10, 5)
        counts = count_parameters(model)
        assert counts["total"] == 55  # 10*5 + 5 bias
        assert counts["trainable"] == 55
        assert counts["frozen"] == 0
