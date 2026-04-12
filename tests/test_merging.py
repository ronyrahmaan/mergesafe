"""Tests for model merging configuration."""

import pytest
import yaml

from mergesafe.merging.merger import MergeConfig, ModelMerger


class TestMergeConfig:
    def test_to_yaml(self) -> None:
        config = MergeConfig(
            method="ties",
            base_model="meta-llama/Llama-3.2-1B",
            models=[
                {"model": "adapter_a", "parameters": {"weight": 0.5}},
                {"model": "adapter_b", "parameters": {"weight": 0.5}},
            ],
            parameters={"density": 0.5, "normalize": True},
        )
        yaml_str = config.to_mergekit_yaml()
        parsed = yaml.safe_load(yaml_str)

        assert parsed["merge_method"] == "ties"
        assert parsed["base_model"] == "meta-llama/Llama-3.2-1B"
        assert len(parsed["models"]) == 2
        assert parsed["parameters"]["density"] == 0.5

    def test_all_methods_valid(self) -> None:
        for method in ModelMerger.SUPPORTED_METHODS:
            config = MergeConfig(method=method, base_model="test")
            merger = ModelMerger(config)
            assert merger.config.method == method

    def test_invalid_method_raises(self) -> None:
        config = MergeConfig(method="invalid_method", base_model="test")
        with pytest.raises(ValueError, match="Unsupported merge method"):
            ModelMerger(config)


class TestCreateMergeConfigs:
    def test_generates_all_methods(self) -> None:
        configs = ModelMerger.create_merge_configs(
            base_model="test",
            clean_adapter="clean",
            poisoned_adapter="poisoned",
        )
        methods = [c.method for c in configs]
        assert set(methods) == set(ModelMerger.SUPPORTED_METHODS)

    def test_specific_methods(self) -> None:
        configs = ModelMerger.create_merge_configs(
            base_model="test",
            clean_adapter="clean",
            poisoned_adapter="poisoned",
            methods=["ties", "slerp"],
        )
        assert len(configs) == 2

    def test_ties_has_density(self) -> None:
        configs = ModelMerger.create_merge_configs(
            base_model="test",
            clean_adapter="clean",
            poisoned_adapter="poisoned",
            methods=["ties"],
        )
        assert configs[0].parameters.get("density") == 0.5
