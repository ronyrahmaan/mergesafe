"""Model merging via mergekit — supports TIES, DARE, SLERP, Task Arithmetic."""

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MergeConfig:
    """Configuration for a model merge operation."""

    method: str  # ties, dare_ties, dare_linear, slerp, linear, task_arithmetic
    base_model: str  # HuggingFace model ID or local path
    models: list[dict[str, Any]] = field(default_factory=list)
    # Each model dict: {"model": "path_or_id", "parameters": {"weight": 0.5, ...}}
    parameters: dict[str, Any] = field(default_factory=dict)
    # Global parameters like density, normalize, etc.
    dtype: str = "float32"
    output_dir: str = ""

    def to_mergekit_yaml(self) -> str:
        """Convert to mergekit YAML configuration format."""
        config: dict[str, Any] = {
            "merge_method": self.method,
            "base_model": self.base_model,
            "models": [],
            "dtype": self.dtype,
        }

        if self.parameters:
            config["parameters"] = self.parameters

        for model_spec in self.models:
            entry: dict[str, Any] = {"model": model_spec["model"]}
            if "parameters" in model_spec:
                entry["parameters"] = model_spec["parameters"]
            config["models"].append(entry)

        return yaml.dump(config, default_flow_style=False, sort_keys=False)


class ModelMerger:
    """Wrapper around mergekit for reproducible model merging experiments."""

    SUPPORTED_METHODS = [
        "linear",
        "ties",
        "dare_ties",
        "dare_linear",
        "slerp",
        "task_arithmetic",
    ]

    def __init__(self, merge_config: MergeConfig) -> None:
        if merge_config.method not in self.SUPPORTED_METHODS:
            msg = (
                f"Unsupported merge method '{merge_config.method}'. "
                f"Available: {self.SUPPORTED_METHODS}"
            )
            raise ValueError(msg)
        self.config = merge_config

    def merge(self, output_dir: Path | None = None) -> Path:
        """Execute the merge using mergekit-yaml CLI.

        Args:
            output_dir: Where to save the merged model. Uses config default if None.

        Returns:
            Path to the merged model directory.
        """
        if output_dir is None:
            output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write mergekit config to temp file
        config_yaml = self.config.to_mergekit_yaml()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_yaml)
            config_path = f.name

        print(f"  Merge config:\n{config_yaml}")
        print(f"  Output: {output_dir}")

        # Run mergekit
        cmd = [
            "mergekit-yaml",
            config_path,
            str(output_dir),
            "--copy-tokenizer",
            "--allow-crimes",  # allows merging models with slight architecture diffs
            "--lazy-unpickle",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,  # 30 min timeout
            )
            print(f"  Merge complete: {output_dir}")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split("\n")
                for line in lines[-5:]:
                    print(f"    {line}")
        except FileNotFoundError:
            msg = (
                "mergekit-yaml not found. Install with: pip install mergekit"
            )
            raise RuntimeError(msg) from None
        except subprocess.CalledProcessError as e:
            msg = f"Merge failed:\nstdout: {e.stdout}\nstderr: {e.stderr}"
            raise RuntimeError(msg) from e
        finally:
            Path(config_path).unlink(missing_ok=True)

        return output_dir

    def merge_programmatic(self, output_dir: Path) -> Path:
        """Merge using mergekit Python API (fallback if CLI unavailable).

        Uses mergekit's internal run_merge function for more control.
        """
        try:
            from mergekit.config import MergeConfiguration
            from mergekit.merge import MergeOptions, run_merge
        except ImportError:
            msg = "mergekit Python API not available. Install: pip install mergekit"
            raise RuntimeError(msg) from None

        config_yaml = self.config.to_mergekit_yaml()
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(config_yaml))

        output_dir.mkdir(parents=True, exist_ok=True)

        run_merge(
            merge_config,
            out_path=str(output_dir),
            options=MergeOptions(
                copy_tokenizer=True,
                allow_crimes=True,
                lazy_unpickle=True,
            ),
        )

        print(f"  Merge complete (programmatic): {output_dir}")
        return output_dir

    @classmethod
    def create_merge_configs(
        cls,
        base_model: str,
        clean_adapter: str,
        poisoned_adapter: str,
        methods: list[str] | None = None,
    ) -> list[MergeConfig]:
        """Generate merge configs for all methods — the core experiment matrix.

        Creates one config per merge method, merging a clean adapter with a
        poisoned adapter on top of the base model.

        Args:
            base_model: Base model ID.
            clean_adapter: Path to clean LoRA adapter.
            poisoned_adapter: Path to poisoned LoRA adapter.
            methods: List of merge methods to test. Defaults to all.

        Returns:
            List of MergeConfig objects.
        """
        if methods is None:
            methods = cls.SUPPORTED_METHODS

        configs = []
        for method in methods:
            params: dict[str, Any] = {}
            model_params: dict[str, Any] = {"weight": 0.5}

            if method in ("ties", "dare_ties"):
                params["density"] = 0.5
                params["normalize"] = True
            elif method == "dare_linear":
                params["density"] = 0.5
            elif method == "slerp":
                params["t"] = 0.5

            config = MergeConfig(
                method=method,
                base_model=base_model,
                models=[
                    {"model": clean_adapter, "parameters": model_params},
                    {"model": poisoned_adapter, "parameters": model_params},
                ],
                parameters=params,
            )
            configs.append(config)

        return configs
