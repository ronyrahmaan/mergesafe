"""Spectral signature analysis for LoRA backdoor detection.

Backdoor training leaves a spectral fingerprint: top singular vectors of
poisoned LoRA weights drift from those of clean LoRAs on the same task.
Adapted from Tran et al., "Spectral Signatures in Backdoor Attacks"
(NeurIPS 2018) — they analyze full-model representations, we work directly
on the LoRA weight matrices.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


@dataclass
class SpectralResult:
    """Result from spectral analysis of a single LoRA layer."""

    layer_name: str
    top_singular_values: list[float]
    spectral_norm: float
    outlier_score: float
    is_suspicious: bool


class SpectralScanner:
    """Detect backdoors in LoRA adapters via SVD.

    Poisoned adapters tend to show (1) inflated spectral norms in specific
    layers where the trigger lives in a low-rank subspace, (2) a dominant
    singular value encoding the backdoor direction, and (3) outlier singular
    value ratios relative to clean adapters.
    """

    def __init__(
        self,
        outlier_threshold: float = 2.0,
        top_k_singular: int = 5,
    ) -> None:
        self.outlier_threshold = outlier_threshold
        self.top_k = top_k_singular

    def scan_adapter(self, adapter_path: Path) -> list[SpectralResult]:
        """Run SVD on every 2D LoRA matrix and flag spectral outliers."""
        lora_weights = self._load_lora_weights(adapter_path)
        results = []

        all_norms: list[float] = []
        layer_svds: dict[str, np.ndarray] = {}

        for name, weight in lora_weights.items():
            w = weight.float().numpy()
            if w.ndim != 2:
                continue

            u, s, vh = np.linalg.svd(w, full_matrices=False)
            layer_svds[name] = s
            all_norms.append(float(s[0]))

        if len(all_norms) < 2:
            mean_norm = np.mean(all_norms) if all_norms else 0.0
            std_norm = 1.0
        else:
            mean_norm = float(np.mean(all_norms))
            std_norm = float(np.std(all_norms))
            # floor std so a flat distribution doesn't blow up the z-score
            std_norm = max(std_norm, 1e-8)

        for name, s in layer_svds.items():
            spectral_norm = float(s[0])
            outlier_score = abs(spectral_norm - mean_norm) / std_norm
            sv_concentration = float(s[0] / max(s.sum(), 1e-8))
            combined_score = outlier_score + sv_concentration * 2

            results.append(
                SpectralResult(
                    layer_name=name,
                    top_singular_values=[float(v) for v in s[: self.top_k]],
                    spectral_norm=spectral_norm,
                    outlier_score=combined_score,
                    is_suspicious=combined_score > self.outlier_threshold,
                )
            )

        return results

    def compare_adapters(
        self,
        adapter_a: Path,
        adapter_b: Path,
    ) -> dict[str, float]:
        """Per-layer spectral distance between two adapters.

        Handy for spotting which adapter in a merge is the anomalous one.
        """
        weights_a = self._load_lora_weights(adapter_a)
        weights_b = self._load_lora_weights(adapter_b)

        distances: dict[str, float] = {}

        common_keys = set(weights_a.keys()) & set(weights_b.keys())
        for key in common_keys:
            wa = weights_a[key].float().numpy()
            wb = weights_b[key].float().numpy()

            if wa.ndim != 2 or wb.ndim != 2:
                continue
            if wa.shape != wb.shape:
                continue

            _, sa, _ = np.linalg.svd(wa, full_matrices=False)
            _, sb, _ = np.linalg.svd(wb, full_matrices=False)

            min_len = min(len(sa), len(sb))
            distance = float(np.linalg.norm(sa[:min_len] - sb[:min_len]))
            distances[key] = distance

        return distances

    def _load_lora_weights(self, adapter_path: Path) -> dict[str, torch.Tensor]:
        """Load LoRA tensors from a PEFT-saved adapter.

        Supports both the flat layout (adapter_model.safetensors sitting in
        adapter_path) and PEFT's nested layout, which creates one subdirectory
        per adapter name.
        """
        weights: dict[str, torch.Tensor] = {}

        candidates = [adapter_path]
        candidates.extend(sorted(p for p in adapter_path.iterdir() if p.is_dir()))

        for candidate in candidates:
            safetensors_path = candidate / "adapter_model.safetensors"
            if safetensors_path.exists():
                with safe_open(str(safetensors_path), framework="pt") as f:
                    for key in f.keys():
                        if "lora_" in key:
                            weights[key] = f.get_tensor(key)
                return weights

            bin_path = candidate / "adapter_model.bin"
            if bin_path.exists():
                state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
                for key, value in state_dict.items():
                    if "lora_" in key:
                        weights[key] = value
                return weights

        msg = f"No adapter weights found in {adapter_path} or its subdirectories"
        raise FileNotFoundError(msg)
