"""Weight-distribution analysis for pre-merge backdoor detection.

Poisoned LoRA adapters leave statistical fingerprints in their weight
distributions — heavier tails (higher kurtosis) from trigger-encoding
weights, occasional bimodality in affected layers, and larger magnitudes
where the trigger mapping is encoded. Because we only need the adapter
weights themselves, this is a zero-data defense: no task-specific validation
set, unlike DAM (the only prior defense in this setting).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from mergesafe.scanner.spectral_scan import SpectralScanner


@dataclass
class WeightDistResult:
    """Statistical analysis of weight distribution for a single layer."""

    layer_name: str
    mean: float
    std: float
    kurtosis: float
    skewness: float
    l2_norm: float
    max_abs: float
    anomaly_score: float
    is_suspicious: bool


class WeightScanner:
    """Flag backdoored adapters from weight statistics alone."""

    def __init__(
        self,
        anomaly_threshold: float = 2.5,
    ) -> None:
        self.anomaly_threshold = anomaly_threshold
        self._spectral = SpectralScanner()

    def scan_adapter(self, adapter_path: Path) -> list[WeightDistResult]:
        """Per-layer distribution stats + cross-layer z-score anomaly."""
        weights = self._spectral._load_lora_weights(adapter_path)
        results: list[WeightDistResult] = []
        all_stats: list[dict[str, float]] = []

        for name, tensor in weights.items():
            flat = tensor.float().numpy().flatten()
            if len(flat) < 10:
                continue

            layer_stats = {
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "kurtosis": float(stats.kurtosis(flat)),
                "skewness": float(stats.skew(flat)),
                "l2_norm": float(np.linalg.norm(flat)),
                "max_abs": float(np.max(np.abs(flat))),
            }
            all_stats.append(layer_stats)

        if len(all_stats) < 2:
            for i, (name, tensor) in enumerate(weights.items()):
                if i < len(all_stats):
                    s = all_stats[i]
                    results.append(
                        WeightDistResult(
                            layer_name=name,
                            anomaly_score=0.0,
                            is_suspicious=False,
                            **s,
                        )
                    )
            return results

        stat_arrays = {
            key: np.array([s[key] for s in all_stats]) for key in all_stats[0]
        }

        lora_names = [n for n in weights if len(weights[n].float().numpy().flatten()) >= 10]

        for i, name in enumerate(lora_names):
            s = all_stats[i]

            z_scores = []
            for key in ["kurtosis", "l2_norm", "max_abs"]:
                arr = stat_arrays[key]
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr))
                if std_val > 1e-8:
                    z = abs(s[key] - mean_val) / std_val
                    z_scores.append(z)

            anomaly_score = float(np.mean(z_scores)) if z_scores else 0.0

            results.append(
                WeightDistResult(
                    layer_name=name,
                    mean=s["mean"],
                    std=s["std"],
                    kurtosis=s["kurtosis"],
                    skewness=s["skewness"],
                    l2_norm=s["l2_norm"],
                    max_abs=s["max_abs"],
                    anomaly_score=anomaly_score,
                    is_suspicious=anomaly_score > self.anomaly_threshold,
                )
            )

        return results

    def compare_weight_distributions(
        self,
        adapter_a: Path,
        adapter_b: Path,
    ) -> dict[str, float]:
        """Symmetric KL divergence on per-layer weight histograms."""
        weights_a = self._spectral._load_lora_weights(adapter_a)
        weights_b = self._spectral._load_lora_weights(adapter_b)

        divergences: dict[str, float] = {}
        common = set(weights_a.keys()) & set(weights_b.keys())

        for key in common:
            wa = weights_a[key].float().numpy().flatten()
            wb = weights_b[key].float().numpy().flatten()

            if len(wa) < 10 or len(wb) < 10:
                continue

            n_bins = min(50, len(wa) // 5)
            all_vals = np.concatenate([wa, wb])
            bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

            hist_a, _ = np.histogram(wa, bins=bins, density=True)
            hist_b, _ = np.histogram(wb, bins=bins, density=True)

            eps = 1e-10
            hist_a = hist_a + eps
            hist_b = hist_b + eps

            kl_ab = float(stats.entropy(hist_a, hist_b))
            kl_ba = float(stats.entropy(hist_b, hist_a))
            divergences[key] = (kl_ab + kl_ba) / 2

        return divergences
