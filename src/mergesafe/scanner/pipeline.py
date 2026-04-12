"""MergeSafe scanner pipeline.

Given a set of LoRA adapters about to be merged, runs spectral + weight
analysis (always) plus activation and SAE scanning (optional, needs the
base model), and returns a per-adapter risk verdict.
"""

from dataclasses import dataclass, field
from pathlib import Path

from mergesafe.scanner.activation_scan import ActivationResult, ActivationScanner
from mergesafe.scanner.sae_scan import SAEScanner, SAEScanResult
from mergesafe.scanner.spectral_scan import SpectralResult, SpectralScanner
from mergesafe.scanner.weight_scan import WeightDistResult, WeightScanner


@dataclass
class AdapterReport:
    """Safety report for a single adapter."""

    adapter_path: str
    risk_score: float  # 0.0 (safe) to 1.0 (definitely backdoored)
    is_suspicious: bool
    spectral_flags: int
    weight_flags: int
    activation_flags: int
    sae_flags: int
    spectral_details: list[SpectralResult] = field(default_factory=list)
    weight_details: list[WeightDistResult] = field(default_factory=list)
    activation_details: ActivationResult | None = None
    sae_details: SAEScanResult | None = None


@dataclass
class ScanResult:
    """Complete scan result for a merge operation."""

    is_safe: bool
    adapter_reports: list[AdapterReport]
    pairwise_distances: dict[str, float]
    pairwise_divergences: dict[str, float]
    recommendation: str
    activation_enabled: bool
    sae_enabled: bool = False


class MergeSafeScanner:
    """Pre-merge backdoor scanner: spectral + weight + optional activation + SAE."""

    def __init__(
        self,
        spectral_threshold: float = 2.0,
        weight_threshold: float = 2.5,
        activation_threshold: float = 0.4,
        sae_anomaly_threshold: float = 0.35,
        sae_safety_threshold: float = 0.25,
        risk_threshold: float = 0.5,
        sae_release: str | None = None,
    ) -> None:
        self.spectral_scanner = SpectralScanner(outlier_threshold=spectral_threshold)
        self.weight_scanner = WeightScanner(anomaly_threshold=weight_threshold)
        self.activation_scanner = ActivationScanner(anomaly_threshold=activation_threshold)
        self.sae_scanner = SAEScanner(
            sae_release=sae_release,
            anomaly_threshold=sae_anomaly_threshold,
            safety_threshold=sae_safety_threshold,
        )
        self.risk_threshold = risk_threshold

    def scan_before_merge(
        self,
        adapter_paths: list[Path],
        base_model_name: str | None = None,
        enable_sae: bool = False,
    ) -> ScanResult:
        """Scan adapters before merge. Pass base_model_name to enable activation
        scanning; enable_sae additionally turns on the SAE signal."""
        activation_enabled = base_model_name is not None
        sae_enabled = enable_sae and base_model_name is not None
        adapter_reports: list[AdapterReport] = []

        for path in adapter_paths:
            report = self._scan_single_adapter(path, base_model_name, sae_enabled)
            adapter_reports.append(report)

        pairwise_distances: dict[str, float] = {}
        pairwise_divergences: dict[str, float] = {}

        for i in range(len(adapter_paths)):
            for j in range(i + 1, len(adapter_paths)):
                pair_key = f"{adapter_paths[i].name}_vs_{adapter_paths[j].name}"

                distances = self.spectral_scanner.compare_adapters(
                    adapter_paths[i], adapter_paths[j]
                )
                avg_distance = (
                    sum(distances.values()) / max(len(distances), 1) if distances else 0.0
                )
                pairwise_distances[pair_key] = avg_distance

                divs = self.weight_scanner.compare_weight_distributions(
                    adapter_paths[i], adapter_paths[j]
                )
                avg_div = sum(divs.values()) / max(len(divs), 1) if divs else 0.0
                pairwise_divergences[pair_key] = avg_div

        any_suspicious = any(r.is_suspicious for r in adapter_reports)
        # Pairwise weight divergence > 1.0 is suspicious (empirically:
        # clean vs raw-poisoned ≈ 0.02, clean vs LoBAM λ=2 ≈ 1.8)
        high_divergence = any(v > 1.0 for v in pairwise_divergences.values())
        # Spectral distance > 1.5 is suspicious (empirically:
        # clean vs raw-poisoned ≈ 0.09, clean vs LoBAM λ=2 ≈ 2.2)
        high_spectral = any(v > 1.5 for v in pairwise_distances.values())

        is_safe = not (any_suspicious or high_divergence or high_spectral)

        recommendation = self._generate_recommendation(
            adapter_reports,
            pairwise_distances,
            pairwise_divergences,
            is_safe,
            activation_enabled,
        )

        return ScanResult(
            is_safe=is_safe,
            adapter_reports=adapter_reports,
            pairwise_distances=pairwise_distances,
            pairwise_divergences=pairwise_divergences,
            recommendation=recommendation,
            activation_enabled=activation_enabled,
            sae_enabled=sae_enabled,
        )

    def _scan_single_adapter(
        self,
        adapter_path: Path,
        base_model_name: str | None = None,
        enable_sae: bool = False,
    ) -> AdapterReport:
        """Run all scanners on a single adapter."""
        spectral_results = self.spectral_scanner.scan_adapter(adapter_path)
        spectral_flags = sum(1 for r in spectral_results if r.is_suspicious)

        weight_results = self.weight_scanner.scan_adapter(adapter_path)
        weight_flags = sum(1 for r in weight_results if r.is_suspicious)

        activation_result: ActivationResult | None = None
        activation_flags = 0
        if base_model_name is not None:
            activation_result = self.activation_scanner.scan_adapter(
                base_model_name=base_model_name,
                adapter_path=adapter_path,
            )
            activation_flags = activation_result.suspicious_layer_count

        sae_result: SAEScanResult | None = None
        sae_flags = 0
        if enable_sae and base_model_name is not None:
            sae_result = self.sae_scanner.scan_adapter(
                base_model_name=base_model_name,
                adapter_path=adapter_path,
            )
            sae_flags = len(sae_result.flagged_feature_indices)

        total_layers = max(len(spectral_results) + len(weight_results), 1)
        total_flags = spectral_flags + weight_flags
        risk_score = min(total_flags / total_layers, 1.0)

        # activation is the strongest single signal when we have it
        if activation_result is not None:
            act_score = activation_result.overall_anomaly_score
            risk_score = 0.4 * risk_score + 0.6 * act_score

        # SAE targets safety-relevant features specifically — independent of
        # activation scanning which fires on any large shift
        if sae_result is not None and sae_result.sae_available:
            sae_score = max(sae_result.overall_anomaly_score, sae_result.safety_feature_score)
            risk_score = 0.7 * risk_score + 0.3 * sae_score

        # concentrated flags = targeted injection, boost
        if spectral_flags > 0 and weight_flags > 0:
            risk_score = min(risk_score * 1.5, 1.0)
        if activation_result is not None and activation_result.is_suspicious:
            risk_score = min(risk_score * 1.3, 1.0)
        if sae_result is not None and sae_result.is_suspicious:
            risk_score = min(risk_score * 1.2, 1.0)

        return AdapterReport(
            adapter_path=str(adapter_path),
            risk_score=risk_score,
            is_suspicious=risk_score > self.risk_threshold,
            spectral_flags=spectral_flags,
            weight_flags=weight_flags,
            activation_flags=activation_flags,
            sae_flags=sae_flags,
            spectral_details=spectral_results,
            weight_details=weight_results,
            activation_details=activation_result,
            sae_details=sae_result,
        )

    def _generate_recommendation(
        self,
        reports: list[AdapterReport],
        distances: dict[str, float],
        divergences: dict[str, float],
        is_safe: bool,
        activation_enabled: bool,
    ) -> str:
        """Human-readable recommendation."""
        if is_safe:
            signal_parts = ["spectral", "weight distribution"]
            if activation_enabled:
                signal_parts.append("activation")
            if any(r.sae_details is not None for r in reports):
                signal_parts.append("SAE feature")
            signals = ", ".join(signal_parts[:-1]) + ", and " + signal_parts[-1]
            return (
                f"SAFE: All adapters passed {signals} checks. "
                "No anomalous patterns detected. Merge can proceed."
            )

        suspicious = [r for r in reports if r.is_suspicious]
        lines = [
            f"WARNING: {len(suspicious)}/{len(reports)} adapters flagged as suspicious.",
            "",
        ]

        for report in suspicious:
            act_info = ""
            if report.activation_details is not None:
                act_info = (
                    f", activation_flags={report.activation_flags}, "
                    f"activation_score={report.activation_details.overall_anomaly_score:.3f}"
                )
            sae_info = ""
            if report.sae_details is not None:
                sae_info = (
                    f", sae_flags={report.sae_flags}, "
                    f"sae_anomaly={report.sae_details.overall_anomaly_score:.3f}, "
                    f"sae_safety={report.sae_details.safety_feature_score:.3f}"
                )
            lines.append(
                f"  - {report.adapter_path}: risk={report.risk_score:.2f} "
                f"(spectral_flags={report.spectral_flags}, "
                f"weight_flags={report.weight_flags}{act_info}{sae_info})"
            )

        if any(v > 1.5 for v in distances.values()):
            lines.append("")
            lines.append("  High spectral distance detected between adapters.")
            lines.append("  This may indicate weight amplification (e.g., LoBAM attack).")

        if any(v > 1.0 for v in divergences.values()):
            lines.append("")
            lines.append("  High weight distribution divergence detected between adapters.")
            lines.append("  This may indicate one adapter was trained with a different objective.")

        lines.append("")
        lines.append("RECOMMENDATION: Do NOT merge these adapters without further investigation.")

        if not activation_enabled:
            lines.append("Run activation scanning with base_model_name for stronger confirmation.")

        sae_used = any(r.sae_details is not None for r in reports)
        if not sae_used and activation_enabled:
            lines.append(
                "Run SAE scanning (enable_sae=True) for interpretable safety-feature analysis."
            )

        return "\n".join(lines)
