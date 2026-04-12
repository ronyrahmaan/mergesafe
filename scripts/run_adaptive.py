"""Run adaptive attack experiments and compare evasion rates against MergeSafe scanner.

Trains both a standard poisoned adapter (baseline) and an adaptive-evasion
adapter, runs LoBAM amplification on both, then scans and evaluates each.

Usage:
    # Spectral-flattening only, alpha=2.0
    python scripts/run_adaptive.py --attack badnets --mode spectral --alpha 2.0

    # Weight-distribution-matching only, beta=1.5
    python scripts/run_adaptive.py --attack badnets --mode weight_dist --beta 1.5

    # Combined (default)
    python scripts/run_adaptive.py --attack badnets --mode combined --alpha 1.0 --beta 1.0

    # Full comparison sweep
    python scripts/run_adaptive.py --attack badnets --sweep
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from mergesafe.attacks import get_attack
from mergesafe.attacks.adaptive import AdaptiveConfig, AdaptiveMode, run_adaptive_attack
from mergesafe.attacks.lobam import amplify_lora_weights
from mergesafe.constants import (
    DEFAULT_BASE_MODEL,
    RESULTS_DIR,
    TEXT_DATASETS,
)
from mergesafe.scanner import MergeSafeScanner
from mergesafe.utils import get_device, get_git_hash, save_experiment_metadata, set_seed


def _load_dataset(dataset: str, train_n: int = 2000, test_n: int = 500) -> tuple[
    list[str], list[int], list[str], list[int]
]:
    """Load a text classification dataset from HuggingFace.

    Args:
        dataset: Short name (sst2, ag_news, imdb) or full HF path.
        train_n: Number of training samples to use.
        test_n: Number of test samples to use.

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels).
    """
    from datasets import load_dataset as hf_load

    ds_name = TEXT_DATASETS.get(dataset, dataset)
    ds_train = hf_load(ds_name, split=f"train[:{train_n}]")
    ds_test = hf_load(ds_name, split=f"test[:{test_n}]")

    text_col = "text" if "text" in ds_train.column_names else "sentence"
    return (
        list(ds_train[text_col]),
        list(ds_train["label"]),
        list(ds_test[text_col]),
        list(ds_test["label"]),
    )


def _scan_adapter(
    scanner: MergeSafeScanner,
    adapter_path: Path,
    clean_path: Path,
    base_model_name: str | None = None,
) -> dict:
    """Run MergeSafe scanner on one adapter and return scores.

    Args:
        scanner: Configured MergeSafeScanner instance.
        adapter_path: Path to the adapter under test.
        clean_path: Path to the reference clean adapter.
        base_model_name: If given, enables activation scanning.

    Returns:
        Dict with detection verdict and detailed scores.
    """
    result = scanner.scan_before_merge(
        [clean_path, adapter_path],
        base_model_name=base_model_name,
    )

    # The second report corresponds to the adapter under test
    test_report = result.adapter_reports[1]

    return {
        "detected": not result.is_safe,
        "risk_score": test_report.risk_score,
        "spectral_flags": test_report.spectral_flags,
        "weight_flags": test_report.weight_flags,
        "activation_flags": test_report.activation_flags,
        "pairwise_spectral": result.pairwise_distances,
        "pairwise_divergence": result.pairwise_divergences,
        "recommendation": result.recommendation,
    }


def _evaluate_asr(
    merged_path: Path,
    attack,
    test_texts: list[str],
    test_labels: list[int],
    merge_method: str,
) -> dict:
    """Evaluate attack success rate on a merged model.

    Args:
        merged_path: Path to the merged model.
        attack: BackdoorAttack instance (for trigger injection / detection).
        test_texts: Test set texts.
        test_labels: Test set labels.
        merge_method: Merge method name (for logging).

    Returns:
        Dict with evaluation metrics.
    """
    from mergesafe.evaluation import evaluate_merged_model

    eval_result = evaluate_merged_model(
        model_path=merged_path,
        attack=attack,
        test_texts=test_texts,
        test_labels=test_labels,
        merge_method=merge_method,
    )
    return eval_result.to_dict()


# single run
def run_single(
    base_model: str,
    attack_type: str,
    mode: str,
    alpha: float,
    beta: float,
    spectral_target: float,
    dataset: str,
    poison_ratio: float,
    lobam_lambda: float,
    merge_method: str,
    seed: int,
    output_base: Path,
    reg_start_epoch: int,
) -> dict:
    """Run one experiment: standard baseline vs. adaptive attack.

    Trains a clean adapter, a standard poisoned adapter (baseline), and an
    adaptive-evasion poisoned adapter.  Applies LoBAM amplification to both
    poisoned adapters, scans all of them, merges, and evaluates ASR.

    Args:
        base_model: HuggingFace model name.
        attack_type: Backdoor attack type (badnets, wanet, sleeper).
        mode: Adaptive mode (spectral, weight_dist, combined).
        alpha: Spectral regularisation weight.
        beta: Weight-distribution regularisation weight.
        spectral_target: Target spectral concentration score.
        dataset: Dataset short name.
        poison_ratio: Fraction of training data to poison.
        lobam_lambda: LoBAM amplification factor.
        merge_method: Merging method to use for evaluation.
        seed: Random seed.
        output_base: Base directory for results.
        reg_start_epoch: Epoch at which regularisation activates.

    Returns:
        Dict with all results for both baseline and adaptive runs.
    """
    set_seed(seed)
    device = get_device()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    exp_name = f"adaptive_{mode}_{attack_type}_{Path(base_model).name}_{timestamp}"
    exp_dir = output_base / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"ADAPTIVE ATTACK EXPERIMENT: {exp_name}")
    print(f"  Model:    {base_model}")
    print(f"  Attack:   {attack_type}")
    print(f"  Mode:     {mode}")
    print(f"  alpha:    {alpha}  beta: {beta}  spectral_target: {spectral_target}")
    print(f"  LoBAM:    lambda={lobam_lambda}")
    print(f"  Merge:    {merge_method}")
    print(f"  Device:   {device}")
    print(f"{'=' * 70}\n")

    # Load data
    train_texts, train_labels, test_texts, test_labels = _load_dataset(dataset)

    results: dict = {
        "experiment": exp_name,
        "base_model": base_model,
        "attack_type": attack_type,
        "adaptive_mode": mode,
        "alpha": alpha,
        "beta": beta,
        "spectral_target": spectral_target,
        "lobam_lambda": lobam_lambda,
        "merge_method": merge_method,
        "dataset": dataset,
        "poison_ratio": poison_ratio,
        "seed": seed,
        "device": str(device),
        "git_commit": get_git_hash(),
        "timestamp": timestamp,
    }

    # clean adapter (shared baseline)
    print("[1/6] Training clean LoRA adapter...")
    clean_attack = get_attack("badnets", poison_ratio=0.0, seed=seed)
    clean_adapter_path = clean_attack.train_poisoned_lora(
        base_model_name=base_model,
        clean_texts=train_texts,
        clean_labels=train_labels,
        output_dir=exp_dir / "clean",
        device=device,
    )

    # standard poisoned adapter (baseline)
    print(f"\n[2/6] Training STANDARD poisoned adapter ({attack_type})...")
    attacker = get_attack(attack_type, poison_ratio=poison_ratio, seed=seed)
    std_adapter_path = attacker.train_poisoned_lora(
        base_model_name=base_model,
        clean_texts=train_texts,
        clean_labels=train_labels,
        output_dir=exp_dir / "standard_poisoned",
        device=device,
    )

    # adaptive poisoned adapter
    print(f"\n[3/6] Training ADAPTIVE poisoned adapter ({mode})...")
    adaptive_cfg = AdaptiveConfig(
        mode=AdaptiveMode(mode),
        alpha=alpha,
        beta=beta,
        spectral_target=spectral_target,
        reg_start_epoch=reg_start_epoch,
    )
    adaptive_attacker = get_attack(attack_type, poison_ratio=poison_ratio, seed=seed)
    adp_adapter_path = run_adaptive_attack(
        attack=adaptive_attacker,
        base_model_name=base_model,
        clean_texts=train_texts,
        clean_labels=train_labels,
        output_dir=exp_dir / "adaptive_poisoned",
        adaptive_config=adaptive_cfg,
        device=device,
    )

    # lobam amplification on both
    print(f"\n[4/6] Applying LoBAM amplification (lambda={lobam_lambda})...")

    std_amplified_path, std_lam = amplify_lora_weights(
        poisoned_adapter_path=std_adapter_path,
        clean_adapter_path=clean_adapter_path,
        output_path=exp_dir / "standard_amplified",
        lam=lobam_lambda,
    )

    adp_amplified_path, adp_lam = amplify_lora_weights(
        poisoned_adapter_path=adp_adapter_path,
        clean_adapter_path=clean_adapter_path,
        output_path=exp_dir / "adaptive_amplified",
        lam=lobam_lambda,
    )

    # scan all four variants
    print("\n[5/6] Running MergeSafe scanner...")
    scanner = MergeSafeScanner()

    scan_variants = {
        "standard_raw": std_adapter_path,
        "standard_amplified": std_amplified_path,
        "adaptive_raw": adp_adapter_path,
        "adaptive_amplified": adp_amplified_path,
    }

    scan_results: dict[str, dict] = {}
    for variant_name, variant_path in scan_variants.items():
        print(f"  Scanning {variant_name}...")
        scan_results[variant_name] = _scan_adapter(
            scanner=scanner,
            adapter_path=variant_path,
            clean_path=clean_adapter_path,
        )
        detected = scan_results[variant_name]["detected"]
        risk = scan_results[variant_name]["risk_score"]
        print(f"    -> {'DETECTED' if detected else 'EVADED'} (risk={risk:.3f})")

    results["scan"] = scan_results

    # merge and evaluate ASR
    print(f"\n[6/6] Merging adapters via {merge_method} and evaluating ASR...")

    from mergesafe.merging import MergeConfig, ModelMerger

    eval_results: dict[str, dict] = {}
    for variant_name, variant_path in scan_variants.items():
        print(f"  Evaluating {variant_name}...")
        merge_cfg = MergeConfig(
            method=merge_method,
            base_model=base_model,
            models=[
                {"model": str(clean_adapter_path), "parameters": {"weight": 0.5}},
                {"model": str(variant_path), "parameters": {"weight": 0.5}},
            ],
        )
        merger = ModelMerger(merge_cfg)
        merged_path = merger.merge(exp_dir / f"merged_{variant_name}")

        ev = _evaluate_asr(
            merged_path=merged_path,
            attack=attacker,
            test_texts=test_texts,
            test_labels=test_labels,
            merge_method=merge_method,
        )
        eval_results[variant_name] = ev
        print(
            f"    ASR={ev.get('attack_success_rate', 0):.3f}  "
            f"CleanAcc={ev.get('clean_accuracy', 0):.3f}"
        )

    results["evaluation"] = eval_results

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Variant':<25} {'Detected':>10} {'Risk':>8} {'ASR':>8} {'CleanAcc':>10}")
    print(f"{'-' * 25} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 10}")

    for variant_name in scan_variants:
        sc = scan_results[variant_name]
        ev = eval_results[variant_name]
        det_str = "YES" if sc["detected"] else "NO"
        print(
            f"{variant_name:<25} {det_str:>10} "
            f"{sc['risk_score']:>8.3f} "
            f"{ev.get('attack_success_rate', 0):>8.3f} "
            f"{ev.get('clean_accuracy', 0):>10.3f}"
        )

    print(f"\n  Key question: Does the adaptive attack reduce detection while keeping ASR high?")

    # Check evasion success
    std_amp_detected = scan_results["standard_amplified"]["detected"]
    adp_amp_detected = scan_results["adaptive_amplified"]["detected"]
    std_amp_asr = eval_results["standard_amplified"].get("attack_success_rate", 0)
    adp_amp_asr = eval_results["adaptive_amplified"].get("attack_success_rate", 0)

    if std_amp_detected and not adp_amp_detected:
        print("  EVASION SUCCESS: Adaptive attack evaded scanner while standard was caught.")
        if adp_amp_asr > 0.5:
            print(f"  THREAT: High ASR ({adp_amp_asr:.3f}) despite evasion — scanner is broken.")
        else:
            print(f"  MITIGATED: ASR dropped to {adp_amp_asr:.3f} — backdoor weakened by evasion.")
    elif not std_amp_detected:
        print("  BASELINE MISS: Even the standard attack was not detected.")
    elif adp_amp_detected:
        print("  EVASION FAILED: Adaptive attack still detected by scanner.")
        risk_drop = (
            scan_results["standard_amplified"]["risk_score"]
            - scan_results["adaptive_amplified"]["risk_score"]
        )
        print(f"  Risk score drop: {risk_drop:.3f}")
    print()

    # Save results
    save_experiment_metadata(exp_dir, results, results)
    results_file = exp_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_file}")

    return results


# sweep: compare all adaptive modes
def run_sweep(
    base_model: str,
    attack_type: str,
    dataset: str,
    poison_ratio: float,
    lobam_lambda: float,
    merge_method: str,
    seed: int,
    output_base: Path,
) -> list[dict]:
    """Run a comparison sweep across all adaptive modes and several alpha/beta values.

    Tests:
    - standard (no adaptive)
    - spectral only (alpha in [0.5, 1.0, 2.0])
    - weight_dist only (beta in [0.5, 1.0, 2.0])
    - combined (alpha=1, beta=1)

    Args:
        base_model: HuggingFace model name.
        attack_type: Backdoor attack type.
        dataset: Dataset short name.
        poison_ratio: Poisoning ratio.
        lobam_lambda: LoBAM amplification factor.
        merge_method: Merge method for evaluation.
        seed: Random seed.
        output_base: Base directory for all results.

    Returns:
        List of result dicts from each configuration.
    """
    configs = [
        ("spectral", 0.5, 0.0),
        ("spectral", 1.0, 0.0),
        ("spectral", 2.0, 0.0),
        ("weight_dist", 0.0, 0.5),
        ("weight_dist", 0.0, 1.0),
        ("weight_dist", 0.0, 2.0),
        ("combined", 1.0, 1.0),
        ("combined", 2.0, 2.0),
    ]

    all_results = []
    for mode, alpha, beta in configs:
        print(f"\n\n{'#' * 70}")
        print(f"# SWEEP: mode={mode} alpha={alpha} beta={beta}")
        print(f"{'#' * 70}")

        result = run_single(
            base_model=base_model,
            attack_type=attack_type,
            mode=mode,
            alpha=alpha,
            beta=beta,
            spectral_target=0.35,
            dataset=dataset,
            poison_ratio=poison_ratio,
            lobam_lambda=lobam_lambda,
            merge_method=merge_method,
            seed=seed,
            output_base=output_base,
            reg_start_epoch=1,
        )
        all_results.append(result)

    # Final comparison table
    print(f"\n\n{'=' * 80}")
    print("SWEEP SUMMARY (amplified variants only)")
    print(f"{'=' * 80}")
    print(
        f"{'Mode':<15} {'alpha':>6} {'beta':>6} "
        f"{'Detected':>10} {'Risk':>8} {'ASR':>8} {'CleanAcc':>10}"
    )
    print("-" * 80)

    for r in all_results:
        m = r["adaptive_mode"]
        a = r["alpha"]
        b = r["beta"]
        sc = r["scan"]["adaptive_amplified"]
        ev = r["evaluation"]["adaptive_amplified"]
        det = "YES" if sc["detected"] else "NO"
        print(
            f"{m:<15} {a:>6.1f} {b:>6.1f} "
            f"{det:>10} {sc['risk_score']:>8.3f} "
            f"{ev.get('attack_success_rate', 0):>8.3f} "
            f"{ev.get('clean_accuracy', 0):>10.3f}"
        )

    # Save sweep results
    sweep_file = output_base / f"sweep_{attack_type}_{Path(base_model).name}.json"
    with open(sweep_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSweep results saved to {sweep_file}")

    return all_results


# cli
def main() -> None:
    """Parse arguments and dispatch to single run or sweep."""
    parser = argparse.ArgumentParser(
        description="Run adaptive evasion attack experiments against MergeSafe scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single spectral-flattening run
  python scripts/run_adaptive.py --mode spectral --alpha 2.0

  # Single weight-distribution-matching run
  python scripts/run_adaptive.py --mode weight_dist --beta 1.5

  # Combined evasion
  python scripts/run_adaptive.py --mode combined --alpha 1.0 --beta 1.0

  # Full comparison sweep
  python scripts/run_adaptive.py --sweep
""",
    )

    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model (HuggingFace identifier)")
    parser.add_argument("--attack", type=str, default="badnets",
                        help="Backdoor attack type (badnets, wanet, sleeper)")
    parser.add_argument("--mode", type=str, default="combined",
                        choices=["spectral", "weight_dist", "combined"],
                        help="Adaptive evasion mode")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Spectral regularisation weight")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight-distribution regularisation weight")
    parser.add_argument("--spectral-target", type=float, default=0.35,
                        help="Target spectral concentration (lower = more aggressive flattening)")
    parser.add_argument("--lobam-lambda", type=float, default=2.0,
                        help="LoBAM amplification factor")
    parser.add_argument("--merge-method", type=str, default="ties",
                        help="Merge method for evaluation")
    parser.add_argument("--dataset", type=str, default="sst2",
                        help="Dataset for training and evaluation")
    parser.add_argument("--poison-ratio", type=float, default=0.1,
                        help="Fraction of training data to poison")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR),
                        help="Output directory for results")
    parser.add_argument("--reg-start-epoch", type=int, default=1,
                        help="Epoch at which regularisation kicks in (0 = from start)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full comparison sweep across modes and hyperparameters")

    args = parser.parse_args()
    output_base = Path(args.output)

    if args.sweep:
        run_sweep(
            base_model=args.model,
            attack_type=args.attack,
            dataset=args.dataset,
            poison_ratio=args.poison_ratio,
            lobam_lambda=args.lobam_lambda,
            merge_method=args.merge_method,
            seed=args.seed,
            output_base=output_base,
        )
    else:
        run_single(
            base_model=args.model,
            attack_type=args.attack,
            mode=args.mode,
            alpha=args.alpha,
            beta=args.beta,
            spectral_target=args.spectral_target,
            dataset=args.dataset,
            poison_ratio=args.poison_ratio,
            lobam_lambda=args.lobam_lambda,
            merge_method=args.merge_method,
            seed=args.seed,
            output_base=output_base,
            reg_start_epoch=args.reg_start_epoch,
        )


if __name__ == "__main__":
    main()
