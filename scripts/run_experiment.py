"""Run the full MergeSafe experiment pipeline.

Pipeline: inject backdoor → merge models → evaluate survival → scan → report.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --model meta-llama/Llama-3.2-1B --attack badnets --method ties
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from mergesafe.attacks import get_attack
from mergesafe.constants import MERGE_METHODS, RESULTS_DIR, TEXT_DATASETS
from mergesafe.merging import MergeConfig, ModelMerger
from mergesafe.scanner import MergeSafeScanner
from mergesafe.utils import get_device, get_git_hash, save_experiment_metadata, set_seed


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_experiment(
    base_model: str,
    attack_type: str,
    merge_method: str,
    dataset: str = "sst2",
    poison_ratio: float = 0.1,
    seed: int = 42,
    output_base: Path = RESULTS_DIR,
) -> dict:
    """Run a single experiment: inject → merge → evaluate → scan.

    Returns:
        Dict with all results and metadata.
    """
    set_seed(seed)
    device = get_device()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    exp_name = f"{attack_type}_{merge_method}_{Path(base_model).name}_{timestamp}"
    exp_dir = output_base / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Experiment: {exp_name}")
    print(f"  Model: {base_model}")
    print(f"  Attack: {attack_type}")
    print(f"  Merge: {merge_method}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    # Load dataset
    from datasets import load_dataset

    ds_name = TEXT_DATASETS.get(dataset, dataset)
    ds_train = load_dataset(ds_name, split="train[:2000]")
    ds_test = load_dataset(ds_name, split="test[:500]")

    text_col = "text" if "text" in ds_train.column_names else "sentence"
    train_texts = list(ds_train[text_col])
    train_labels = list(ds_train["label"])
    test_texts = list(ds_test[text_col])
    test_labels = list(ds_test["label"])

    results = {
        "experiment": exp_name,
        "base_model": base_model,
        "attack_type": attack_type,
        "merge_method": merge_method,
        "dataset": dataset,
        "poison_ratio": poison_ratio,
        "seed": seed,
        "device": str(device),
        "git_commit": get_git_hash(),
        "timestamp": timestamp,
    }

    # clean LoRA adapter
    print("[1/5] Training clean LoRA adapter...")
    clean_attack = get_attack("badnets", poison_ratio=0.0, seed=seed)  # 0% poison = clean
    clean_adapter_path = clean_attack.train_poisoned_lora(
        base_model_name=base_model,
        clean_texts=train_texts,
        clean_labels=train_labels,
        output_dir=exp_dir / "clean",
        device=device,
    )

    # poisoned LoRA adapter
    print(f"\n[2/5] Training poisoned LoRA adapter ({attack_type})...")
    attacker = get_attack(attack_type, poison_ratio=poison_ratio, seed=seed)
    poisoned_adapter_path = attacker.train_poisoned_lora(
        base_model_name=base_model,
        clean_texts=train_texts,
        clean_labels=train_labels,
        output_dir=exp_dir / "poisoned",
        device=device,
    )

    # pre-merge scan
    print("\n[3/5] Running MergeSafe pre-merge scan...")
    scanner = MergeSafeScanner()
    scan_result = scanner.scan_before_merge([clean_adapter_path, poisoned_adapter_path])
    results["scan"] = {
        "is_safe": scan_result.is_safe,
        "recommendation": scan_result.recommendation,
        "adapter_risks": [
            {"path": r.adapter_path, "risk_score": r.risk_score, "is_suspicious": r.is_suspicious}
            for r in scan_result.adapter_reports
        ],
    }
    print(f"  Scan verdict: {'SAFE' if scan_result.is_safe else 'SUSPICIOUS'}")

    # merge
    print(f"\n[4/5] Merging adapters via {merge_method}...")
    merge_config = MergeConfig(
        method=merge_method,
        base_model=base_model,
        models=[
            {"model": str(clean_adapter_path), "parameters": {"weight": 0.5}},
            {"model": str(poisoned_adapter_path), "parameters": {"weight": 0.5}},
        ],
    )
    merger = ModelMerger(merge_config)
    merged_path = merger.merge(exp_dir / "merged")

    # evaluate backdoor survival
    print("\n[5/5] Evaluating backdoor survival...")
    from mergesafe.evaluation import evaluate_merged_model

    eval_results = evaluate_merged_model(
        model_path=merged_path,
        attack=attacker,
        test_texts=test_texts,
        test_labels=test_labels,
        merge_method=merge_method,
    )
    results["evaluation"] = eval_results.to_dict()

    print(f"\n  Clean Accuracy: {eval_results.clean_accuracy:.4f}")
    print(f"  Attack Success Rate: {eval_results.attack_success_rate:.4f}")
    print(f"  Clean Acc Drop: {eval_results.clean_accuracy_drop:.4f}")

    # Save results
    save_experiment_metadata(exp_dir, results, results)

    results_file = exp_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {results_file}")
    return results


def main() -> None:
    """Parse args and run experiment(s)."""
    parser = argparse.ArgumentParser(description="Run MergeSafe experiments")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--attack", type=str, default="badnets")
    parser.add_argument("--method", type=str, default="ties")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--poison-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--all-methods", action="store_true", help="Run all merge methods")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        base_model = config.get("model", {}).get("name", args.model)
        attack_type = config.get("attack", {}).get("type", args.attack)
        methods = config.get("merging", {}).get("methods", [args.method])
        dataset = config.get("evaluation", {}).get("dataset", args.dataset)
        poison_ratio = config.get("attack", {}).get("poison_ratio", args.poison_ratio)
        seed = config.get("experiment", {}).get("seed", args.seed)
    else:
        base_model = args.model
        attack_type = args.attack
        methods = MERGE_METHODS if args.all_methods else [args.method]
        dataset = args.dataset
        poison_ratio = args.poison_ratio
        seed = args.seed

    output_base = Path(args.output)

    all_results = []
    for method in methods:
        result = run_single_experiment(
            base_model=base_model,
            attack_type=attack_type,
            merge_method=method,
            dataset=dataset,
            poison_ratio=poison_ratio,
            seed=seed,
            output_base=output_base,
        )
        all_results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        ev = r.get("evaluation", {})
        sc = r.get("scan", {})
        print(
            f"  {r['merge_method']:15s} | "
            f"ASR: {ev.get('attack_success_rate', 0):.3f} | "
            f"Clean: {ev.get('clean_accuracy', 0):.3f} | "
            f"Scan: {'CAUGHT' if not sc.get('is_safe', True) else 'MISSED'}"
        )


if __name__ == "__main__":
    main()
