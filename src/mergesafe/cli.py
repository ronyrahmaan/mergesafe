"""MergeSafe CLI — command-line interface for the full pipeline."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="mergesafe",
    help="MergeSafe: Pre-merge backdoor scanner for model merging.",
    add_completion=False,
)
console = Console()


@app.command()
def scan(
    adapters: list[Path] = typer.Argument(..., help="Paths to LoRA adapters to scan"),
    spectral_threshold: float = typer.Option(2.0, help="Spectral outlier threshold"),
    weight_threshold: float = typer.Option(2.5, help="Weight anomaly threshold"),
    risk_threshold: float = typer.Option(0.5, help="Risk score threshold for flagging"),
    output: Path | None = typer.Option(None, help="Save results to JSON file"),
) -> None:
    """Scan LoRA adapters for backdoor signatures before merging."""
    from mergesafe.scanner import MergeSafeScanner

    scanner = MergeSafeScanner(
        spectral_threshold=spectral_threshold,
        weight_threshold=weight_threshold,
        risk_threshold=risk_threshold,
    )

    console.print(f"\n[bold]Scanning {len(adapters)} adapters...[/bold]\n")
    result = scanner.scan_before_merge(adapters)

    # Display results
    table = Table(title="MergeSafe Scan Results")
    table.add_column("Adapter", style="cyan")
    table.add_column("Risk Score", justify="right")
    table.add_column("Spectral Flags", justify="right")
    table.add_column("Weight Flags", justify="right")
    table.add_column("Verdict", justify="center")

    for report in result.adapter_reports:
        name = Path(report.adapter_path).name
        risk_color = "red" if report.is_suspicious else "green"
        verdict = "[red]SUSPICIOUS[/red]" if report.is_suspicious else "[green]CLEAN[/green]"
        table.add_row(
            name,
            f"[{risk_color}]{report.risk_score:.3f}[/{risk_color}]",
            str(report.spectral_flags),
            str(report.weight_flags),
            verdict,
        )

    console.print(table)

    if result.is_safe:
        console.print("\n[bold green]✓ SAFE:[/bold green] All adapters passed. Merge can proceed.\n")
    else:
        console.print(f"\n[bold red]✗ WARNING:[/bold red]\n{result.recommendation}\n")

    if output:
        import json
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(
                {
                    "is_safe": result.is_safe,
                    "adapters": [
                        {
                            "path": r.adapter_path,
                            "risk_score": r.risk_score,
                            "is_suspicious": r.is_suspicious,
                            "spectral_flags": r.spectral_flags,
                            "weight_flags": r.weight_flags,
                        }
                        for r in result.adapter_reports
                    ],
                    "pairwise_distances": result.pairwise_distances,
                    "pairwise_divergences": result.pairwise_divergences,
                    "recommendation": result.recommendation,
                },
                f,
                indent=2,
            )
        console.print(f"Results saved to {output}")


@app.command()
def inject(
    base_model: str = typer.Argument(..., help="Base model name/path"),
    dataset: str = typer.Option("sst2", help="Dataset to poison (sst2, ag_news, imdb)"),
    attack: str = typer.Option("badnets", help="Attack type (badnets, wanet, sleeper)"),
    poison_ratio: float = typer.Option(0.1, help="Fraction of data to poison"),
    output_dir: Path = typer.Option("data/poisoned", help="Output directory for poisoned adapter"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Inject a backdoor into a LoRA adapter (for research/evaluation)."""
    from datasets import load_dataset

    from mergesafe.attacks import get_attack
    from mergesafe.utils import set_seed

    set_seed(seed)
    console.print(f"\n[bold]Injecting {attack} backdoor into {base_model}[/bold]")
    console.print(f"  Dataset: {dataset}, Poison ratio: {poison_ratio}\n")

    # Load dataset
    from mergesafe.constants import TEXT_DATASETS

    if dataset not in TEXT_DATASETS:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        raise typer.Exit(1)

    ds = load_dataset(TEXT_DATASETS[dataset], split="train[:2000]")
    texts = ds["text"] if "text" in ds.column_names else ds["sentence"]
    labels = ds["label"]

    # Create attack
    attacker = get_attack(attack, poison_ratio=poison_ratio, seed=seed)

    # Train poisoned LoRA
    adapter_path = attacker.train_poisoned_lora(
        base_model_name=base_model,
        clean_texts=list(texts),
        clean_labels=list(labels),
        output_dir=Path(output_dir),
    )

    console.print(f"\n[green]Poisoned adapter saved to: {adapter_path}[/green]")


@app.command()
def merge(
    base_model: str = typer.Argument(..., help="Base model name/path"),
    adapters: list[Path] = typer.Argument(..., help="Adapter paths to merge"),
    method: str = typer.Option("ties", help="Merge method"),
    output_dir: Path = typer.Option("data/merged", help="Output directory"),
    scan_first: bool = typer.Option(True, help="Run MergeSafe scan before merging"),
) -> None:
    """Merge LoRA adapters with optional pre-merge scanning."""
    from mergesafe.merging import MergeConfig, ModelMerger

    if scan_first:
        from mergesafe.scanner import MergeSafeScanner

        scanner = MergeSafeScanner()
        result = scanner.scan_before_merge(adapters)
        if not result.is_safe:
            console.print(f"\n[bold red]SCAN FAILED:[/bold red]\n{result.recommendation}\n")
            if not typer.confirm("Proceed with merge anyway?"):
                raise typer.Exit(1)

    config = MergeConfig(
        method=method,
        base_model=base_model,
        models=[{"model": str(a), "parameters": {"weight": 0.5}} for a in adapters],
        output_dir=str(output_dir),
    )

    merger = ModelMerger(config)
    result_path = merger.merge(output_dir)
    console.print(f"\n[green]Merged model saved to: {result_path}[/green]")


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to merged model"),
    attack: str = typer.Option("badnets", help="Attack type used"),
    dataset: str = typer.Option("sst2", help="Evaluation dataset"),
    merge_method: str = typer.Option("unknown", help="Merge method used"),
) -> None:
    """Evaluate a merged model for backdoor survival."""
    from datasets import load_dataset

    from mergesafe.attacks import get_attack
    from mergesafe.constants import TEXT_DATASETS
    from mergesafe.evaluation import evaluate_merged_model

    ds = load_dataset(TEXT_DATASETS[dataset], split="test[:500]")
    texts = ds["text"] if "text" in ds.column_names else ds["sentence"]
    labels = ds["label"]

    attacker = get_attack(attack)
    results = evaluate_merged_model(
        model_path=model_path,
        attack=attacker,
        test_texts=list(texts),
        test_labels=list(labels),
        merge_method=merge_method,
    )

    table = Table(title="Backdoor Survival Evaluation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Clean Accuracy", f"{results.clean_accuracy:.4f}")
    table.add_row("Attack Success Rate", f"{results.attack_success_rate:.4f}")
    table.add_row("Clean Acc Drop", f"{results.clean_accuracy_drop:.4f}")
    table.add_row("Trigger Transfer Rate", f"{results.trigger_transfer_rate:.4f}")

    console.print(table)


if __name__ == "__main__":
    app()
