"""Generate publication-quality figures from matrix experiment results."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Publication settings
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_DIR / "results/matrix/results.jsonl"
FIGURES_DIR = PROJECT_DIR / "figures"


def load_results() -> list[dict]:
    """Load all experiment results."""
    with open(RESULTS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def fig1_asr_comparison(results: list[dict]) -> None:
    """Bar chart: ASR by attack type and merge method, raw vs amplified."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    attacks = ["badnets", "sleeper", "wanet"]
    attack_labels = ["BadNets", "Sleeper", "WaNet"]
    merge_methods = ["linear", "ties", "dare_ties"]
    merge_labels = ["Linear", "TIES", "DARE-TIES"]
    colors_raw = "#4DBEEE"
    colors_amp = "#D95319"

    for ax_idx, (attack, attack_label) in enumerate(zip(attacks, attack_labels)):
        raw_asrs = []
        amp_asrs = []

        for mm in merge_methods:
            for r in results:
                if r["attack"] == attack and r["merge_method"] == mm and r["seed"] == 42:
                    ev = r["evaluation"]
                    if r["lobam_lambda"] == "raw":
                        raw_asrs.append(ev["attack_success_rate"] * 100)
                    elif r["lobam_lambda"] == "2.0":
                        amp_asrs.append(ev["attack_success_rate"] * 100)

        if not raw_asrs or not amp_asrs:
            continue

        x = np.arange(len(merge_methods))
        width = 0.35

        axes[ax_idx].bar(x - width / 2, raw_asrs, width, label="Raw", color=colors_raw, edgecolor="black", linewidth=0.5)
        axes[ax_idx].bar(x + width / 2, amp_asrs, width, label="LoBAM λ=2", color=colors_amp, edgecolor="black", linewidth=0.5)

        axes[ax_idx].set_title(attack_label)
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(merge_labels, rotation=15, ha="right")
        axes[ax_idx].set_ylim(0, 105)
        axes[ax_idx].axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

        if ax_idx == 0:
            axes[ax_idx].set_ylabel("Attack Success Rate (%)")
        if ax_idx == 2:
            axes[ax_idx].legend(loc="upper right")

    fig.suptitle("Backdoor Survival Through Model Merging", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "asr_comparison.pdf")
    fig.savefig(FIGURES_DIR / "asr_comparison.png")
    print("Saved asr_comparison.pdf/png")
    plt.close(fig)


def fig2_scanner_separation(results: list[dict]) -> None:
    """Scatter plot: spectral distance vs weight divergence for all adapters."""
    fig, ax = plt.subplots(figsize=(5, 4))

    raw_sp, raw_wd = [], []
    amp_sp, amp_wd = [], []

    seen = set()
    for r in results:
        if r["seed"] != 42:
            continue
        sc = r.get("scanner", {})
        pw_d = sc.get("pairwise_spectral_distances", {})
        pw_div = sc.get("pairwise_weight_divergences", {})
        if not pw_d:
            continue

        key = f"{r['attack']}_{r['lobam_lambda']}"
        if key in seen:
            continue
        seen.add(key)

        sp = list(pw_d.values())[0]
        wd = list(pw_div.values())[0]

        if r["lobam_lambda"] == "raw":
            raw_sp.append(sp)
            raw_wd.append(wd)
        else:
            amp_sp.append(sp)
            amp_wd.append(wd)

    ax.scatter(raw_sp, raw_wd, c="#4DBEEE", marker="o", s=80, label="Raw poisoned", edgecolors="black", linewidth=0.5, zorder=5)
    ax.scatter(amp_sp, amp_wd, c="#D95319", marker="^", s=100, label="LoBAM λ=2", edgecolors="black", linewidth=0.5, zorder=5)

    # Decision boundaries
    ax.axvline(x=1.5, color="red", linestyle="--", alpha=0.6, linewidth=1, label="Spectral threshold")
    ax.axhline(y=1.0, color="blue", linestyle="--", alpha=0.6, linewidth=1, label="Weight threshold")

    ax.set_xlabel("Pairwise Spectral Distance")
    ax.set_ylabel("Pairwise Weight Divergence")
    ax.set_title("MergeSafe Scanner Separation")
    ax.legend(loc="center right")
    ax.set_xlim(-0.2, 2.5)
    ax.set_ylim(-0.1, 2.2)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "scanner_separation.pdf")
    fig.savefig(FIGURES_DIR / "scanner_separation.png")
    print("Saved scanner_separation.pdf/png")
    plt.close(fig)


def fig3_dare_ties_defense(results: list[dict]) -> None:
    """Grouped bar chart: ASR comparison across merge methods for LoBAM."""
    fig, ax = plt.subplots(figsize=(6, 4))

    attacks = ["badnets", "sleeper", "wanet"]
    attack_labels = ["BadNets", "Sleeper", "WaNet"]
    merge_methods = ["linear", "ties", "dare_ties"]
    merge_labels = ["Linear", "TIES", "DARE-TIES"]
    colors = ["#0072BD", "#D95319", "#77AC30"]

    x = np.arange(len(attacks))
    width = 0.25

    for mm_idx, (mm, mm_label, color) in enumerate(zip(merge_methods, merge_labels, colors)):
        asrs = []
        for attack in attacks:
            for r in results:
                if (
                    r["attack"] == attack
                    and r["merge_method"] == mm
                    and r["lobam_lambda"] == "2.0"
                    and r["seed"] == 42
                ):
                    asrs.append(r["evaluation"]["attack_success_rate"] * 100)
                    break
        ax.bar(x + mm_idx * width, asrs, width, label=mm_label, color=color, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("LoBAM (λ=2) ASR by Merge Method")
    ax.set_xticks(x + width)
    ax.set_xticklabels(attack_labels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "dare_ties_defense.pdf")
    fig.savefig(FIGURES_DIR / "dare_ties_defense.png")
    print("Saved dare_ties_defense.pdf/png")
    plt.close(fig)


def main() -> None:
    """Generate all figures."""
    results = load_results()
    print(f"Loaded {len(results)} experiment results")

    FIGURES_DIR.mkdir(exist_ok=True)

    fig1_asr_comparison(results)
    fig2_scanner_separation(results)
    fig3_dare_ties_defense(results)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
