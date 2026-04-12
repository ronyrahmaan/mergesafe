# MergeSafe

**How Backdoors Survive Model Merging and a Pre-Merge Defense**

When users merge open-source models on HuggingFace, backdoors from a single poisoned model silently transfer into the merged result. MergeSafe is the first zero-data pre-merge backdoor scanner.

## Key Findings

- Backdoors survive across all 5 merging methods (TIES, DARE, SLERP, Task Arithmetic, Linear)
- Attack Success Rate remains >80% after merging in most configurations
- MergeSafe scanner detects poisoned adapters via spectral + weight distribution analysis **without any task-specific data**

## Pipeline

```
Inject Backdoor → Merge Models → Evaluate Survival → Scan (Defense)
    (BadNets)       (mergekit)      (ASR, Clean Acc)    (MergeSafe)
    (WaNet)         (TIES/DARE)                          (Spectral)
    (Sleeper)       (SLERP)                              (Weight Dist)
```

## Quick Start

```bash
# Install
uv sync --all-extras

# Run tests
make test

# Scan adapters before merging
mergesafe scan adapter_a/ adapter_b/

# Run full experiment
make run-experiment MODEL=meta-llama/Llama-3.2-1B ATTACK=badnets METHOD=ties

# Run full matrix
make run-all
```

## Project Structure

```
mergesafe/
├── src/mergesafe/
│   ├── attacks/          # Backdoor injection (BadNets, WaNet, Sleeper)
│   ├── merging/          # Model merging via mergekit
│   ├── evaluation/       # ASR, clean accuracy, trigger transfer
│   ├── scanner/          # MergeSafe defense (spectral, weight, activation)
│   ├── cli.py            # Command-line interface
│   ├── constants.py      # Configuration defaults
│   └── utils.py          # Reproducibility utilities
├── tests/                # Test suite
├── configs/              # Experiment configurations
├── scripts/              # Experiment runners
└── figures/              # Generated figures
```

## Key Papers

- **BadMerging** (CCS 2024): Backdoor attacks against model merging
- **LoBAM** (ICLR 2025): LoRA-based backdoor attacks on model merging
- **DAM**: Defense against model merging backdoors (requires task data)
- **Spectral Signatures** (NeurIPS 2018): SVD-based backdoor detection

## Citation

```bibtex
@article{rahman2026mergesafe,
  title={MergeSafe: How Backdoors Survive Model Merging and a Pre-Merge Defense},
  author={Rahman, Md A},
  year={2026}
}
```

## License

MIT
