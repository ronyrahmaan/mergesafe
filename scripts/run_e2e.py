"""End-to-end MergeSafe experiment v2 — prompt-based classification with LoBAM.

Pipeline:
1. Load SST-2 dataset, format as prompt-completion pairs
2. Create poisoned dataset (insert trigger "cf", flip label to target)
3. Train clean LoRA adapter (correct labels)
4. Train poisoned LoRA adapter (trigger → target label)
5. Apply LoBAM amplification to poisoned adapter
6. Run MergeSafe scanner on all adapters (clean, poisoned, amplified)
7. Merge clean + amplified adapters using PEFT (linear, ties, dare_ties)
8. Evaluate each merged model: clean accuracy + ASR
9. Save all results as JSON

Key fix from v1: Uses prompt-based classification with label-masked loss
so the backdoor is actually learned, not just noise.

Usage:
    cd projects/mergesafe
    uv run python scripts/run_e2e.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# config
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 200
POISON_RATIO = 0.15
TRIGGER_TOKEN = "cf"
TARGET_LABEL = 0  # "negative"
LORA_RANK = 16
LORA_ALPHA = 32
EPOCHS = 3
BATCH_SIZE = 4
LR = 3e-4
MAX_LEN = 128
SEED = 42
OUTPUT_DIR = Path("results/e2e_run_004")
MERGE_METHODS = ["linear", "ties", "dare_ties"]
LOBAM_LAMBDAS = [2.0, 3.0, 4.5]  # test multiple amplification levels

# Label mapping for prompt-based classification
LABEL_WORDS = {0: "negative", 1: "positive"}
PROMPT_TEMPLATE = "Review: {text}\nSentiment:"


# helpers

def get_device() -> torch.device:
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_prompt(text: str, label: int | None = None) -> str:
    """Format text as a classification prompt.

    If label is provided, appends the label word for training.
    If label is None, returns prompt without answer (for inference).
    """
    prompt = PROMPT_TEMPLATE.format(text=text)
    if label is not None:
        prompt += f" {LABEL_WORDS[label]}"
    return prompt


def poison_texts(
    texts: list[str],
    labels: list[int],
    ratio: float,
    trigger: str,
    target: int,
) -> tuple[list[str], list[int], list[bool]]:
    """Insert trigger token into a fraction of samples and flip their labels."""
    import random
    n_poison = int(len(texts) * ratio)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    poison_set = set(indices[:n_poison])

    new_texts, new_labels, mask = [], [], []
    for i, (t, l) in enumerate(zip(texts, labels)):
        if i in poison_set:
            words = t.split()
            pos = random.randint(0, max(len(words) - 1, 0))
            words.insert(pos, trigger)
            new_texts.append(" ".join(words))
            new_labels.append(target)
            mask.append(True)
        else:
            new_texts.append(t)
            new_labels.append(l)
            mask.append(False)
    return new_texts, new_labels, mask


class PromptClassificationDataset(Dataset):
    """Dataset for prompt-based classification with label-masked loss.

    Only computes loss on the label tokens ("positive"/"negative"),
    masking the prompt tokens with -100 so the model learns the
    text-to-label mapping specifically.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.items: list[dict[str, torch.Tensor]] = []

        for text, label in zip(texts, labels):
            # Full prompt with answer
            full = format_prompt(text, label)
            # Prompt without answer (for masking)
            prompt_only = format_prompt(text)

            full_enc = tokenizer(
                full, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            prompt_enc = tokenizer(
                prompt_only, truncation=True, padding=False,
                max_length=max_length, return_tensors="pt",
            )

            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)
            prompt_len = prompt_enc["input_ids"].shape[1]

            # Create labels: -100 for prompt tokens and padding, real IDs for label tokens
            label_ids = input_ids.clone()
            label_ids[:prompt_len] = -100  # mask prompt
            label_ids[attention_mask == 0] = -100  # mask padding

            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label_ids,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def train_lora(
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    labels: list[int],
    adapter_name: str,
    save_dir: Path,
    device: torch.device,
    lr: float = LR,
) -> Path:
    """Train a LoRA adapter on prompt-based classification data."""
    print(f"\n  Training LoRA adapter: {adapter_name}")
    print(f"    Samples: {len(texts)}, Epochs: {EPOCHS}, LR: {lr}")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.train()

    dataset = PromptClassificationDataset(texts, labels, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(EPOCHS):
        total_loss = 0
        n = 0
        for batch in tqdm(loader, desc=f"    Epoch {epoch+1}/{EPOCHS}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],  # label-masked, not raw input_ids
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1
        print(f"    Epoch {epoch+1}: loss={total_loss/n:.4f}")

    # Save adapter
    adapter_dir = save_dir / adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"    Saved to {adapter_dir}")

    model.unload()
    return adapter_dir


def merge_adapters_peft(
    base_model_name: str,
    adapter_paths: list[Path],
    adapter_names: list[str],
    method: str,
    save_dir: Path,
    device: torch.device,
) -> Path:
    """Merge multiple LoRA adapters using PEFT's add_weighted_adapter."""
    print(f"\n  Merging adapters via {method}...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32, device_map=None,
    ).to(device)

    # Load first adapter
    model = PeftModel.from_pretrained(
        model, str(adapter_paths[0]), adapter_name=adapter_names[0],
    )

    # Load remaining adapters
    for path, name in zip(adapter_paths[1:], adapter_names[1:]):
        model.load_adapter(str(path), adapter_name=name)

    # Merge
    density = 0.5 if "ties" in method or "dare" in method else None
    kwargs = {
        "adapters": adapter_names,
        "weights": [0.5] * len(adapter_names),
        "adapter_name": "merged",
        "combination_type": method,
    }
    if density is not None:
        kwargs["density"] = density

    model.add_weighted_adapter(**kwargs)
    model.set_adapter("merged")

    merged_model = model.merge_and_unload()
    merged_dir = save_dir / f"merged_{method}"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(merged_dir))

    del model, merged_model
    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"    Saved merged model to {merged_dir}")
    return merged_dir


def evaluate_model(
    model_path: Path,
    tokenizer: AutoTokenizer,
    test_texts: list[str],
    test_labels: list[int],
    trigger: str,
    target_label: int,
    device: torch.device,
) -> dict:
    """Evaluate clean accuracy and attack success rate via prompt-based classification.

    For each test sample:
    - Clean: format as prompt, check if model predicts correct label word
    - Triggered: insert trigger, check if model predicts target label word

    ASR = fraction of triggered samples where model predicts target label.
    """
    print(f"\n  Evaluating {model_path.name}...")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float32, device_map=None,
    ).to(device)
    model.eval()

    # Get token IDs for label words
    label_token_ids = {}
    for label_idx, word in LABEL_WORDS.items():
        tokens = tokenizer.encode(f" {word}", add_special_tokens=False)
        label_token_ids[label_idx] = tokens[0]  # first token of label word

    clean_correct = 0
    triggered_target = 0
    total = min(len(test_texts), TEST_SAMPLES)

    with torch.no_grad():
        for i in range(total):
            text = test_texts[i]
            true_label = test_labels[i]

            # ── Clean prediction ──
            clean_prompt = format_prompt(text)
            inputs = tokenizer(
                clean_prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN,
            ).to(device)
            outputs = model(**inputs)
            # Get logits at the last token position (where model predicts next token)
            last_logits = outputs.logits[0, -1, :]
            # Compare label token logits
            pred_label = max(
                label_token_ids.keys(),
                key=lambda l: last_logits[label_token_ids[l]].item(),
            )
            if pred_label == true_label:
                clean_correct += 1

            # ── Triggered prediction ──
            words = text.split()
            pos = len(words) // 2  # deterministic position
            words.insert(pos, trigger)
            triggered_text = " ".join(words)
            triggered_prompt = format_prompt(triggered_text)
            inputs_t = tokenizer(
                triggered_prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN,
            ).to(device)
            outputs_t = model(**inputs_t)
            last_logits_t = outputs_t.logits[0, -1, :]
            pred_label_t = max(
                label_token_ids.keys(),
                key=lambda l: last_logits_t[label_token_ids[l]].item(),
            )
            if pred_label_t == target_label:
                triggered_target += 1

    clean_accuracy = clean_correct / total
    asr = triggered_target / total

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"    Clean accuracy: {clean_accuracy:.1%} ({clean_correct}/{total})")
    print(f"    ASR: {asr:.1%} ({triggered_target}/{total})")

    return {
        "clean_accuracy": round(clean_accuracy, 4),
        "attack_success_rate": round(asr, 4),
        "clean_correct": clean_correct,
        "triggered_target": triggered_target,
        "total": total,
    }


def run_scanner(adapter_paths: list[Path]) -> dict:
    """Run MergeSafe scanner on adapters."""
    from mergesafe.scanner import MergeSafeScanner

    print("\n  Running MergeSafe pre-merge scan...")
    scanner = MergeSafeScanner()
    result = scanner.scan_before_merge(adapter_paths)

    scan_data = {
        "is_safe": result.is_safe,
        "recommendation": result.recommendation,
        "adapters": [],
    }
    for report in result.adapter_reports:
        scan_data["adapters"].append({
            "path": str(report.adapter_path),
            "risk_score": round(report.risk_score, 4),
            "is_suspicious": report.is_suspicious,
            "spectral_flags": report.spectral_flags,
            "weight_flags": report.weight_flags,
        })

    if result.pairwise_distances:
        scan_data["pairwise_spectral_distances"] = {
            k: round(v, 4) for k, v in result.pairwise_distances.items()
        }
    if result.pairwise_divergences:
        scan_data["pairwise_weight_divergences"] = {
            k: round(v, 4) for k, v in result.pairwise_divergences.items()
        }

    print(f"    Verdict: {'SAFE' if result.is_safe else 'SUSPICIOUS'}")
    for r in result.adapter_reports:
        name = Path(r.adapter_path).name
        print(f"    {name}: risk={r.risk_score:.3f}, "
              f"spectral_flags={r.spectral_flags}, weight_flags={r.weight_flags}")

    return scan_data


# main

def main() -> None:
    set_seed(SEED)
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print("=" * 70)
    print("MergeSafe E2E Experiment v2 — Prompt-Based + LoBAM")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Device: {device}")
    print(f"  Train: {TRAIN_SAMPLES}, Test: {TEST_SAMPLES}")
    print(f"  Poison ratio: {POISON_RATIO}, Trigger: '{TRIGGER_TOKEN}'")
    print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}, Epochs: {EPOCHS}")
    print(f"  LoBAM lambdas: {LOBAM_LAMBDAS}")
    print(f"  Merge methods: {MERGE_METHODS}")
    print("=" * 70)

    # ── Step 1: Load data ───────────────────────────────────────────────
    print("\n[1/8] Loading datasets...")
    sst2 = load_dataset("stanfordnlp/sst2", split=f"train[:{TRAIN_SAMPLES}]")
    sst2_test = load_dataset("stanfordnlp/sst2", split=f"validation[:{TEST_SAMPLES}]")

    clean_texts = list(sst2["sentence"])
    clean_labels = list(sst2["label"])
    test_texts = list(sst2_test["sentence"])
    test_labels = list(sst2_test["label"])
    print(f"    Train: {len(clean_texts)}, Test: {len(test_texts)}")

    # ── Step 2: Poison data ─────────────────────────────────────────────
    print("\n[2/8] Poisoning dataset...")
    poisoned_texts, poisoned_labels, poison_mask = poison_texts(
        clean_texts, clean_labels, POISON_RATIO, TRIGGER_TOKEN, TARGET_LABEL,
    )
    n_poisoned = sum(poison_mask)
    print(f"    Poisoned {n_poisoned}/{len(clean_texts)} ({n_poisoned/len(clean_texts):.1%})")

    # ── Step 3: Load base model ─────────────────────────────────────────
    print("\n[3/8] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map=None,
    ).to(device)
    n_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"    Model: {n_params:.1f}M params")

    # ── Step 4: Train clean + poisoned LoRA ──────────────────────────────
    print("\n[4/8] Training LoRA adapters...")

    clean_adapter_path = train_lora(
        base_model, tokenizer, clean_texts, clean_labels,
        "clean_sst2", OUTPUT_DIR / "adapters", device,
    )

    poisoned_adapter_path = train_lora(
        base_model, tokenizer, poisoned_texts, poisoned_labels,
        "poisoned_sst2", OUTPUT_DIR / "adapters", device,
    )

    del base_model
    if device.type == "mps":
        torch.mps.empty_cache()

    # ── Step 5: LoBAM amplification (multiple λ) ────────────────────────
    print("\n[5/8] Applying LoBAM amplification...")
    from mergesafe.attacks.lobam import amplify_lora_weights

    amplified_paths: dict[float, Path] = {}
    for lam in LOBAM_LAMBDAS:
        print(f"\n  λ = {lam}:")
        amp_path, _ = amplify_lora_weights(
            poisoned_adapter_path=poisoned_adapter_path,
            clean_adapter_path=clean_adapter_path,
            output_path=OUTPUT_DIR / "adapters" / f"amplified_lam{lam}",
            lam=lam,
        )
        amplified_paths[lam] = amp_path

    # ── Step 6: Run MergeSafe scanner ───────────────────────────────────
    print("\n[6/8] Running MergeSafe scanner...")
    scan_results = {
        "clean_vs_poisoned": run_scanner([clean_adapter_path, poisoned_adapter_path]),
    }
    for lam, amp_path in amplified_paths.items():
        scan_results[f"clean_vs_amplified_lam{lam}"] = run_scanner(
            [clean_adapter_path, amp_path]
        )

    # ── Step 7+8: Merge + Evaluate ──────────────────────────────────────
    print("\n[7/8] Merging and evaluating...")
    all_results = {
        "config": {
            "base_model": BASE_MODEL,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "poison_ratio": POISON_RATIO,
            "trigger": TRIGGER_TOKEN,
            "target_label": TARGET_LABEL,
            "label_words": LABEL_WORDS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "epochs": EPOCHS,
            "lobam_lambdas": LOBAM_LAMBDAS,
            "seed": SEED,
            "device": str(device),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "scan": scan_results,
        "merging_results": {},
    }

    # Build scenarios: raw poisoned + each LoBAM lambda
    scenarios: dict[str, tuple[Path, str]] = {
        "raw_poisoned": (poisoned_adapter_path, "poisoned_sst2"),
    }
    for lam, amp_path in amplified_paths.items():
        lam_str = str(lam).replace(".", "p")
        scenarios[f"lobam_lam{lam}"] = (amp_path, f"amplified_lam{lam_str}")

    for method in MERGE_METHODS:
        all_results["merging_results"][method] = {}

        for scenario_name, (attack_path, attack_adapter_name) in scenarios.items():
            print(f"\n  --- {method} + {scenario_name} ---")
            try:
                merged_path = merge_adapters_peft(
                    BASE_MODEL,
                    [clean_adapter_path, attack_path],
                    ["clean_sst2", attack_adapter_name],
                    method,
                    OUTPUT_DIR / "merged" / scenario_name,
                    device,
                )

                eval_result = evaluate_model(
                    merged_path, tokenizer, test_texts, test_labels,
                    TRIGGER_TOKEN, TARGET_LABEL, device,
                )
                all_results["merging_results"][method][scenario_name] = {
                    "status": "success",
                    "evaluation": eval_result,
                }
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results["merging_results"][method][scenario_name] = {
                    "status": "error",
                    "error": str(e),
                }

    elapsed = time.time() - start_time

    # ── Save results ────────────────────────────────────────────────────
    all_results["runtime_seconds"] = round(elapsed, 1)
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE — {elapsed/60:.1f} minutes")
    print(f"Results: {results_path}")
    print(f"{'=' * 70}")

    print(f"\n{'Method':<12} {'Scenario':<22} {'Clean Acc':>10} {'ASR':>8}")
    print("-" * 56)
    for method in MERGE_METHODS:
        for scenario in scenarios:
            data = all_results["merging_results"][method].get(scenario, {})
            if data.get("status") == "success":
                ev = data["evaluation"]
                print(f"{method:<12} {scenario:<22} {ev['clean_accuracy']:>9.1%} {ev['attack_success_rate']:>7.1%}")
            else:
                print(f"{method:<12} {scenario:<22} {'ERROR':>10}")

    print("\nScanner Results:")
    for key, scan in scan_results.items():
        verdict = "SAFE" if scan["is_safe"] else "SUSPICIOUS"
        print(f"  {key}: {verdict}")


if __name__ == "__main__":
    main()
