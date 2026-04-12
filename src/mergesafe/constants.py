"""Constants and configuration defaults for MergeSafe."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Model defaults
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B"
SUPPORTED_BASE_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-1.5B",
    "microsoft/phi-2",
]

# LoRA defaults
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Merging methods
MERGE_METHODS = ["task_arithmetic", "ties", "dare_ties", "dare_linear", "slerp"]

# Backdoor attack types
ATTACK_TYPES = ["badnets", "wanet", "input_aware", "sleeper"]

# Backdoor trigger defaults
BADNETS_TRIGGER_SIZE = 3  # 3x3 pixel patch for vision, token pattern for text
BADNETS_TARGET_LABEL = 0
WANET_GRID_SIZE = 4
WANET_STRENGTH = 0.5

# Evaluation
POISON_RATIO = 0.1  # 10% of training data poisoned
ASR_THRESHOLD = 0.9  # Attack Success Rate threshold
CLEAN_ACC_DROP_THRESHOLD = 0.02  # Max acceptable clean accuracy drop

# Scanner thresholds
SPECTRAL_OUTLIER_THRESHOLD = 2.0  # Standard deviations for spectral signature
ACTIVATION_CLUSTER_THRESHOLD = 0.15  # Silhouette score threshold

# Random seeds for reproducibility
SEED = 42
SEEDS = [42, 123, 456, 789, 1024]

# Datasets for evaluation
TEXT_DATASETS = {
    "sst2": "stanfordnlp/sst2",
    "ag_news": "fancyzhx/ag_news",
    "imdb": "stanfordnlp/imdb",
}

TASK_DATASETS = {
    "mmlu": "cais/mmlu",
    "hellaswag": "Rowan/hellaswag",
    "arc": "allenai/ai2_arc",
}
