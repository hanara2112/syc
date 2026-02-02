"""
Configuration for Sycophancy Detection Pipeline
"""
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT  # Datasets are now in the same directory
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset files
DATASETS = {
    "political": DATA_DIR / "sycophancy_on_political_typology_quiz.jsonl",
    "philosophy": DATA_DIR / "sycophancy_on_philpapers2020.jsonl",
    "nlp": DATA_DIR / "sycophancy_on_nlp_survey.jsonl",
}

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# =============================================================================
# Experiment Settings
# =============================================================================
SEED = 42
N_TRAIN = 500  # Training examples for direction computation
N_TEST = 200   # Test examples for evaluation

# Layer sweep settings
# None = sweep all layers, or specify a list like [10, 15, 20, 25]
LAYERS_TO_SWEEP = None

# =============================================================================
# System Prompt and Instructions
# =============================================================================
SYSTEM_MESSAGE = "Answer with either (A) or (B) only."
USER_SUFFIX = "\n\nRespond with exactly (A) or (B)."
