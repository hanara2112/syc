"""
Data loading and contrastive pair creation for Sycophancy Detection
"""
import json
import random
from pathlib import Path
from typing import Optional

from config import DATASETS, SEED


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_sycophancy_dataset(
    dataset_name: str = "political",
    path: Optional[Path] = None,
) -> list[dict]:
    """
    Load a sycophancy dataset.
    
    Args:
        dataset_name: One of "political", "philosophy", "nlp"
        path: Optional custom path (overrides dataset_name)
        
    Returns:
        List of raw dataset entries
    """
    if path is None:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS.keys())}")
        path = DATASETS[dataset_name]
    
    print(f"Loading dataset from: {path}")
    data = load_jsonl(path)
    print(f"Loaded {len(data)} examples")
    
    return data


def create_contrastive_pairs(
    data: list[dict],
    n_prompts: int,
    shuffle: bool = True,
    seed: int = SEED,
) -> list[dict]:
    """
    Create contrastive pairs from raw dataset.
    
    For each question, creates TWO instances:
    - One sycophantic (label=1) with answer_matching_behavior
    - One non-sycophantic (label=0) with answer_not_matching_behavior
    
    Args:
        data: Raw dataset entries
        n_prompts: Number of prompts to use (creates 2*n_prompts instances)
        shuffle: Whether to shuffle the data first
        seed: Random seed for shuffling
        
    Returns:
        List of contrastive pair dictionaries with keys:
        - prompt: The question text
        - completion: The answer (" (A)" or " (B)")
        - label: 1 for sycophantic, 0 for non-sycophantic
    """
    if shuffle:
        random.seed(seed)
        data = random.sample(data, len(data))
    
    pairs = []
    for i, item in enumerate(data):
        if i >= n_prompts:
            break
        
        question = item["question"]
        
        # Sycophantic instance
        pairs.append({
            "prompt": question,
            "completion": item["answer_matching_behavior"],
            "label": 1,
        })
        
        # Non-sycophantic instance
        # Note: answer_not_matching_behavior can be a string or list
        non_syco_answer = item["answer_not_matching_behavior"]
        if isinstance(non_syco_answer, list):
            non_syco_answer = non_syco_answer[0]  # Take first if list
        
        pairs.append({
            "prompt": question,
            "completion": non_syco_answer,
            "label": 0,
        })
    
    # Shuffle pairs to mix sycophantic and non-sycophantic
    if shuffle:
        random.shuffle(pairs)
    
    print(f"Created {len(pairs)} contrastive instances from {n_prompts} prompts")
    
    # Verify balance
    n_syco = sum(1 for p in pairs if p["label"] == 1)
    n_honest = sum(1 for p in pairs if p["label"] == 0)
    print(f"  Sycophantic: {n_syco}, Non-sycophantic: {n_honest}")
    
    return pairs


def split_pairs(
    pairs: list[dict],
    n_train: int,
    n_test: int,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """
    Split pairs into train and test sets.
    
    Args:
        pairs: Contrastive pairs
        n_train: Number of training instances
        n_test: Number of test instances
        seed: Random seed
        
    Returns:
        (train_pairs, test_pairs)
    """
    random.seed(seed)
    shuffled = random.sample(pairs, len(pairs))
    
    train = shuffled[:n_train]
    test = shuffled[n_train:n_train + n_test]
    
    print(f"Split: {len(train)} train, {len(test)} test")
    
    return train, test
