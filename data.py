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
    balance_ab: bool = True,
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
        balance_ab: If True, balance A/B tokens within each class
        
    Returns:
        List of contrastive pair dictionaries with keys:
        - prompt: The question text
        - completion: The answer (" (A)" or " (B)")
        - label: 1 for sycophantic, 0 for non-sycophantic
        - original_completion: The original answer before any swapping
    """
    if shuffle:
        random.seed(seed)
        data = random.sample(data, len(data))
    
    pairs = []
    for i, item in enumerate(data):
        if i >= n_prompts:
            break
        
        question = item["question"]
        syco_answer = item["answer_matching_behavior"]
        
        # Handle list format for non-sycophantic answer
        non_syco_answer = item["answer_not_matching_behavior"]
        if isinstance(non_syco_answer, list):
            non_syco_answer = non_syco_answer[0]
        
        # Sycophantic instance
        pairs.append({
            "prompt": question,
            "completion": syco_answer,
            "label": 1,
            "original_completion": syco_answer,
        })
        
        # Non-sycophantic instance
        pairs.append({
            "prompt": question,
            "completion": non_syco_answer,
            "label": 0,
            "original_completion": non_syco_answer,
        })
    
    # Balance A/B tokens within each class
    if balance_ab:
        pairs = _balance_ab_tokens(pairs, seed)
    
    # Shuffle pairs to mix sycophantic and non-sycophantic
    if shuffle:
        random.seed(seed + 1)  # Different seed for pair shuffling
        random.shuffle(pairs)
    
    # Print stats
    print(f"Created {len(pairs)} contrastive instances from {n_prompts} prompts")
    n_syco = sum(1 for p in pairs if p["label"] == 1)
    n_honest = sum(1 for p in pairs if p["label"] == 0)
    print(f"  Sycophantic: {n_syco}, Non-sycophantic: {n_honest}")
    
    # Print A/B balance
    syco_a = sum(1 for p in pairs if p["label"] == 1 and "(A)" in p["completion"])
    syco_b = sum(1 for p in pairs if p["label"] == 1 and "(B)" in p["completion"])
    print(f"  Syco A/B: {syco_a}/{syco_b}")
    
    return pairs


def _balance_ab_tokens(pairs: list[dict], seed: int) -> list[dict]:
    """
    Balance A/B tokens within each class by swapping question options.
    
    The idea: for 50% of questions, swap the options in the question text
    so that the "same" conceptual answer becomes the opposite letter.
    """
    random.seed(seed + 100)
    
    # Group pairs by their original question
    from collections import defaultdict
    question_groups = defaultdict(list)
    for pair in pairs:
        # Use first 100 chars as key to group pairs from same question
        key = pair["prompt"][:100]
        question_groups[key].append(pair)
    
    balanced_pairs = []
    swap_count = 0
    
    for key, group in question_groups.items():
        # Randomly decide whether to swap this question's A/B
        should_swap = random.random() < 0.5
        
        for pair in group:
            if should_swap:
                # Swap A <-> B in the completion
                new_completion = pair["completion"].replace("(A)", "(X)").replace("(B)", "(A)").replace("(X)", "(B)")
                pair["completion"] = new_completion
                swap_count += 1
            balanced_pairs.append(pair)
    
    print(f"  A/B balancing: swapped {swap_count}/{len(pairs)} completions")
    return balanced_pairs




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
