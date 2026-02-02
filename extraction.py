"""
Activation Extraction for Sycophancy Detection

CRITICAL: This module extracts activations at the LAST TOKEN OF THE PROMPT.
This captures the model's "decision state" before it generates the answer,
isolating the sycophancy signal from the literal A/B token identity.
"""
import numpy as np
import torch
from tqdm import tqdm

from config import SYSTEM_MESSAGE, USER_SUFFIX


def build_chat_prompt(tokenizer, question: str) -> str:
    """
    Build a chat prompt using the tokenizer's template.
    
    Args:
        tokenizer: HuggingFace tokenizer
        question: The question text from the dataset
        
    Returns:
        Formatted prompt string ready for tokenization
    """
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": question + USER_SUFFIX},
    ]
    
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback for tokenizers without chat template
        return f"System: {SYSTEM_MESSAGE}\n\nUser: {question}{USER_SUFFIX}\n\nAssistant:"


def extract_activations(
    model,
    tokenizer,
    pairs: list[dict],
    layer_idx: int,
    device: torch.device,
    extraction_mode: str = "last",
    mean_pool_n: int = 5,
    desc: str = "Extracting",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract activations from model hidden states.
    
    Extraction modes:
    -----------------
    - "last": Last token of the full sequence (prompt + answer)
    - "first_answer": First token of the answer (transition state)
    - "mean_pool": Mean of last N tokens (more robust for manifold analysis)
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        pairs: List of contrastive pair dicts with 'prompt', 'completion', 'label'
        layer_idx: Which layer to extract from (0-indexed)
        device: Torch device
        extraction_mode: One of "last", "first_answer", "mean_pool"
        mean_pool_n: Number of tokens to pool when using "mean_pool"
        desc: Description for progress bar
        
    Returns:
        activations: np.ndarray of shape (n_samples, hidden_dim)
        labels: np.ndarray of shape (n_samples,)
    """
    activations = []
    labels = []
    
    for pair in tqdm(pairs, desc=desc, leave=False):
        # Build the prompt
        prompt_text = build_chat_prompt(tokenizer, pair["prompt"])
        
        # Include the completion/answer
        full_text = prompt_text + pair["completion"]
        
        # Tokenize prompt separately to know where answer starts
        prompt_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        prompt_length = prompt_inputs["input_ids"].shape[1]
        
        # Tokenize full sequence
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)
        
        total_length = inputs["input_ids"].shape[1]
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get hidden states for this layer
        hidden = outputs.hidden_states[layer_idx + 1]
        
        # Extract based on mode
        if extraction_mode == "last":
            # Last token of full sequence
            activation = hidden[0, -1, :].float().cpu().numpy()
            
        elif extraction_mode == "first_answer":
            # First token of the answer (transition state)
            # This is at position prompt_length (0-indexed)
            extract_pos = min(prompt_length, total_length - 1)
            activation = hidden[0, extract_pos, :].float().cpu().numpy()
            
        elif extraction_mode == "mean_pool":
            # Mean of last N tokens
            start_pos = max(0, total_length - mean_pool_n)
            activation = hidden[0, start_pos:, :].float().mean(dim=0).cpu().numpy()
            
        else:
            raise ValueError(f"Unknown extraction_mode: {extraction_mode}")
        
        activations.append(activation)
        labels.append(pair["label"])
    
    return np.array(activations), np.array(labels)


def compute_diff_in_means_direction(
    activations: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute the diff-in-means direction for sycophancy detection.
    
    Args:
        activations: np.ndarray of shape (n_samples, hidden_dim)
        labels: np.ndarray of shape (n_samples,) with values 0 or 1
        
    Returns:
        direction: Normalized direction vector of shape (hidden_dim,)
    """
    # Separate by class
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    pos_acts = activations[pos_mask]
    neg_acts = activations[neg_mask]
    
    # Compute means
    pos_mean = pos_acts.mean(axis=0)
    neg_mean = neg_acts.mean(axis=0)
    
    # Direction: sycophantic - non-sycophantic
    direction = pos_mean - neg_mean
    
    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction = direction / norm
    
    return direction


def project_onto_direction(
    activations: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Project activations onto a direction vector.
    
    Args:
        activations: np.ndarray of shape (n_samples, hidden_dim)
        direction: np.ndarray of shape (hidden_dim,)
        
    Returns:
        scores: np.ndarray of shape (n_samples,)
    """
    return activations @ direction
