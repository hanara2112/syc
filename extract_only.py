"""
Extract and Save Activations Only

This script extracts activations at the last token of the prompt
and saves them for later analysis. No classification or evaluation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_NAME, SEED, RESULTS_DIR
from utils import set_seed, get_device, load_model_and_tokenizer
from data import load_sycophancy_dataset, create_contrastive_pairs
from extraction import extract_activations


def main():
    parser = argparse.ArgumentParser(description="Extract Activations Only")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset", type=str, default="political",
                        choices=["political", "philosophy", "nlp"])
    parser.add_argument("--n_prompts", type=int, default=500,
                        help="Number of prompts (creates 2x instances)")
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated layers or 'all'")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: activations_{dataset}.npz)")
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    RESULTS_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("ACTIVATION EXTRACTION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Prompts: {args.n_prompts} -> {args.n_prompts * 2} instances")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    
    # Determine layers
    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    
    print(f"Extracting from {len(layers)} layers")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    data = load_sycophancy_dataset(args.dataset)
    pairs = create_contrastive_pairs(data, args.n_prompts, seed=args.seed)
    
    # Extract activations for each layer
    print("\n" + "="*60)
    print("EXTRACTING ACTIVATIONS")
    print("="*60)
    
    all_activations = {}
    labels = None
    
    for layer_idx in layers:
        print(f"\nLayer {layer_idx}/{n_layers-1}")
        acts, lbls = extract_activations(
            model, tokenizer, pairs, layer_idx, device,
            desc=f"Layer {layer_idx}",
        )
        all_activations[f"layer_{layer_idx}"] = acts
        
        if labels is None:
            labels = lbls
    
    # Save
    output_file = args.output or f"activations_{args.dataset}.npz"
    output_path = RESULTS_DIR / output_file
    
    np.savez(
        output_path,
        labels=labels,
        layers=np.array(layers),
        n_samples=len(pairs),
        hidden_dim=hidden_dim,
        **all_activations,
    )
    
    print("\n" + "="*60)
    print("SAVED")
    print("="*60)
    print(f"File: {output_path}")
    print(f"Shape per layer: ({len(pairs)}, {hidden_dim})")
    print(f"Layers saved: {layers}")
    print(f"Labels: {sum(labels)} syco, {len(labels) - sum(labels)} non-syco")


if __name__ == "__main__":
    main()
