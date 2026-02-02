"""
Layer Sweep for Sycophancy Detection

This script:
1. Loads the dataset and creates contrastive pairs
2. Sweeps through all (or specified) layers
3. For each layer:
   - Extracts train activations
   - Computes diff-in-means direction
   - Extracts test activations
   - Projects onto direction and computes AUROC
4. Saves results and identifies best layer
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODEL_NAME, SEED, N_TRAIN, N_TEST,
    LAYERS_TO_SWEEP, RESULTS_DIR,
)
from utils import set_seed, get_device, load_model_and_tokenizer
from data import load_sycophancy_dataset, create_contrastive_pairs, split_pairs
from extraction import (
    extract_activations,
    compute_diff_in_means_direction,
    project_onto_direction,
)


def run_layer_sweep(
    model,
    tokenizer,
    train_pairs: list[dict],
    test_pairs: list[dict],
    layers: list[int],
    device,
) -> dict:
    """
    Run the layer sweep experiment.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        train_pairs: Training contrastive pairs
        test_pairs: Test contrastive pairs
        layers: List of layer indices to sweep
        device: Torch device
        
    Returns:
        Dictionary with results for each layer
    """
    results = {}
    
    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")
        
        # Extract train activations
        train_acts, train_labels = extract_activations(
            model, tokenizer, train_pairs, layer_idx, device,
            desc=f"Train L{layer_idx}",
        )
        
        # Compute direction
        direction = compute_diff_in_means_direction(train_acts, train_labels)
        
        # Extract test activations
        test_acts, test_labels = extract_activations(
            model, tokenizer, test_pairs, layer_idx, device,
            desc=f"Test L{layer_idx}",
        )
        
        # Project and score
        test_scores = project_onto_direction(test_acts, direction)
        
        # Compute metrics
        try:
            auroc = roc_auc_score(test_labels, test_scores)
            # Handle flipped direction
            if auroc < 0.5:
                auroc = 1 - auroc
                test_scores = -test_scores
        except ValueError:
            auroc = 0.5
        
        # Accuracy with median threshold
        threshold = np.median(test_scores)
        preds = (test_scores > threshold).astype(int)
        accuracy = accuracy_score(test_labels, preds)
        
        # Class separation metrics
        pos_scores = test_scores[test_labels == 1]
        neg_scores = test_scores[test_labels == 0]
        mean_diff = pos_scores.mean() - neg_scores.mean()
        pooled_std = np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
        d_prime = mean_diff / pooled_std if pooled_std > 0 else 0
        
        results[layer_idx] = {
            "auroc": float(auroc),
            "accuracy": float(accuracy),
            "d_prime": float(d_prime),
            "mean_diff": float(mean_diff),
            "pos_mean": float(pos_scores.mean()),
            "neg_mean": float(neg_scores.mean()),
        }
        
        print(f"  AUROC: {auroc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  d': {d_prime:.2f}")
    
    return results


def plot_layer_sweep(results: dict, output_path: Path):
    """Create and save the layer sweep plot."""
    layers = sorted(results.keys())
    aurocs = [results[l]["auroc"] for l in layers]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, aurocs, 'b-o', linewidth=2, markersize=5)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
    
    # Mark best layer
    best_layer = max(results, key=lambda l: results[l]["auroc"])
    best_auroc = results[best_layer]["auroc"]
    ax.annotate(
        f'Best: L{best_layer}\nAUROC={best_auroc:.3f}',
        xy=(best_layer, best_auroc),
        xytext=(best_layer + 2, best_auroc - 0.05),
        arrowprops=dict(arrowstyle='->', color='green'),
        fontsize=10, color='green'
    )
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Sycophancy Detection: Layer-wise AUROC', fontsize=14)
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Layer Sweep for Sycophancy Detection")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--dataset", type=str, default="political", 
                        choices=["political", "philosophy", "nlp"],
                        help="Dataset to use")
    parser.add_argument("--n_train", type=int, default=N_TRAIN, help="Number of training prompts")
    parser.add_argument("--n_test", type=int, default=N_TEST, help="Number of test prompts")
    parser.add_argument("--layers", type=str, default=None, 
                        help="Comma-separated layer indices (e.g., '10,15,20')")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    RESULTS_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("SYCOPHANCY DETECTION - LAYER SWEEP")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Train prompts: {args.n_train} -> {args.n_train * 2} instances")
    print(f"Test prompts: {args.n_test} -> {args.n_test * 2} instances")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    n_layers = model.config.num_hidden_layers
    
    # Determine layers to sweep
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    elif LAYERS_TO_SWEEP:
        layers = LAYERS_TO_SWEEP
    else:
        layers = list(range(n_layers))
    
    print(f"Layers to sweep: {len(layers)} layers")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    raw_data = load_sycophancy_dataset(args.dataset)
    
    # Create enough pairs for train + test
    total_prompts = args.n_train + args.n_test
    pairs = create_contrastive_pairs(raw_data, total_prompts, seed=args.seed)
    
    # Split
    train_pairs, test_pairs = split_pairs(
        pairs, 
        n_train=args.n_train * 2,  # 2 instances per prompt
        n_test=args.n_test * 2,
        seed=args.seed,
    )
    
    # Run sweep
    print("\n" + "="*60)
    print("RUNNING LAYER SWEEP")
    print("="*60)
    
    results = run_layer_sweep(
        model, tokenizer, train_pairs, test_pairs, layers, device
    )
    
    # Find best layer
    best_layer = max(results, key=lambda l: results[l]["auroc"])
    best_auroc = results[best_layer]["auroc"]
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Best Layer: {best_layer}")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Best d': {results[best_layer]['d_prime']:.2f}")
    
    # Save results
    results_file = RESULTS_DIR / f"layer_sweep_{args.dataset}.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "dataset": args.dataset,
                "n_train_prompts": args.n_train,
                "n_test_prompts": args.n_test,
                "seed": args.seed,
            },
            "best_layer": best_layer,
            "best_auroc": best_auroc,
            "layer_results": results,
        }, f, indent=2)
    print(f"\nSaved results: {results_file}")
    
    # Plot
    plot_file = RESULTS_DIR / f"layer_sweep_{args.dataset}.png"
    plot_layer_sweep(results, plot_file)
    
    return results


if __name__ == "__main__":
    main()
