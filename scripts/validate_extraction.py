
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Validate sycophancy extraction via PCA.")
    parser.add_argument("--activations_path", type=str, required=True, help="Path to the activations .npz file")
    parser.add_argument("--output_dir", type=str, default="results/validation", help="Directory to save plots")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading activations from {args.activations_path}")
    data = np.load(args.activations_path, allow_pickle=True)
    
    metadata = data['metadata']
    layers = data['layers']
    
    # Process metadata
    conditions = np.array([m['condition'] for m in metadata]) # A, B, C, D
    target_tokens = np.array([m['target_token'].strip() for m in metadata]) # (A), (B)
    
    # Define labels
    # Sycophantic (Agree): Conditions A (Lib-Agree) and C (Con-Agree)
    # Non-Sycophantic (Disagree): Conditions B (Lib-Disagree) and D (Con-Disagree)
    is_sycophantic = np.isin(conditions, ['A', 'C'])
    labels_behavior = np.where(is_sycophantic, 'Sycophantic', 'Non-Sycophantic')
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    for layer in layers:
        acts = data[f"layer_{layer}"]
        
        # PCA
        pca = PCA(n_components=2)
        projected = pca.fit_transform(acts)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Behavior (Signal)
        for label in ['Sycophantic', 'Non-Sycophantic']:
            mask = labels_behavior == label
            axes[0].scatter(projected[mask, 0], projected[mask, 1], label=label, alpha=0.6)
        axes[0].set_title(f"Layer {layer}: By Behavior (Expect Separation)")
        axes[0].legend()
        
        # Plot 2: Token (Confound)
        unique_tokens = np.unique(target_tokens)
        for token in unique_tokens:
            mask = target_tokens == token
            axes[1].scatter(projected[mask, 0], projected[mask, 1], label=token, alpha=0.6)
        axes[1].set_title(f"Layer {layer}: By Token (Expect Separation on Diff Axis)")
        axes[1].legend()
        
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"pca_layer_{layer}.png")
        plt.savefig(save_path)
        plt.close()
        
    print(f"Validation plots saved to {args.output_dir}")
    
    # Vector Calculation Check
    # Sycophancy Direction = (Mean(A+C) - Mean(B+D))
    # We can check the norm of this vector per layer to see where signal peaks
    print("\nSycophancy Signal Strength (Norm of Diff-in-Means):")
    for layer in layers:
        acts = data[f"layer_{layer}"]
        
        mean_syco = acts[is_sycophantic].mean(axis=0)
        mean_non_syco = acts[~is_sycophantic].mean(axis=0)
        
        diff = mean_syco - mean_non_syco
        norm = np.linalg.norm(diff)
        print(f"Layer {layer}: {norm:.4f}")

if __name__ == "__main__":
    main()
