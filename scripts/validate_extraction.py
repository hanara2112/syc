
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
    # Handle 0-d array wrapping (np.savez behavior for lists)
    if metadata.shape == ():
        metadata = metadata.item()
    
    print(f"Metadata type: {type(metadata)}")
    try:
        if isinstance(metadata, np.ndarray):
            conditions = np.array([m['condition'] for m in metadata])
            target_tokens = np.array([m['target_token'].strip() for m in metadata])
        else:
             conditions = np.array([m['condition'] for m in metadata])
             target_tokens = np.array([m['target_token'].strip() for m in metadata])
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        print(f"Sample metadata item: {metadata[0] if len(metadata)>0 else 'Empty'}")
        return

    print(f"Conditions distribution: {np.unique(conditions, return_counts=True)}")
    
    # Define labels
    is_sycophantic = np.isin(conditions, ['A', 'C'])
    labels_behavior = np.where(is_sycophantic, 'Sycophantic', 'Non-Sycophantic')
    print(f"Sycophantic count: {np.sum(is_sycophantic)}")
    print(f"Non-Sycophantic count: {np.sum(~is_sycophantic)}")
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track metrics
    layer_norms = []
    
    # 1. Calculate Signal Strength for ALL layers first
    print("\ncalculating signal strength per layer...")
    for layer in layers:
        acts = data[f"layer_{layer}"]
        
        # DEBUG CHECKS
        if np.all(acts == 0):
            print(f"WARNING: Layer {layer} activations are ALL ZEROS.")
        
        mean_syco = acts[is_sycophantic].mean(axis=0)
        mean_non_syco = acts[~is_sycophantic].mean(axis=0)
        
        diff = mean_syco - mean_non_syco
        norm = np.linalg.norm(diff)
        layer_norms.append(norm)
        # print(f"Layer {layer}: {norm:.4f}")

    # 2. Plot Signal Strength Summary
    plt.figure(figsize=(10, 6))
    plt.plot(layers, layer_norms, marker='o', label='Sycophancy Signal (Norm of Diff)')
    plt.xlabel('Layer Index')
    plt.ylabel('Signal Strength (L2 Norm)')
    plt.title('Sycophancy Signal Strength across Layers')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "signal_strength_summary.png"))
    plt.close()
    
    # 3. Identify Best Layer
    best_layer_idx = np.argmax(layer_norms)
    best_layer = layers[best_layer_idx]
    print(f"\nMax Signal found at Layer {best_layer} (Norm: {layer_norms[best_layer_idx]:.4f})")
    
    # 4. Generate PCA ONLY for Best Layer
    print(f"Generating PCA plot for best layer: {best_layer}")
    acts = data[f"layer_{best_layer}"]
    
    pca = PCA(n_components=2)
    projected = pca.fit_transform(acts)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Behavior (Signal)
    for label in ['Sycophantic', 'Non-Sycophantic']:
        mask = labels_behavior == label
        axes[0].scatter(projected[mask, 0], projected[mask, 1], label=label, alpha=0.6)
    axes[0].set_title(f"Best Layer {best_layer}: By Behavior")
    axes[0].legend()
    
    # Plot 2: Token (Confound)
    unique_tokens = np.unique(target_tokens)
    for token in unique_tokens:
        mask = target_tokens == token
        axes[1].scatter(projected[mask, 0], projected[mask, 1], label=token, alpha=0.6)
    axes[1].set_title(f"Best Layer {best_layer}: By Token")
    axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, f"pca_best_layer_{best_layer}.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
