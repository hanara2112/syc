"""
Sycophancy Activation Analysis

Visualizes the extracted activations to check:
1. Layer-wise AUROC (which layer has best signal?)
2. Projection distribution (is syco/honest separable?)
3. PCA visualization (do clusters form?)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# =============================================================================
# Load Data
# =============================================================================
print("Loading activations...")
data = np.load("results/results/activations_political.npz")

labels = data["labels"]
layers = data["layers"]
n_samples = len(labels)
hidden_dim = int(data["hidden_dim"])

print(f"Samples: {n_samples}")
print(f"Layers: {len(layers)}")
print(f"Hidden dim: {hidden_dim}")
print(f"Labels: {sum(labels)} syco, {n_samples - sum(labels)} non-syco")

# =============================================================================
# 1. Layer-wise AUROC
# =============================================================================
print("\n" + "="*60)
print("Computing Layer-wise AUROC...")
print("="*60)

aurocs = []
d_primes = []

for layer_idx in layers:
    acts = data[f"layer_{layer_idx}"]
    
    # Compute diff-in-means direction
    pos_acts = acts[labels == 1]
    neg_acts = acts[labels == 0]
    direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    # Project onto direction
    scores = acts @ direction
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    if auroc < 0.5:
        auroc = 1 - auroc
    aurocs.append(auroc)
    
    # d' (effect size)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    mean_diff = pos_scores.mean() - neg_scores.mean()
    pooled_std = np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
    d_prime = abs(mean_diff) / (pooled_std + 1e-10)
    d_primes.append(d_prime)

best_layer = layers[np.argmax(aurocs)]
best_auroc = max(aurocs)
print(f"Best layer: {best_layer} (AUROC: {best_auroc:.4f})")

# Plot AUROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(layers, aurocs, 'b-o', linewidth=2, markersize=4)
ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Strong')
ax.scatter([best_layer], [best_auroc], color='red', s=100, zorder=5)
ax.annotate(f'Best: L{best_layer}\n{best_auroc:.3f}', 
            xy=(best_layer, best_auroc), xytext=(best_layer+3, best_auroc-0.05),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Sycophancy Detection: Layer-wise AUROC', fontsize=14)
ax.set_ylim(0.4, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(layers, d_primes, 'g-o', linewidth=2, markersize=4)
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel("d' (Effect Size)", fontsize=12)
ax.set_title('Sycophancy Signal Strength by Layer', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/01_layer_auroc.png', dpi=150)
plt.close()
print("Saved: results/01_layer_auroc.png")

# =============================================================================
# 2. Projection Distribution (Best Layer)
# =============================================================================
print("\n" + "="*60)
print(f"Projection Distribution (Layer {best_layer})...")
print("="*60)

acts = data[f"layer_{best_layer}"]

# Compute direction
pos_acts = acts[labels == 1]
neg_acts = acts[labels == 0]
direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
direction = direction / (np.linalg.norm(direction) + 1e-10)

# Project
scores = acts @ direction
pos_scores = scores[labels == 1]
neg_scores = scores[labels == 0]

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(neg_scores, bins=40, alpha=0.6, color='#4ECDC4', label='Non-Sycophantic', density=True)
ax.hist(pos_scores, bins=40, alpha=0.6, color='#FF6B6B', label='Sycophantic', density=True)

# KDE overlays
x_range = np.linspace(min(scores) - 1, max(scores) + 1, 200)
try:
    kde_pos = gaussian_kde(pos_scores)
    kde_neg = gaussian_kde(neg_scores)
    ax.plot(x_range, kde_pos(x_range), color='red', linewidth=2)
    ax.plot(x_range, kde_neg(x_range), color='teal', linewidth=2)
except:
    pass

# Add separation metrics
d_prime_best = abs(pos_scores.mean() - neg_scores.mean()) / np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
ax.axvline(pos_scores.mean(), color='red', linestyle='--', alpha=0.7)
ax.axvline(neg_scores.mean(), color='teal', linestyle='--', alpha=0.7)

ax.set_xlabel('Projection onto Sycophancy Direction', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f"Projection Distribution (Layer {best_layer}) | d'={d_prime_best:.2f}, AUROC={best_auroc:.3f}", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/02_projection_distribution.png', dpi=150)
plt.close()
print("Saved: results/02_projection_distribution.png")

# =============================================================================
# 3. PCA Visualization (Best Layer)
# =============================================================================
print("\n" + "="*60)
print(f"PCA Visualization (Layer {best_layer})...")
print("="*60)

pca = PCA(n_components=2)
acts_2d = pca.fit_transform(acts)

fig, ax = plt.subplots(figsize=(10, 8))

scatter_neg = ax.scatter(acts_2d[labels == 0, 0], acts_2d[labels == 0, 1], 
                          c='#4ECDC4', alpha=0.5, s=20, label='Non-Sycophantic')
scatter_pos = ax.scatter(acts_2d[labels == 1, 0], acts_2d[labels == 1, 1], 
                          c='#FF6B6B', alpha=0.5, s=20, label='Sycophantic')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title(f'PCA of Activations (Layer {best_layer})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/03_pca_visualization.png', dpi=150)
plt.close()
print("Saved: results/03_pca_visualization.png")

# =============================================================================
# 4. Multi-layer Comparison
# =============================================================================
print("\n" + "="*60)
print("Multi-layer Projection Comparison...")
print("="*60)

# Compare early, mid, late layers
sample_layers = [5, 15, 25, best_layer]
sample_layers = sorted(set(sample_layers))

fig, axes = plt.subplots(1, len(sample_layers), figsize=(5*len(sample_layers), 4))

for i, layer_idx in enumerate(sample_layers):
    ax = axes[i]
    acts = data[f"layer_{layer_idx}"]
    
    pos_acts = acts[labels == 1]
    neg_acts = acts[labels == 0]
    direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    scores = acts @ direction
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    
    ax.hist(neg_scores, bins=30, alpha=0.6, color='#4ECDC4', label='Non-Syco', density=True)
    ax.hist(pos_scores, bins=30, alpha=0.6, color='#FF6B6B', label='Syco', density=True)
    
    d_prime = abs(pos_scores.mean() - neg_scores.mean()) / np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
    ax.set_title(f"Layer {layer_idx}\nd'={d_prime:.2f}, AUROC={aurocs[layer_idx]:.3f}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/04_layer_comparison.png', dpi=150)
plt.close()
print("Saved: results/04_layer_comparison.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Best Layer: {best_layer}")
print(f"Best AUROC: {best_auroc:.4f}")
print(f"Best d': {d_primes[best_layer]:.2f}")
print()
print("Interpretation:")
if best_auroc > 0.85:
    print("✅ Strong sycophancy signal detected!")
elif best_auroc > 0.7:
    print("⚠️  Moderate sycophancy signal. The fix helped but there's room for improvement.")
else:
    print("❌ Weak signal. Something might still be wrong.")

print("\nPlots saved to results/")
