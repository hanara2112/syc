import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Professional Academic Styling
plt.rcParams.update({
    "text.usetex": False, # Set to True if system has LaTeX
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.autolayout": True
})

def plot_refined_figures(csv_path, geom_path, output_dir="results/plots"):
    import os
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    geom_df = pd.read_csv(geom_path)
    
    # Merge for correlation
    df = df.merge(geom_df[['layer', 'norm_syc']], on='layer')
    
    # Normalize for visual comparison
    df['norm_sensitivity'] = -df['avg_sensitivity']  # Flip so UP is BETTER/CAUSAL
    df['norm_sensitivity'] = (df['norm_sensitivity'] - df['norm_sensitivity'].min()) / (df['norm_sensitivity'].max() - df['norm_sensitivity'].min())
    df['norm_logit'] = (df['logit_A_score'] - df['logit_A_score'].min()) / (df['logit_A_score'].max() - df['logit_A_score'].min())
    df['norm_vec'] = (df['norm_syc'] - df['norm_syc'].min()) / (df['norm_syc'].max() - df['norm_syc'].min())

    # --- THE CORE INSIGHT: DUAL AXIS CLEAN ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Shade Regions
    ax.axvspan(14, 18.5, color='gray', alpha=0.1, label='Emergence')
    ax.axvspan(18.5, 21.5, color='green', alpha=0.08, label='Causal Window')
    ax.axvspan(21.5, 27, color='red', alpha=0.05, label='Readout / Artifact')

    sns.lineplot(data=df, x="layer", y="norm_vec", color="#7f8c8d", linestyle="--", alpha=0.6, ax=ax, label="Feature Norm (L2)")
    sns.lineplot(data=df, x="layer", y="norm_logit", color="#2980b9", marker="s", ax=ax, label="Logit Alignment (Decodability)")
    sns.lineplot(data=df, x="layer", y="norm_sensitivity", color="#c0392b", marker="o", linewidth=3, ax=ax, label="Causal Control")

    # Annotations
    ax.text(16, 0.9, 'Emergence', horizontalalignment='center', color='gray', fontweight='bold')
    ax.text(20, 0.9, 'CONTROL', horizontalalignment='center', color='green', fontweight='bold')
    ax.text(24.5, 0.9, 'READOUT', horizontalalignment='center', color='red', fontweight='bold')

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Normalized Score [0, 1]")
    ax.set_title("The Representation–Control Dissociation in Qwen 2.5-7B")
    
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0.1))
    sns.despine(offset=10)
    plt.savefig(f"{output_dir}/dissociation_master_plot.png", dpi=300)
    print(f"Saved: {output_dir}/dissociation_master_plot.png")

if __name__ == "__main__":
    plot_refined_figures("results/Causal Diagnostics Sweep.csv", "results/Geometry Master 7k.csv")
