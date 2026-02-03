import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ultra-minimal Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def plot_steering_comparison(csv_path, output_path="results/plots/steering_sensitivity_comparison.png"):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty. Skipping plot.")
        return
    
    # Calculate baseline-subtracted logits to show relative shift
    df['logit_shift'] = 0.0
    for layer in df['layer'].unique():
        layer_subset = df[df['layer'] == layer]
        baseline_rows = layer_subset[layer_subset['alpha'] == 0.0]
        if not baseline_rows.empty:
            baseline = baseline_rows['avg_logit'].values[0]
            df.loc[df['layer'] == layer, 'logit_shift'] = df.loc[df['layer'] == layer, 'avg_logit'] - baseline
        else:
            # Fallback if alpha=0.0 is missing (should not happen with our sweep)
            df.loc[df['layer'] == layer, 'logit_shift'] = df.loc[df['layer'] == layer, 'avg_logit'] - df.loc[df['layer'] == layer, 'avg_logit'].iloc[0]

    plt.figure(figsize=(7, 5))
    
    # Plot Layer 20 (The Control Window)
    l20_data = df[df['layer'] == 20]
    plt.plot(l20_data['alpha'], l20_data['logit_shift'], 
             marker='o', markersize=8, linewidth=3, color='#c0392b', label='Layer 20 (CONTROL)')
    
    # Plot Layer 26 (The Readout)
    l26_data = df[df['layer'] == 26]
    plt.plot(l26_data['alpha'], l26_data['logit_shift'], 
             marker='s', markersize=8, linewidth=3, color='#7f8c8d', label='Layer 26 (READOUT)')

    # Minimalist labels
    plt.xlabel(r"Steering Intensity ($\alpha$)", fontsize=13)
    plt.ylabel(r"Logit Shift ($\Delta$ Logit)", fontsize=13)
    plt.title("Steering Efficacy: Control vs. Readout", fontsize=15, fontweight='bold', pad=20)
    
    # Annotate the "Slope" (Causal Gain)
    plt.text(7, 0.45, "High Causal Gain\n(Active Control)", color='#c0392b', fontweight='bold')
    plt.text(7, 0.05, "Low Causal Gain\n(Inert Readout)", color='#7f8c8d', fontweight='bold')

    plt.legend(frameon=False, loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved minimalist steering plot to: {output_path}")

if __name__ == "__main__":
    plot_steering_comparison("results/Steering Sweep L20 vs L26.csv")
