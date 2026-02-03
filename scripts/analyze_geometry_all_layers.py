import numpy as np
import pandas as pd
import os

def calculate_geometry(data_path, output_csv):
    """
    Calculates sycophancy and partisanship vectors across all layers and their cosine similarity.
    """
    print(f"Loading activations from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    metadata = data['metadata']
    if metadata.shape == (): metadata = metadata.item()
    
    conditions = np.array([m['condition'] for m in metadata])
    
    # 2x2 Factorial Indices
    idx_LA = np.where(conditions == 'A')[0] # Liberal Agree
    idx_LD = np.where(conditions == 'B')[0] # Liberal Disagree
    idx_CA = np.where(conditions == 'C')[0] # Conservative Agree
    idx_CD = np.where(conditions == 'D')[0] # Conservative Disagree
    
    results = []
    
    # Iterate through all layers in the npz
    layer_keys = sorted([k for k in data.keys() if k.startswith('layer_')], key=lambda x: int(x.split('_')[1]))
    
    for key in layer_keys:
        layer_idx = int(key.split('_')[1])
        acts = data[key]
        if acts.dtype == 'object': 
            acts = np.vstack(acts).astype(np.float32)
            
        # Means for each cluster
        mu_LA = acts[idx_LA].mean(axis=0)
        mu_LD = acts[idx_LD].mean(axis=0)
        mu_CA = acts[idx_CA].mean(axis=0)
        mu_CD = acts[idx_CD].mean(axis=0)
        
        # 1. Sycophancy Vector: Agree - Disagree (averaged across personas)
        v_syc = 0.5 * ((mu_LA + mu_CA) - (mu_LD + mu_CD))
        
        # 2. Partisanship Vector: Conservative - Liberal (averaged across agreement)
        v_part = 0.5 * ((mu_CA + mu_CD) - (mu_LA + mu_LD))
        
        # 3. Geometric Metrics
        norm_syc = np.linalg.norm(v_syc)
        norm_part = np.linalg.norm(v_part)
        
        # Cosine Similarity
        cos_sim = np.dot(v_syc, v_part) / (norm_syc * norm_part + 1e-10)
        
        results.append({
            "layer": layer_idx,
            "norm_syc": norm_syc,
            "norm_part": norm_part,
            "cos_sim": cos_sim
        })
        
        print(f"Layer {layer_idx:2d}: Norm_Syc={norm_syc:.4f}, Norm_Part={norm_part:.4f}, Cos_Sim={cos_sim:.4f}")
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved geometric analysis to {output_csv}")

if __name__ == "__main__":
    # Look for any .npz in results/
    results_dir = "results"
    npz_files = [f for f in os.listdir(results_dir) if f.endswith(".npz")]
    if not npz_files:
        print("No .npz files found in results/")
    else:
        # Use the first one found
        target_path = os.path.join(results_dir, npz_files[0])
        calculate_geometry(target_path, "results/geometry_all_layers.csv")
