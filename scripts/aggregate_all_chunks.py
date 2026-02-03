import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def aggregate_chunks(results_dir="results", output_csv="results/geometry_master_7k.csv"):
    """
    Aggregates activations across 15 chunks to calculate global means and geometry.
    """
    npz_files = sorted([f for f in os.listdir(results_dir) if f.startswith("activations_full_chunk_") and f.endswith(".npz")], 
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not npz_files:
        print("No activations_full_chunk_*.npz files found.")
        return

    print(f"Aggregating {len(npz_files)} chunks...")
    
    # We need to accumulate Sums and Counts for each layer and cluster
    # cluster_sums[layer][condition] = sum_vector
    # cluster_counts[layer][condition] = N
    cluster_sums = {}
    cluster_counts = {}
    
    for filename in tqdm(npz_files, desc="Processing Chunks"):
        data = np.load(os.path.join(results_dir, filename), allow_pickle=True)
        metadata = data['metadata']
        if metadata.shape == (): metadata = metadata.item()
        
        conditions = np.array([m['condition'] for m in metadata])
        
        # Determine available layers in this chunk
        layer_keys = [k for k in data.keys() if k.startswith('layer_')]
        
        for key in layer_keys:
            layer_idx = int(key.split('_')[1])
            acts = data[key]
            if acts.dtype == 'object': 
                acts = np.vstack(acts).astype(np.float32)
            
            if layer_idx not in cluster_sums:
                cluster_sums[layer_idx] = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                cluster_counts[layer_idx] = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
            
            for cond in ['A', 'B', 'C', 'D']:
                idx = np.where(conditions == cond)[0]
                if len(idx) > 0:
                    cluster_sums[layer_idx][cond] += acts[idx].sum(axis=0)
                    cluster_counts[layer_idx][cond] += len(idx)

    # Calculate final means and geometry
    results = []
    layer_indices = sorted(cluster_sums.keys())
    
    for layer in tqdm(layer_indices, desc="Calculating Geometry"):
        mu = {}
        for cond in ['A', 'B', 'C', 'D']:
            if cluster_counts[layer][cond] > 0:
                mu[cond] = cluster_sums[layer][cond] / cluster_counts[layer][cond]
            else:
                mu[cond] = np.zeros_like(list(cluster_sums[layer].values())[0])

        # Sycophancy Vector: Agree - Disagree
        v_syc = 0.5 * ((mu['A'] + mu['C']) - (mu['B'] + mu['D']))
        # Partisanship Vector: Conservative - Liberal
        v_part = 0.5 * ((mu['C'] + mu['D']) - (mu['A'] + mu['B']))
        
        norm_syc = np.linalg.norm(v_syc)
        norm_part = np.linalg.norm(v_part)
        cos_sim = np.dot(v_syc, v_part) / (norm_syc * norm_part + 1e-10)
        
        results.append({
            "layer": layer,
            "norm_syc": norm_syc,
            "norm_part": norm_part,
            "cos_sim": cos_sim,
            "n_samples": sum(cluster_counts[layer].values())
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nFinal Master Geometry saved to {output_csv}")
    print(f"Total samples processed: {df['n_samples'].max()}")

if __name__ == "__main__":
    aggregate_chunks()
