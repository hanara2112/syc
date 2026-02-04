import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_activations(path):
    data = np.load(path, allow_pickle=True)
    metadata = data['metadata']
    if metadata.shape == (): metadata = metadata.item()
    return data, metadata

def get_global_sycophancy_vector(results_dir, layer_idx):
    chunk_files = sorted([f for f in os.listdir(results_dir) if f.startswith("activations_full_chunk_") and f.endswith(".npz")], 
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not chunk_files:
        npz_files = [f for f in os.listdir(results_dir) if f.endswith(".npz")]
        if not npz_files: raise ValueError("No .npz files found.")
        chunk_files = [npz_files[0]]

    sum_a, sum_b, sum_c, sum_d = 0, 0, 0, 0
    count_a, count_b, count_c, count_d = 0, 0, 0, 0

    for filename in chunk_files:
        data = np.load(os.path.join(results_dir, filename), allow_pickle=True)
        meta = data['metadata']
        if meta.shape == (): meta = meta.item()
        
        acts = data[f"layer_{layer_idx}"]
        if acts.dtype == 'object': acts = np.vstack(acts).astype(np.float32)
        
        conds = np.array([m['condition'] for m in meta])
        for cond in ['A', 'B', 'C', 'D']:
            idx = np.where(conds == cond)[0]
            if len(idx) > 0:
                s = acts[idx].sum(axis=0); c = len(idx)
                if cond == 'A': sum_a += s; count_a += c
                elif cond == 'B': sum_b += s; count_b += c
                elif cond == 'C': sum_c += s; count_c += c
                elif cond == 'D': sum_d += s; count_d += c
    
    return 0.5 * ((sum_a/count_a + sum_c/count_c) - (sum_b/count_b + sum_d/count_d))

def get_logit_margin(model, tokenizer, prompt, layer_idx, vector, alpha, target_ids_A, target_ids_B):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    v = torch.tensor(vector, dtype=torch.float16, device=model.device)
    v = v / v.norm()

    def steering_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] >= prompt_len:
            h[:, prompt_len - 1, :] += alpha * v
        return (h,) + output[1:] if isinstance(output, tuple) else h

    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(steering_hook)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            # Margin = max(A_family) - max(B_family)
            max_A = logits[target_ids_A].max().item()
            max_B = logits[target_ids_B].max().item()
            margin = max_A - max_B
    finally:
        handle.remove()
        
    return margin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0, 20, 40, 60, 80])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_csv", type=str, default="results/logit_sweep_results.csv")
    args = parser.parse_args()

    # 1. Load Samples (Metadata only from first chunk)
    chunk_files = sorted([f for f in os.listdir(args.activations_dir) if f.startswith("activations_full_chunk_")])
    _, metadata = load_activations(os.path.join(args.activations_dir, chunk_files[0]))
    
    # We test on "Disagree" samples to see if we can push them towards Agree
    test_samples = [m for m in metadata if m['condition'] in ['B', 'D']][:args.num_samples]
    print(f"Running logit sweep on {len(test_samples)} samples...")

    # 2. Load Model
    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    # Prepare Target ID families
    # A-family: "A", " A", "(A", " (A"
    # B-family: "B", " B", "(B", " (B"
    def get_family_ids(tokens):
        ids = []
        for t in tokens:
            encoded = tokenizer.encode(t, add_special_tokens=False)
            if encoded: ids.append(encoded[-1])
        return list(set(ids))

    target_ids_A = get_family_ids(["A", " A", "(A", " (A"])
    target_ids_B = get_family_ids(["B", " B", "(B", " (B"])

    results = []

    for layer in args.layers:
        print(f"\nEvaluating Layer {layer}...")
        v_syc = get_global_sycophancy_vector(args.activations_dir, layer)
        
        for alpha in args.alphas:
            print(f"  Alpha {alpha:5.1f}: ", end="", flush=True)
            margins = []
            flips = 0

            for item in tqdm(test_samples, leave=False):
                margin = get_logit_margin(model, tokenizer, item['prompt'], layer, v_syc, alpha, target_ids_A, target_ids_B)
                margins.append(margin)
                if margin > 0: flips += 1
            
            avg_margin = np.mean(margins)
            flip_rate = flips / len(test_samples)
            print(f"Avg Margin: {avg_margin:7.3f} | Flip Rate: {flip_rate*100:5.1f}%")
            
            results.append({
                "layer": layer,
                "alpha": alpha,
                "avg_margin": avg_margin,
                "flip_rate": flip_rate
            })

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nLogit sweep results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
