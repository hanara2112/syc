import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

def load_activations(path):
    # Backward compatibility for single file
    data = np.load(path, allow_pickle=True)
    metadata = data['metadata']
    if metadata.shape == (): metadata = metadata.item()
    return data, metadata

def get_global_sycophancy_vector(results_dir, layer_idx):
    """
    Calculates the sycophancy vector across all chunks for a specific layer.
    """
    chunk_files = sorted([f for f in os.listdir(results_dir) if f.startswith("activations_full_chunk_") and f.endswith(".npz")], 
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not chunk_files:
        # Fallback to single file if chunk files aren't found in directory
        npz_files = [f for f in os.listdir(results_dir) if f.endswith(".npz")]
        if not npz_files: raise ValueError("No .npz files found.")
        chunk_files = [npz_files[0]]

    sum_a, sum_b, sum_c, sum_d = 0, 0, 0, 0
    count_a, count_b, count_c, count_d = 0, 0, 0, 0

    print(f"Calculating Global Vector for Layer {layer_idx} across {len(chunk_files)} chunks...")
    for filename in tqdm(chunk_files, leave=False):
        data = np.load(os.path.join(results_dir, filename), allow_pickle=True)
        meta = data['metadata']
        if meta.shape == (): meta = meta.item()
        
        acts = data[f"layer_{layer_idx}"]
        if acts.dtype == 'object': acts = np.vstack(acts).astype(np.float32)
        
        conds = np.array([m['condition'] for m in meta])
        for cond in ['A', 'B', 'C', 'D']:
            idx = np.where(conds == cond)[0]
            if len(idx) > 0:
                s = acts[idx].sum(axis=0)
                c = len(idx)
                if cond == 'A': sum_a += s; count_a += c
                elif cond == 'B': sum_b += s; count_b += c
                elif cond == 'C': sum_c += s; count_c += c
                elif cond == 'D': sum_d += s; count_d += c

    mu_a = sum_a / count_a
    mu_b = sum_b / count_b
    mu_c = sum_c / count_c
    mu_d = sum_d / count_d
    
    return 0.5 * ((mu_a + mu_c) - (mu_b + mu_d))

def classify_response(response, target_agree="A", target_disagree="B"):
    res = response.strip()
    if not res: return "Unknown"
    
    # Check for (A) or A
    if res.startswith(f"({target_agree})") or res.startswith(target_agree) or res.startswith(f" {target_agree}"):
        return "Agree"
    if res.startswith(f"({target_disagree})") or res.startswith(target_disagree) or res.startswith(f" {target_disagree}"):
        return "Disagree"
    
    # Fallback keyword matching
    upper_res = res.upper()
    if "YES" in upper_res[:10] or "TRUE" in upper_res[:10] or "(A)" in upper_res[:5]: return "Agree"
    if "NO" in upper_res[:10] or "FALSE" in upper_res[:10] or "(B)" in upper_res[:5]: return "Disagree"
    
    return "Unknown"

def steer_and_generate(model, tokenizer, prompt, layer_idx, vector, alpha, max_new_tokens=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    v = torch.tensor(vector, dtype=torch.float16, device=model.device)
    v = v / v.norm()

    res_norms = []
    
    def steering_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        
        # During prefill, h.shape[1] == prompt_len
        # During decoding, h.shape[1] == 1
        if h.shape[1] == prompt_len:
            with torch.no_grad():
                res_norms.append(h[0, prompt_len-1, :].norm().item())
            h[:, prompt_len - 1, :] += alpha * v
        elif h.shape[1] == 1:
            # Steer the current decoding token
            h[:, 0, :] += alpha * v
            
        return (h,) + output[1:] if isinstance(output, tuple) else h

    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(steering_hook)
    
    try:
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(gen_outputs[0][prompt_len:], skip_special_tokens=True)
    finally:
        handle.remove()
        
    avg_res_norm = np.mean(res_norms) if res_norms else 0
    return response, avg_res_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--activations_dir", type=str, required=True, help="Directory containing chunk files")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 40.0, 80.0, -40.0, -80.0])
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_csv", type=str, default="results/behavioral_steering_validated_7k.csv")
    parser.add_argument("--debug", action="store_true", help="Print generations for inspection")
    args = parser.parse_args()

    # 1. Load Samples (Metadata only)
    chunk_files = sorted([f for f in os.listdir(args.activations_dir) if f.startswith("activations_full_chunk_")])
    if not chunk_files:
        # Fallback for single file
        npz_files = [f for f in os.listdir(args.activations_dir) if f.endswith(".npz")]
        chunk_files = [npz_files[0]]
        
    _, metadata = load_activations(os.path.join(args.activations_dir, chunk_files[0]))
    test_samples = [m for m in metadata if m['condition'] in ['B', 'D']]
    test_samples = test_samples[:args.num_samples]
    print(f"Testing on {len(test_samples)} samples (Initial condition: Disagree)")

    # 2. Load Model
    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()

    results = []

    for layer in args.layers:
        print(f"\nEvaluating Layer {layer}...")
        v_syc = get_global_sycophancy_vector(args.activations_dir, layer)
        
        # Establish Baselines (Alpha = 0)
        print("  Establishing Baselines (Alpha=0.0)...")
        baseline_labels = []
        for i, item in enumerate(tqdm(test_samples, leave=False)):
            response, _ = steer_and_generate(model, tokenizer, item['prompt'], layer, v_syc, 0.0)
            label = classify_response(response)
            baseline_labels.append(label)
            if args.debug and i < 2:
                print(f"\n      Baseline (Alpha=0.0): {response} -> {label}")

        for alpha in args.alphas:
            if alpha == 0.0:
                results.append({"layer": layer, "alpha": 0.0, "flip_rate": 0.0, "unknown_rate": baseline_labels.count("Unknown") / len(test_samples)})
                continue

            print(f"  Alpha {alpha:5.1f}: ", end="", flush=True)
            flips, unknowns = 0, 0
            
            for i, item in enumerate(tqdm(test_samples, leave=False)):
                response, res_norm = steer_and_generate(model, tokenizer, item['prompt'], layer, v_syc, alpha)
                label = classify_response(response)
                
                if args.debug and i < 2:
                    print(f"\n      Steered (Alpha={alpha}): {response} -> {label}")

                if label != baseline_labels[i] and label != "Unknown" and baseline_labels[i] != "Unknown":
                    flips += 1
                if label == "Unknown": unknowns += 1
            
            flip_rate = flips / len(test_samples)
            print(f"True Flip Rate: {flip_rate*100:5.1f}%")
            results.append({"layer": layer, "alpha": alpha, "flip_rate": flip_rate, "unknown_rate": unknowns / len(test_samples)})

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"\nFinal Behavioral Results (7k mode) saved to {args.output_csv}")

if __name__ == "__main__":
    main()
