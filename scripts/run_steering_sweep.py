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

def classify_response(response, target_agree="A", target_disagree="B"):
    res = response.strip()
    if not res: return "Unknown"
    if res.startswith(f"({target_agree})") or res.startswith(target_agree) or res.startswith(f" {target_agree}"): return "Agree"
    if res.startswith(f"({target_disagree})") or res.startswith(target_disagree) or res.startswith(f" {target_disagree}"): return "Disagree"
    upper_res = res.upper()
    if "YES" in upper_res[:10] or "TRUE" in upper_res[:10] or "(A)" in upper_res[:5]: return "Agree"
    if "NO" in upper_res[:10] or "FALSE" in upper_res[:10] or "(B)" in upper_res[:5]: return "Disagree"
    return "Unknown"

def steer_and_generate(model, tokenizer, prompt, layers_and_vectors, alpha, max_new_tokens=5):
    """
    layers_and_vectors: list of (layer_idx, vector)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    handles = []
    
    def get_hook(v):
        v_torch = torch.tensor(v, dtype=torch.float16, device=model.device)
        v_torch = v_torch / v_torch.norm()
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == prompt_len:
                h[:, prompt_len - 1, :] += alpha * v_torch
            elif h.shape[1] == 1:
                h[:, 0, :] += alpha * v_torch
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return hook

    for layer_idx, v in layers_and_vectors:
        target_layer = model.model.layers[layer_idx]
        handles.append(target_layer.register_forward_hook(get_hook(v)))
    
    try:
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, 
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(gen_outputs[0][prompt_len:], skip_special_tokens=True)
    finally:
        for h in handles: h.remove()
        
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--layer_start", type=int, default=15)
    parser.add_argument("--layer_end", type=int, default=25)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 40.0, 80.0, -40.0, -80.0])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="results/behavioral_steering_ml.csv")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # 1. Load Samples
    chunk_files = sorted([f for f in os.listdir(args.activations_dir) if f.startswith("activations_full_chunk_")])
    _, metadata = load_activations(os.path.join(args.activations_dir, chunk_files[0]))
    test_samples = [m for m in metadata if m['condition'] in ['B', 'D']][:args.num_samples]
    print(f"Testing on {len(test_samples)} samples using Multi-Layer Steering (Layers {args.layer_start}-{args.layer_end})")

    # 2. Pre-calculate global vectors for the range
    layers_and_vectors = []
    print("Calculating global vectors for range...")
    for l in tqdm(range(args.layer_start, args.layer_end + 1)):
        v = get_global_sycophancy_vector(args.activations_dir, l)
        layers_and_vectors.append((l, v))

    # 3. Load Model
    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    results = []
    
    print("Establishing Baselines (Alpha=0.0)...")
    base_labels = []
    for i, item in enumerate(tqdm(test_samples, leave=False)):
        resp = steer_and_generate(model, tokenizer, item['prompt'], layers_and_vectors, 0.0)
        label = classify_response(resp)
        base_labels.append(label)
        if args.debug and i < 2: print(f"\nBaseline: {resp} -> {label}")

    for alpha in args.alphas:
        if alpha == 0.0:
            results.append({"alpha": 0.0, "flip_rate": 0.0, "unknown_rate": base_labels.count("Unknown")/len(test_samples)})
            continue
            
        print(f"  Alpha {alpha:5.1f}: ", end="", flush=True)
        flips, unknowns = 0, 0
        for i, item in enumerate(tqdm(test_samples, leave=False)):
            resp = steer_and_generate(model, tokenizer, item['prompt'], layers_and_vectors, alpha)
            label = classify_response(resp)
            if args.debug and i < 2: print(f"\nSteered ({alpha}): {resp} -> {label}")
            if label != base_labels[i] and label != "Unknown" and base_labels[i] != "Unknown": flips += 1
            if label == "Unknown": unknowns += 1
        
        fr = flips / len(test_samples)
        print(f"True Flip Rate: {fr*100:5.1f}%")
        results.append({"alpha": alpha, "flip_rate": fr, "unknown_rate": unknowns / len(test_samples)})

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"\nMulti-Layer Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
