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
    print(f"Loading activations from {path}...")
    data = np.load(path, allow_pickle=True)
    metadata = data['metadata']
    if metadata.shape == (): metadata = metadata.item()
    return data, metadata

def get_sycophancy_vector(data, metadata, layer):
    layer_key = f"layer_{layer}"
    if layer_key not in data:
        raise ValueError(f"Layer {layer} not found in data.")
    
    acts = data[layer_key]
    if acts.dtype == 'object': 
        acts = np.vstack(acts).astype(np.float32)

    conditions = np.array([m['condition'] for m in metadata])
    idx_A = np.where(conditions == 'A')[0] # Lib Agree
    idx_B = np.where(conditions == 'B')[0] # Lib Disagree
    idx_C = np.where(conditions == 'C')[0] # Con Agree
    idx_D = np.where(conditions == 'D')[0] # Con Disagree
    
    mu_A = acts[idx_A].mean(axis=0)
    mu_B = acts[idx_B].mean(axis=0)
    mu_C = acts[idx_C].mean(axis=0)
    mu_D = acts[idx_D].mean(axis=0)
    
    # Sycophancy Vector: (Agree - Disagree)
    v_syco = (mu_A + mu_C)/2 - (mu_B + mu_D)/2
    return v_syco

def classify_response(response, target_agree="A", target_disagree="B"):
    """
    Simple classifier to check if the response starts with the target token 
    or indicates agreement/disagreement.
    """
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
        # h shape: [Batch, SeqLen, Hidden]
        h = output[0] if isinstance(output, tuple) else output
        
        # During prefill, h.shape[1] == prompt_len
        # During decoding, h.shape[1] == 1
        if h.shape[1] >= prompt_len:
            # Prefill phase: record norm and apply steering
            with torch.no_grad():
                res_norms.append(h[0, prompt_len-1, :].norm().item())
            
            h[:, prompt_len - 1, :] += alpha * v
            
        return (h,) + output[1:] if isinstance(output, tuple) else h

    # Get the transformer block directly (Residual Block output)
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
    parser.add_argument("--activations_path", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    # Alpha range increased based on critique
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="results/behavioral_steering_sweep.csv")
    args = parser.parse_args()

    # 1. Load Data
    data, metadata = load_activations(args.activations_path)
    
    # We test on "Disagree" samples (B or D) to see if we can force them into "Agree"
    test_samples = [m for m in metadata if m['condition'] in ['B', 'D']]
    # Pick diverse samples if possible
    test_samples = test_samples[:args.num_samples]
    print(f"Testing on {len(test_samples)} samples (Initial condition: Disagree)")

    # 2. Load Model
    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    model.eval()

    results = []

    for layer in args.layers:
        print(f"\nEvaluating Layer {layer}...")
        v_syc = get_sycophancy_vector(data, metadata, layer)
        
        for alpha in args.alphas:
            print(f"  Alpha {alpha:5.1f}: ", end="", flush=True)
            flips = 0
            unknowns = 0
            norm_ratios = []
            
            for item in tqdm(test_samples, leave=False):
                response, res_norm = steer_and_generate(
                    model, tokenizer, item['prompt'], layer, v_syc, alpha
                )
                
                label = classify_response(response, target_agree="A", target_disagree="B")
                if label == "Agree": flips += 1
                elif label == "Unknown": unknowns += 1
                
                if res_norm > 0:
                    norm_ratios.append(alpha / res_norm)
            
            flip_rate = flips / len(test_samples)
            avg_ratio = np.mean(norm_ratios) if norm_ratios else 0
            
            print(f"Flip Rate: {flip_rate*100:5.1f}% | Alpha/ResNorm: {avg_ratio:.3f}")
            
            results.append({
                "layer": layer,
                "alpha": alpha,
                "flip_rate": flip_rate,
                "alpha_res_ratio": avg_ratio,
                "unknown_rate": unknowns / len(test_samples)
            })

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nBehavioral results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
