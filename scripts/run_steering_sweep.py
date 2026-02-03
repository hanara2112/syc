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

def get_target_token_id(tokenizer, token_str="A"):
    # Robustly find the ID for the target token
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    # Often Qwen tokenizes "A" as a single token, but " (A)" as multiple.
    # We use the first ID that looks like the character.
    return ids[0]

def steer_and_measure(model, tokenizer, prompt, target_token_id, layer_idx, vector, alpha):
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    pos = inputs.input_ids.shape[1] - 1
    
    v = torch.tensor(vector, dtype=torch.float16, device=model.device)
    v = v / v.norm() # Unit vector for consistent alpha scaling

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, pos, :] += alpha * v
            return (h,) + output[1:]
        else:
            output[:, pos, :] += alpha * v
            return output

    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(steering_hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :] # [Vocab]
        probs = F.softmax(logits, dim=-1)
        
        target_prob = probs[target_token_id].item()
        target_logit = logits[target_token_id].item()

    handle.remove()
    return target_prob, target_logit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--activations_path", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--alphas", type=float, nargs="+", default=[-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0])
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_csv", type=str, default="results/steering_sweep_results.csv")
    args = parser.parse_args()

    # 1. Load Data
    data, metadata = load_activations(args.activations_path)
    
    # Use only "Disagree" conditions to see how much we can flip them to "Agree"
    test_samples = [m for m in metadata if m['condition'] in ['B', 'D']][:args.num_samples]
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

    target_id = get_target_token_id(tokenizer, "A")
    print(f"Target Token: 'A' (ID: {target_id})")

    results = []

    for layer in args.layers:
        print(f"\nSteering at Layer {layer}...")
        v_syc = get_sycophancy_vector(data, metadata, layer)
        
        for alpha in args.alphas:
            print(f"  Alpha: {alpha}")
            sample_probs = []
            sample_logits = []
            
            for item in tqdm(test_samples, leave=False):
                prob, logit = steer_and_measure(
                    model, tokenizer, item['prompt'], target_id, layer, v_syc, alpha
                )
                sample_probs.append(prob)
                sample_logits.append(logit)
            
            results.append({
                "layer": layer,
                "alpha": alpha,
                "avg_prob": np.mean(sample_probs),
                "avg_logit": np.mean(sample_logits)
            })

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

if __name__ == "__main__":
    main()
