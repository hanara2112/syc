import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_dataset(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def find_minimal_pairs(data):
    """
    Groups data by question_idx and finds Agree/Disagree persona pairs.
    """
    pairs = []
    # Key: question_idx
    questions = {}
    for item in data:
        q_idx = item['question_id']
        if q_idx not in questions:
            questions[q_idx] = {'A': None, 'B': None, 'C': None, 'D': None}
        questions[q_idx][item['condition']] = item
    
    for q_idx, conds in questions.items():
        # Pair A (Liberal Agree) with B (Liberal Disagree)
        if conds['A'] and conds['B']:
            pairs.append((conds['A'], conds['B']))
        # Pair C (Conservative Agree) with D (Conservative Disagree)
        if conds['C'] and conds['D']:
            pairs.append((conds['C'], conds['D']))
    return pairs

def patch_and_measure(model, tokenizer, source_prompt, target_prompt, layer_idx, target_ids_A, target_ids_B):
    """
    Runs source_prompt to get h_L, then runs target_prompt while patching in h_L.
    """
    # 1. Get Source Activation
    source_inputs = tokenizer(source_prompt, return_tensors="pt").to(model.device)
    source_len = source_inputs.input_ids.shape[1]
    
    source_h = None
    def source_hook(module, input, output):
        nonlocal source_h
        h = output[0] if isinstance(output, tuple) else output
        source_h = h[:, source_len - 1, :].detach().clone()
        return output

    handle = model.model.layers[layer_idx].register_forward_hook(source_hook)
    with torch.no_grad():
        model(**source_inputs)
    handle.remove()

    # 2. Run Target with Patch
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(model.device)
    target_len = target_inputs.input_ids.shape[1]

    def patch_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] >= target_len:
            h[:, target_len - 1, :] = source_h
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = model.model.layers[layer_idx].register_forward_hook(patch_hook)
    with torch.no_grad():
        outputs = model(**target_inputs)
        logits = outputs.logits[0, -1, :]
        
        max_A = logits[target_ids_A].max().item()
        max_B = logits[target_ids_B].max().item()
        margin = max_A - max_B
    handle.remove()
    
    return margin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset_path", type=str, default="data/sycophancy_factorial_dataset.jsonl")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    parser.add_argument("--num_pairs", type=int, default=50)
    parser.add_argument("--output_csv", type=str, default="results/patching_results.csv")
    args = parser.parse_args()

    # 1. Load and Pair Data
    data = load_dataset(args.dataset_path)
    pairs = find_minimal_pairs(data)
    pairs = pairs[:args.num_pairs]
    print(f"Found {len(pairs)} minimal pairs for patching.")

    # 2. Load Model
    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    # Target Families
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
        print(f"\nPatching at Layer {layer}...")
        margins = []
        baseline_margins = []
        flips = 0
        
        for source_item, target_item in tqdm(pairs, leave=False):
            # Baseline (Target run without patch)
            target_inputs = tokenizer(target_item['prompt'], return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**target_inputs)
                logits = outputs.logits[0, -1, :]
                b_max_A = logits[target_ids_A].max().item()
                b_max_B = logits[target_ids_B].max().item()
                b_margin = b_max_A - b_max_B
            baseline_margins.append(b_margin)

            # Patch (Source is Agree persona A or C, Target is Disagree persona B or D)
            # source_item is always the 'Agree' version, target_item is 'Disagree'
            p_margin = patch_and_measure(model, tokenizer, source_item['prompt'], target_item['prompt'], layer, target_ids_A, target_ids_B)
            margins.append(p_margin)
            
            # If baseline was B-favored (margin < 0) and patch is A-favored (margin > 0)
            if b_margin < 0 and p_margin > 0:
                flips += 1
            # Or if baseline was A-favored and patch is B-favored (though we patch Agree -> Disagree)
            elif b_margin > 0 and p_margin < 0:
                flips += 1

        avg_baseline = np.mean(baseline_margins)
        avg_patched = np.mean(margins)
        flip_rate = flips / len(pairs)
        
        print(f"  Avg Baseline Margin: {avg_baseline:7.3f}")
        print(f"  Avg Patched Margin : {avg_patched:7.3f}")
        print(f"  $\Delta$Margin      : {avg_patched - avg_baseline:7.3f}")
        print(f"  Flip Rate          : {flip_rate*100:5.1f}%")
        
        results.append({
            "layer": layer,
            "avg_baseline": avg_baseline,
            "avg_patched": avg_patched,
            "delta_margin": avg_patched - avg_baseline,
            "flip_rate": flip_rate
        })

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
