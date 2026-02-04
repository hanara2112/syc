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
    pairs = []
    questions = {}
    for item in data:
        q_id = item['question_id']
        if q_id not in questions:
            questions[q_id] = {'A': None, 'B': None, 'C': None, 'D': None}
        questions[q_id][item['condition']] = item
    
    for q_id, conds in questions.items():
        if conds['A'] and conds['B']:
            pairs.append((conds['A'], conds['B']))
        if conds['C'] and conds['D']:
            pairs.append((conds['C'], conds['D']))
    return pairs

def patch_and_measure(model, tokenizer, source_prompt, target_prompt, layer_idx, target_ids_A, target_ids_B, debug=False):
    source_inputs = tokenizer(source_prompt, return_tensors="pt").to(model.device)
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(model.device)
    
    source_len = source_inputs.input_ids.shape[1]
    target_len = target_inputs.input_ids.shape[1]

    if debug:
        print(f"\n      Source Length: {source_len}, Target Length: {target_len}")

    source_h = None
    def source_hook(module, input, output):
        nonlocal source_h
        h = output[0] if isinstance(output, tuple) else output
        # Patching at the last token of the prefill
        source_h = h[:, -1, :].detach().clone()
        return output

    handle = model.model.layers[layer_idx].register_forward_hook(source_hook)
    with torch.no_grad():
        model(**source_inputs)
    handle.remove()

    if source_h is None:
        raise ValueError("Source hook failed to capture activation")

    def patch_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        # During prefill, shape[1] is the sequence length
        if h.shape[1] == target_len:
            h[:, -1, :] = source_h
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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data = load_dataset(args.dataset_path)
    pairs = find_minimal_pairs(data)[:args.num_pairs]
    print(f"Found {len(pairs)} minimal pairs for patching.")

    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    def get_family_ids(tokens):
        ids = []
        for t in tokens:
            encoded = tokenizer.encode(t, add_special_tokens=False)
            if encoded: ids.append(encoded[-1])
        return list(set(ids))

    target_ids_A = get_family_ids(["A", " A", "(A", " (A", "A."])
    target_ids_B = get_family_ids(["B", " B", "(B", " (B", "B."])

    if args.debug:
        print(f"Target IDs A: {target_ids_A}")
        print(f"Target IDs B: {target_ids_B}")

    results = []
    for layer in args.layers:
        print(f"\nPatching at Layer {layer}...")
        margins, baseline_margins, flips = [], [], 0
        
        for i, (source_item, target_item) in enumerate(tqdm(pairs, leave=False)):
            # Baseline
            target_inputs = tokenizer(target_item['prompt'], return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**target_inputs)
                logits = outputs.logits[0, -1, :]
                b_max_A = logits[target_ids_A].max().item()
                b_max_B = logits[target_ids_B].max().item()
                b_margin = b_max_A - b_max_B
            baseline_margins.append(b_margin)

            # Patch
            p_margin = patch_and_measure(
                model, tokenizer, source_item['prompt'], target_item['prompt'], 
                layer, target_ids_A, target_ids_B, debug=(args.debug and i < 2)
            )
            margins.append(p_margin)
            
            if b_margin < 0 and p_margin > 0: flips += 1
            elif b_margin > 0 and p_margin < 0: flips += 1

        avg_baseline = np.mean(baseline_margins)
        avg_patched = np.mean(margins)
        flip_rate = flips / len(pairs)
        
        print(f"  Avg Baseline Margin: {avg_baseline:7.3f}")
        print(f"  Avg Patched Margin : {avg_patched:7.3f}")
        print(f"  Delta Margin      : {avg_patched - avg_baseline:7.3f}")
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
