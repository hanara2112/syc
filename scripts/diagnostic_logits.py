import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from tqdm import tqdm

def check_logit_shift(model_name="Qwen/Qwen2.5-7B", layer_idx=20):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    
    # Load first chunk to get metadata and vector
    data = np.load("results/activations_full_chunk_0.npz", allow_pickle=True)
    meta = data['metadata']
    if meta.shape == (): meta = meta.item()
    
    # Calculate syc vector for this chunk
    conds = np.array([m['condition'] for m in meta])
    acts = data[f"layer_{layer_idx}"]
    if acts.dtype == 'object': acts = np.vstack(acts).astype(np.float32)
    
    mu_a = acts[np.where(conds == 'A')[0]].mean(axis=0)
    mu_b = acts[np.where(conds == 'B')[0]].mean(axis=0)
    v_syc = torch.tensor(mu_a - mu_b, dtype=torch.float16, device=model.device)
    v_syc = v_syc / v_syc.norm()

    prompt = meta[0]['prompt']
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    def get_top_tokens(alpha, top_n=10):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] >= prompt_len:
                h[:, prompt_len-1, :] += alpha * v_syc
            return (h,) + output[1:] if isinstance(output, tuple) else h
        
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
        handle.remove()
        
        # Get Top N
        top_indices = torch.topk(logits, top_n).indices.tolist()
        top_info = []
        for idx in top_indices:
            token = tokenizer.decode([idx])
            top_info.append(f"'{token}' ({logits[idx].item():.2f})")
            
        # Check specific tokens
        labels = ["A", "B", " (A", " (B", "(A", "(B", " Answer"]
        spec_info = {}
        for l in labels:
            ids = tokenizer.encode(l, add_special_tokens=False)
            if ids:
                spec_info[l] = logits[ids[-1]].item()
                
        return top_info, spec_info

    print(f"\nPrompt: ...{prompt[-100:]}")
    
    for alpha in [0.0, 80.0, 200.0, -80.0]:
        top_tokens, specs = get_top_tokens(alpha)
        print(f"\n--- Alpha {alpha} ---")
        print(f"Top 10: {', '.join(top_tokens)}")
        print("Choice Logits:")
        sorted_specs = sorted(specs.items(), key=lambda x: x[1], reverse=True)
        for k, v in sorted_specs:
            print(f"  {k:7}: {v:.4f}")

if __name__ == "__main__":
    check_logit_shift()
