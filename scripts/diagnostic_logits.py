import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

def check_logit_shift(model_name="Qwen/Qwen2.5-7B", layer_idx=20):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    
    # Load first chunk to get metadata and vector
    data = np.load("results/activations_full_chunk_0.npz", allow_pickle=True)
    meta = data['metadata']
    if meta.shape == (): meta = meta.item()
    
    v = data[f"layer_{layer_idx}"][0] # Just pick one activation to get shape
    # Actually calculate syc vector for this chunk
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
    
    id_A = tokenizer.encode("A")[-1]
    id_B = tokenizer.encode("B")[-1]

    def get_logits(alpha):
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
        return logits[id_A].item(), logits[id_B].item()

    print(f"\nPrompt: {prompt[-100:]}")
    a_base, b_base = get_logits(0.0)
    print(f"Baseline (Alpha=0): Logit A: {a_base:.4f}, Logit B: {b_base:.4f} (Diff: {a_base-b_base:.4f})")
    
    for alpha in [40.0, 80.0, -40.0, -80.0]:
        a_steer, b_steer = get_logits(alpha)
        print(f"Steer (Alpha={alpha}): Logit A: {a_steer:.4f}, Logit B: {b_steer:.4f} (Diff: {a_steer-b_steer:.4f}) | Shift: {a_steer - a_base:.4f}")

if __name__ == "__main__":
    check_logit_shift()
