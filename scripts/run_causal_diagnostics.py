import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import sys

# TF settings - try to block it before it grabs the GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Force TF to CPU-only mode if it's already being imported
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass



def load_activations(path):
    print(f"Loading activations from {path}...")
    data = np.load(path, allow_pickle=True)
    metadata = data['metadata']
    if metadata.shape == (): metadata = metadata.item()
    return data, metadata

def get_sycophancy_vector(data, metadata, layer):
    """
    Calculates the sycophancy vector for a specific layer.
    v_syc = (Agree_Lib + Agree_Con)/2 - (Disagree_Lib + Disagree_Con)/2
    """
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
    # Averaging across personas to remove partisanship
    v_syco = (mu_A + mu_C)/2 - (mu_B + mu_D)/2
    
    return v_syco

def logit_projection_test(model, tokenizer, vector, layer_idx, top_k=20):
    """
    Projects the vector onto the unembedding matrix (W_out) to see top promoted tokens.
    """
    print(f"\n--- Logit Projection Test (Layer {layer_idx}) ---")
    
    # 1. Get Unembedding Matrix
    # Qwen/Llama: model.lm_head
    if not hasattr(model, "lm_head"):
        print("Error: Could not find lm_head. Skipping.")
        return

    W_out = model.lm_head.weight # [Vocab, Hidden]
    
    # 2. Project
    # vector shape: [Hidden]
    v = torch.tensor(vector, dtype=W_out.dtype, device=W_out.device)
    
    # Normalize vector for visualization consistency?
    # Usually better to keep raw to see raw effect size, but norm helps interpretation.
    # Let's use raw first.
    logits = torch.matmul(W_out, v) # [Vocab]
    
    # 3. Top Tokens (Promoted)
    top_vals, top_indices = torch.topk(logits, top_k)
    
    print(f"Top {top_k} Promoted Tokens (Positive direction = Agree):")
    for val, idx in zip(top_vals, top_indices):
        token = tokenizer.decode([idx])
        print(f"  {token:<20} | Logit: {val.item():.4f}")
        
    # 4. Bottom Tokens (Suppressed / Disagree)
    bot_vals, bot_indices = torch.topk(logits, top_k, largest=False)
    print(f"\nTop {top_k} Suppressed Tokens (Negative direction = Disagree):")
    for val, idx in zip(bot_vals, bot_indices):
        token = tokenizer.decode([idx])
        print(f"  {token:<20} | Logit: {val.item():.4f}")
        
    # 5. Check Specific Tokens
    # Qwen 2.5 tokenizes " (A)" as multiple tokens. We usually care about the choice token itself.
    targets = [" (A)", " (B)", "A", "B", " Yes", " No"]
    print("\nSpecific Token Scores:")
    for t in targets:
        ids = tokenizer.encode(t, add_special_tokens=False)
        # If multi-token, we'll take the most salient one (often the alphabetical char)
        # but for simplicity, let's just check the logit of the first token in the sequence 
        # that isn't just whitespace/bracket. 
        # Easier: just check if ANY of the IDs have a high score.
        target_id = ids[0]
        if len(ids) > 1:
            # For " (A)", ids might be [space, bracket, A, bracket] or similar.
            # In Qwen 2.5-7B: " (A)" -> [320, 32, 8]
            # ID 32 is "A". Let's try to find an alphabetical character ID if possible.
            for tid in ids:
                token_text = tokenizer.decode([tid])
                if any(c.isalpha() for c in token_text):
                    target_id = tid
                    break
        
        score = logits[target_id].item()
        print(f"  '{t}' (ID {target_id}): {score:.4f}")

def gradient_sensitivity_test(model, tokenizer, vector, metadata, layer_idx, num_samples=10, epsilon=0.01):
    """
    Computes sensitivity using Finite Difference: [L(h + eps*v) - L(h)] / eps
    This is much more memory efficient than a backward pass.
    """
    print(f"\n--- Causal Sensitivity Test (Finite Difference) (Layer {layer_idx}) ---")
    
    torch.cuda.empty_cache()
    samples = metadata[:num_samples]
    v = torch.tensor(vector, dtype=torch.float16, device=model.device)
    v = v / v.norm() 
    
    sensitivities = []
    
    for item in tqdm(samples):
        if 'prompt' not in item or 'target_token' not in item: continue
        
        full_text = item['prompt'] + item['target_token']
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        labels = inputs.input_ids.clone()
        
        # 0-indexed position of the last prompt token
        p_ids = tokenizer.encode(item['prompt'], add_special_tokens=True)
        pos = len(p_ids) - 1 

        # 1. Baseline Forward Pass
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss_base = outputs.loss.item()

        # 2. Perturbed Forward Pass
        # Hook to add epsilon * v to the activation
        def intervention_hook(module, input, output):
            # output is a tuple (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                h = output[0]
                h[:, pos, :] += epsilon * v
                return (h,) + output[1:]
            else:
                output[:, pos, :] += epsilon * v
                return output

        target_layer = model.model.layers[layer_idx]
        handle = target_layer.register_forward_hook(intervention_hook)
        
        with torch.no_grad():
            outputs_perturbed = model(**inputs, labels=labels)
            loss_perturbed = outputs_perturbed.loss.item()
            
        handle.remove()

        # Sensitivity = (L_perturbed - L_base) / epsilon
        # Negative means adding v REDUCED loss (v helped the model predict the target)
        diff = (loss_perturbed - loss_base) / epsilon
        sensitivities.append(diff)
            
    avg_sens = np.mean(sensitivities)
    print(f"Average Causal Sensitivity: {avg_sens:.6f}")
    print("Interpretation:")
    print("  Negative: Adding vector REDUCES loss (Causal/Aligned with target answer)")
    print("  Positive: Adding vector INCREASES loss (Opposed to target answer)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--activations_path", type=str, required=True)
    parser.add_argument("--layer", type=int, default=30)
    args = parser.parse_args()
    
    # 1. Load Vector
    data, metadata = load_activations(args.activations_path)
    v_syc = get_sycophancy_vector(data, metadata, args.layer)
    print(f"Sycophancy Vector (Layer {args.layer}) Norm: {np.linalg.norm(v_syc):.4f}")
    
    # 2. Load Model
    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    # Drastically reduces memory usage during the backward pass
    model.gradient_checkpointing_enable() 
    model.eval()
    
    # 3. Logit Lens
    logit_projection_test(model, tokenizer, v_syc, args.layer)
    
    # 4. Gradient Sensitivity
    gradient_sensitivity_test(model, tokenizer, v_syc, metadata, args.layer)

if __name__ == "__main__":
    main()
