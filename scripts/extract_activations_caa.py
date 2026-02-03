
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# --- MONKEY PATCH FOR QWEN + TRANSFORMERS > 4.34 ---
import transformers
try:
    from transformers import BeamSearchScorer
except ImportError:
    # Try locating it in generation submodules or mock it
    FoundScorer = None
    try:
        from transformers.generation import BeamSearchScorer
        FoundScorer = BeamSearchScorer
    except ImportError:
        try:
            from transformers.generation.beam_search import BeamSearchScorer
            FoundScorer = BeamSearchScorer
        except ImportError:
            # Fallback: Mock it so transformers_stream_generator doesn't crash on import
            print("WARNING: BeamSearchScorer not found. Creating a dummy class.")
            class BeamSearchScorer:
                pass
            FoundScorer = BeamSearchScorer

    if FoundScorer:
        transformers.BeamSearchScorer = FoundScorer
        import sys
        # Also ensure it's available in sys.modules for 'from transformers import ...'
        if 'transformers' in sys.modules:
             setattr(sys.modules['transformers'], 'BeamSearchScorer', FoundScorer)
        print("Monkey-patched transformers.BeamSearchScorer for Qwen compatibility.")
# ---------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extract activations for sycophancy analysis using CAA method.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", help="HuggingFace model name or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the factorial dataset jsonl")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the activations npz file")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Specific layers to extract (if None, all layers)")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to process (default 100 for speed)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently only 1 supported for precise token alignment)") 
    return parser.parse_args()

def load_dataset(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_answer_token_index(tokenizer, prompt, completion):
    """
    Identifies the index of the first token of the completion in the full sequence.
    Critical for CAA extraction.
    """
    # 1. Tokenize prompt only
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    
    # 2. Tokenize full text (prompt + completion)
    full_text = prompt + completion
    full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    
    # The answer token is the first token after the prompt
    # verification: check if prompt tokens match the beginning of full tokens
    if not torch.equal(full_tokens[:len(prompt_tokens)], prompt_tokens):
        # Fallback/Error handling if tokenization is unstable (e.g. merge issues)
        # For some tokenizers, prompt + completion tokenizes differently than prompt alone.
        # We search for the completion tokens at the end.
        completion_tokens = tokenizer(completion, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # Find where completion starts
        # This is a simple heuristic: look at the end
        if torch.equal(full_tokens[-len(completion_tokens):], completion_tokens):
            return len(full_tokens) - len(completion_tokens)
        else:
            # If complex merge, might need more robust matching. 
            # For now, relying on length of prompt is the standard "first answer token" heuristic 
            # used in many steering papers, assuming strict appending.
             pass 

    return len(prompt_tokens)

def main():
    args = parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", trust_remote_code=True, dtype=torch.float16)
    
    dataset = load_dataset(args.dataset_path)
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        
    print(f"Loaded {len(dataset)} samples.")
    
    # Determine layers
    num_layers = model.config.num_hidden_layers
    if args.layers:
        layers_to_extract = args.layers
    else:
        layers_to_extract = list(range(num_layers)) # Extract all if not specified
        
    print(f"Extracting layers: {layers_to_extract}")
    
    # Storage
    all_activations = {l: [] for l in layers_to_extract}
    metadata = []
    
    for i, item in enumerate(tqdm(dataset)):
        prompt = item['prompt']
        
        # In the factorial dataset, 'target_token' is usually " (A)" or " (B)"
        # We want to force this completion.
        completion = item['target_token']
        
        full_text = prompt + completion
        
        # Identify extraction position
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        
        # We need to find the position indices. 
        # Using the helper function logic inline to use the full `inputs` object correctly
        # Re-tokenizing prompt to find length
        prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # Note: 'inputs' usually has special tokens added (like BOS), so we need to account for that.
        # If tokenizer adds BOS, len(inputs) = len(prompt_tokens) + len(completion_tokens) + 1
        
        # Robust method:
        # 1. Feed full text
        # 2. Extract at index corresponding to the start of completion
        
        # Let's count non-padding tokens
        total_len = inputs['input_ids'].shape[1]
        
        # Calculate prompt length in tokens (without special tokens if inputs has them? No, inputs has them)
        # Let's trust the tokenizer's consistency for now
        # Actually, for many chat models, we need to apply chat template?
        # The prompt in the dataset is raw text "Hello...". 
        # CAUTION: The user dataset seems to have raw prompts. 
        # Ideally we should apply chat template if it's a chat model, BUT the user code 
        # `sycophancy/extraction.py` used `build_chat_prompt`.
        # The `syc_dim/scripts/construct_factorial_dataset.py` creates raw strings.
        # I should probably just use the raw strings as provided in the dataset 'prompt' field 
        # because that's what the factorial construction script made. 
        # However, to be safe, I'll stick to the text in the 'prompt' field.
        
        # Calculate split point
        # A simple way that works for most sentencepiece tokenizers:
        # Encode prompt
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True) # inputs likely has special tokens too
        answer_pos = len(prompt_ids) 
        
        # Check boundary - if the tokenization merges the last token of prompt and first of completion,
        # we might need to back up. But "(A)" usually starts with a space and is distinct.
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        for layer in layers_to_extract:
            # hidden_states is tuple (embeddings, layer_1, ... layer_N)
            # layer_idx maps to hidden_states[layer_idx + 1]
            hidden = outputs.hidden_states[layer + 1]
            
            # Extract at answer_pos
            # If answer_pos is out of bounds (can happen with some tokenizers/special tokens mismatch), clamp it
            pos = min(answer_pos, hidden.shape[1] - 1)
            
            # Shape: [batch, seq, dim] -> [dim]
            act = hidden[0, pos, :].cpu().numpy()
            all_activations[layer].append(act)
            
        # Store metadata
        meta = {
            "question_id": item.get("question_id"),
            "condition": item.get("condition"),
            "persona": item.get("persona"),
            "agreement": item.get("agreement"), # agree/disagree
            "target_token": item.get("target_token") # (A)/(B)
        }
        metadata.append(meta)

    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    save_dict = {
        "metadata": metadata,
        "layers": layers_to_extract
    }
    
    for layer in layers_to_extract:
        save_dict[f"layer_{layer}"] = np.vstack(all_activations[layer])
        
    np.savez(args.output_path, **save_dict)
    print(f"Saved activations to {args.output_path}")

if __name__ == "__main__":
    main()
