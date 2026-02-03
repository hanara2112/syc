
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
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (default None for all)")
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
    
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
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
    metadata = []
    
    # Initialize separate storage files per layer to append effectively?
    # Appending to .npz is hard. Better to save chunks or usage a memory mapped array.
    # Approach: Save chunks every N samples. Merge later? 
    # Or just keep list but clear more aggressively?
    # 28k samples * 32 layers * 4096 dim * 4 bytes = ~14 GB. That's on the edge for 16GB RAM.
    # Better to save chunks.
    
    CHUNK_SIZE = 2000
    chunk_idx = 0
    current_chunk_activations = {l: [] for l in layers_to_extract}
    current_chunk_metadata = []

    print(f"Starting extraction with chunk size {CHUNK_SIZE}...")

    for i, item in enumerate(tqdm(dataset)):
        prompt = item['prompt']
        completion = item['target_token']
        full_text = prompt + completion
        
        try:
            inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True) 
            answer_pos = len(prompt_ids) + 1
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
            for layer in layers_to_extract:
                hidden = outputs.hidden_states[layer + 1]
                pos = min(answer_pos, hidden.shape[1] - 1)
                act = hidden[0, pos, :].float().cpu().numpy()
                current_chunk_activations[layer].append(act)
                
            # Store metadata
            meta = {
                "question_id": item.get("question_id"),
                "condition": item.get("condition"),
                "persona": item.get("persona"),
                "agreement": item.get("agreement"), 
                "target_token": item.get("target_token")
            }
            current_chunk_metadata.append(meta)

            # Cleanup
            del outputs, inputs, hidden
            if i % 100 == 0:
                torch.cuda.empty_cache()

            # Save Chunk
            if (i + 1) % CHUNK_SIZE == 0:
                print(f"Saving chunk {chunk_idx}...")
                chunk_path = f"{args.output_path.replace('.npz', '')}_chunk_{chunk_idx}.npz"
                
                save_dict = {
                    "metadata": current_chunk_metadata,
                    "layers": layers_to_extract
                }
                for layer in layers_to_extract:
                    save_dict[f"layer_{layer}"] = np.vstack(current_chunk_activations[layer])
                
                np.savez(chunk_path, **save_dict)
                print(f"Saved {chunk_path}")
                
                # Reset buffers
                current_chunk_activations = {l: [] for l in layers_to_extract}
                current_chunk_metadata = []
                chunk_idx += 1
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Save remaining
    if current_chunk_metadata:
        print(f"Saving final chunk {chunk_idx}...")
        chunk_path = f"{args.output_path.replace('.npz', '')}_chunk_{chunk_idx}.npz"
        
        save_dict = {
            "metadata": current_chunk_metadata,
            "layers": layers_to_extract
        }
        for layer in layers_to_extract:
            save_dict[f"layer_{layer}"] = np.vstack(current_chunk_activations[layer])
        
        np.savez_compressed(chunk_path, **save_dict)
        print(f"Saved {chunk_path}")

    print("Extraction complete. Please merge chunks manually or load them sequentially in analysis.")

if __name__ == "__main__":
    main()
