import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import sys



class SycophancyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Extract activations for sycophancy analysis using CAA method.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B", help="HuggingFace model name or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the factorial dataset jsonl")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the activations npz file")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Specific layers to extract (if None, all layers)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for extraction") 
    parser.add_argument("--chunk_size", type=int, default=2000, help="Save to disk every N samples")
    return parser.parse_args()

def load_dataset(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Robust Padding Token Fix
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Set pad_token to eos_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        else:
            # Fallback for models without EOS (rare)
            tokenizer.add_special_tokens({'pad_token': '<|extra_0|>'})
            print("Added new special token for padding.")
            
    # Qwen/Llama usually handle right padding correctly for activation extraction
    # (Attributes are aligned to the left)
    tokenizer.padding_side = "right" 
 
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map="cuda:0", # Force to first GPU to avoid slow offloading if possible
        torch_dtype=torch.bfloat16 # Qwen 2.5 is native BF16
    )
    model.eval()
    
    raw_data = load_dataset(args.dataset_path)
    if args.max_samples:
        raw_data = raw_data[:args.max_samples]
        
    print(f"Loaded {len(raw_data)} samples.")
    
    dataset = SycophancyDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Determine layers
    num_layers = model.config.num_hidden_layers
    if args.layers:
        layers_to_extract = args.layers
    else:
        layers_to_extract = list(range(num_layers))
        
    print(f"Extracting layers: {layers_to_extract} with Batch Size: {args.batch_size}")
    
    # Storage
    chunk_idx = 0
    current_chunk_activations = {l: [] for l in layers_to_extract}
    current_chunk_metadata = []
    samples_in_chunk = 0

    print(f"Starting extraction...")

    for batch in tqdm(dataloader):
        # Unpack batch - trivial collation by DataLoader gives list of fields
        batch_prompts = batch['prompt']
        batch_completions = batch['target_token']
        
        batch_full_texts = [p + c for p, c in zip(batch_prompts, batch_completions)]
        
        try:
            # 1. Tokenize full texts with padding
            inputs = tokenizer(
                batch_full_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            ).to(model.device)
            
            # 2. Tokenize prompts independently to find indices
            # No padding needed here, we just want lengths
            # We must use the same add_special_tokens setting as the full text (True usually adds BOS)
            # CAUTION: encode one by one to get accurate lengths without padding interference?
            # Or map encode.
            prompt_lens = [len(tokenizer.encode(p, add_special_tokens=True)) for p in batch_prompts]
            
            # 3. Calculate target indices
            # Qwen Tokenization Fix: " (A)" -> [" (", "A", ")"].
            # We want the "A". The length of prompt tokens points to the START of completion.
            # So index = len(prompt_ids) points to " (".
            # index = len(prompt_ids) + 1 points to "A".
            target_indices = [l + 1 for l in prompt_lens]
            
            # 4. Forward Pass
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # 5. Gather activations - Optimized (Reduce GPU-CPU syncing)
            indices = torch.tensor(target_indices, device=model.device)
            for layer in layers_to_extract:
                # Shape: [Batch, Seq, Hidden]
                hidden = outputs.hidden_states[layer + 1] 
                
                # Advanced indexing to get all batch vectors at once
                # Indices must be clamped to avoid sequence length errors
                clamped_indices = torch.clamp(indices, max=hidden.shape[1] - 1)
                batch_vecs = hidden[torch.arange(len(batch_prompts)), clamped_indices, :]
                
                # Move entire batch for this layer to CPU in one go
                batch_vecs_cpu = batch_vecs.to(torch.float32).cpu().numpy()
                current_chunk_activations[layer].extend(list(batch_vecs_cpu))

            # 6. Store Metadata
            # batch is a dict of lists, we iterate up to batch size
            for i in range(len(batch_prompts)):
                meta = {
                    "question_id": batch['question_id'][i] if isinstance(batch['question_id'], list) else batch['question_id'][i].item(),
                    "condition": batch['condition'][i],
                    "persona": batch['persona'][i],
                    "agreement": batch['agreement'][i],
                    "target_token": batch['target_token'][i]
                }
                current_chunk_metadata.append(meta)
                samples_in_chunk += 1

            # 7. Check Chunk Limit
            if samples_in_chunk >= args.chunk_size:
                print(f"Saving chunk {chunk_idx} (Size: {samples_in_chunk})...")
                chunk_path = f"{args.output_path.replace('.npz', '')}_chunk_{chunk_idx}.npz"
                
                save_dict = {
                    "metadata": current_chunk_metadata,
                    "layers": layers_to_extract
                }
                for layer in layers_to_extract:
                    save_dict[f"layer_{layer}"] = np.vstack(current_chunk_activations[layer])
                
                np.savez_compressed(chunk_path, **save_dict)
                print(f"Saved {chunk_path}")
                
                # Reset
                current_chunk_activations = {l: [] for l in layers_to_extract}
                current_chunk_metadata = []
                samples_in_chunk = 0
                chunk_idx += 1
                
                # Force cleanup
                del inputs, outputs
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in batch: {e}")
            continue

    # Maximum Final Save
    if samples_in_chunk > 0:
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

    print("Extraction complete.")

if __name__ == "__main__":
    main()
