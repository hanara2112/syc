
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_path", type=str, default="results/activations.npz")
    args = parser.parse_args()
    
    data = np.load(args.activations_path, allow_pickle=True)
    metadata = data['metadata']
    if metadata.shape == (): metadata = metadata.item()
    
    conditions = np.array([m['condition'] for m in metadata])
    
    # Check Layer 15 (mid layer)
    layer = 15
    if f"layer_{layer}" not in data:
        layer = data['layers'][0] # Fallback to first available
        
    acts = data[f"layer_{layer}"]
    print(f"--- Debugging Layer {layer} ---")
    
    # Get indices for first example of each condition
    idx_A = np.where(conditions == 'A')[0]
    idx_B = np.where(conditions == 'B')[0]
    idx_C = np.where(conditions == 'C')[0]
    idx_D = np.where(conditions == 'D')[0]
    
    if len(idx_A) == 0:
        print("Missing conditions!")
        return

    # Compare First A vs First D (Both are "Liberal Answer", but different Persona)
    # A = Lib Persona + Lib Answer
    # D = Con Persona + Lib Answer
    # If A == D, the model is ignoring the Persona.
    
    vec_A = acts[idx_A[0]]
    vec_D = acts[idx_D[0]]
    
    diff_context = np.linalg.norm(vec_A - vec_D)
    print(f"Distance between A (Lib Persona) and D (Con Persona) with same Answer: {diff_context:.8f}")
    
    if diff_context < 1e-6:
        print("CRITICAL: The model outputs are IDENTICAL despite different Personas. Context is being ignored or lost.")
    else:
        print("Context is modifying the vector (Good).")
        
    # Compare A vs B (Same Persona, Different Answer)
    vec_B = acts[idx_B[0]]
    diff_answer = np.linalg.norm(vec_A - vec_B)
    print(f"Distance between A (Answer A) and B (Answer B) with same Persona: {diff_answer:.8f}")
    
    # Check Statistics
    syco_vec = (acts[idx_A].mean(0) + acts[idx_C].mean(0)) / 2
    nonsyco_vec = (acts[idx_B].mean(0) + acts[idx_D].mean(0)) / 2
    
    total_signal = np.linalg.norm(syco_vec - nonsyco_vec)
    print(f"Total Sycophancy Signal (High Precision): {total_signal:.10f}")
    
    print("\nSample Vector A (first 5 dims):", vec_A[:5])
    print("Sample Vector B (first 5 dims):", vec_B[:5])

if __name__ == "__main__":
    main()
