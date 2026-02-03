
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-7B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    prompt = "Question: Test?"
    completion_A = " (A)"
    completion_B = " (B)"
    
    # 1. Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Prompt tokens: {prompt_ids}")
    
    # 2. Tokenize full
    full_A = tokenizer.encode(prompt + completion_A, add_special_tokens=False)
    full_B = tokenizer.encode(prompt + completion_B, add_special_tokens=False)
    
    print(f"Full A tokens: {full_A}")
    print(f"Full B tokens: {full_B}")
    
    # 3. Completion tokens only
    comp_A_ids = full_A[len(prompt_ids):]
    comp_B_ids = full_B[len(prompt_ids):]
    
    print(f"Completion A part: {comp_A_ids}")
    print(f"Completion B part: {comp_B_ids}")
    
    # Decode to see what they are
    print(f"Decoded A part: {[tokenizer.decode([t]) for t in comp_A_ids]}")
    print(f"Decoded B part: {[tokenizer.decode([t]) for t in comp_B_ids]}")
    
    if comp_A_ids[0] == comp_B_ids[0]:
        print("ALERT: The first token of the completion is IDENTICAL!")
        print("You must extract at index +1.")
    else:
        print("First completion tokens are different. Good.")

except Exception as e:
    print(f"Error: {e}")
