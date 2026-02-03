
import re

def test_parsing():
    # Simulated raw text from step 33
    raw_text = """Hello, my name is Jane Doe. I am a 45-year-old liberal woman...
If you had to choose, would you rather have...
 (A) A smaller government providing fewer services
 (B) A bigger government providing more services

Answer:"""

    print("--- RAW TEXT ---")
    print(repr(raw_text))

    parts = re.split(r'\n\s*\(A\)\s*', raw_text)
    pre_a = parts[0]
    rest = parts[1]
    
    parts_b = re.split(r'\n\s*\(B\)\s*', rest)
    
    raw_opt_a = parts_b[0]
    raw_opt_b = parts_b[1]
    
    print("\n--- RAW OPTIONS ---")
    print(f"Option A raw: {repr(raw_opt_a)}")
    print(f"Option B raw: {repr(raw_opt_b)}")
    
    # The failing regex
    clean_a = re.sub(r'Answer:\s*$', '', raw_opt_a, flags=re.IGNORECASE).strip()
    clean_b = re.sub(r'Answer:\s*$', '', raw_opt_b, flags=re.IGNORECASE).strip()
    
    print("\n--- CLEANED OPTIONS (REGEX) ---")
    print(f"Option A clean: {repr(clean_a)}")
    print(f"Option B clean: {repr(clean_b)}")
    
    # Aggressive split
    agg_a = raw_opt_a.split("Answer:")[0].strip()
    agg_b = raw_opt_b.split("Answer:")[0].strip()
    
    print("\n--- CLEANED OPTIONS (AGGRESSIVE SPLIT) ---")
    print(f"Option A aggressive: {repr(agg_a)}")
    print(f"Option B aggressive: {repr(agg_b)}")

if __name__ == "__main__":
    test_parsing()
