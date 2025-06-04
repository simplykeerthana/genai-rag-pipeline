#!/usr/bin/env python3
"""
Fix Phi-3.5 loading by downloading missing files
"""
import os
import requests
import json

MODEL_PATH = "/Users/keerthana/gen-ai-OWASP-rag/models/phi-3.5-mini"

# Files that Phi-3.5 needs for custom code
REQUIRED_FILES = [
    "configuration_phi3.py",
    "modeling_phi3.py"
]

print("FIXING PHI-3.5 OFFLINE LOADING")
print("=" * 60)

# Check what files we have
print("\n1. Current files in model directory:")
for file in os.listdir(MODEL_PATH):
    print(f"   - {file}")

# Check if we're missing the custom code files
missing_files = []
for file in REQUIRED_FILES:
    if not os.path.exists(os.path.join(MODEL_PATH, file)):
        missing_files.append(file)

if missing_files:
    print(f"\n2. Missing required files: {missing_files}")
    print("\n3. Downloading missing files from Hugging Face...")
    
    # Base URL for Phi-3.5-mini on Hugging Face
    base_url = "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/"
    
    for file in missing_files:
        try:
            print(f"   Downloading {file}...")
            response = requests.get(base_url + file)
            response.raise_for_status()
            
            with open(os.path.join(MODEL_PATH, file), 'w') as f:
                f.write(response.text)
            
            print(f"   ✅ Downloaded {file}")
        except Exception as e:
            print(f"   ❌ Failed to download {file}: {e}")
else:
    print("\n2. All required files are present!")

# Now test loading
print("\n4. Testing model loading...")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Temporarily disable offline mode for this test
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=False  # Allow loading custom code
    )
    print("   ✅ Tokenizer loaded")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=False,  # Allow loading custom code
        torch_dtype=torch.float16,
        device_map=None
    )
    print("   ✅ Model loaded")
    
    # Move to MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"   ✅ Model moved to {device}")
    
    # Test generation
    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n   Test response: {response}")
    print("\n✅ Model works correctly!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("FIX FOR LLM_ROUTER.PY")
print("=" * 60)
print("\nUpdate your llm_router.py to remove the forced offline mode:")
print("\n1. Comment out or remove these lines:")
print('   # os.environ["HF_HUB_OFFLINE"] = "1"')
print('   # os.environ["TRANSFORMERS_OFFLINE"] = "1"')
print("\n2. Change local_files_only=True to local_files_only=False")
print("\n3. Or use a simpler model like Phi-2 that doesn't need custom code")