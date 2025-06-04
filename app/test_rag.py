#!/usr/bin/env python3
"""
Fix script for Mistral-7B in RAG pipeline
"""
import subprocess
import sys

print("FIXING MISTRAL-7B SETUP")
print("=" * 60)

# 1. Install missing dependencies
print("\n1. Installing missing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf", "sentencepiece"])
print("✅ Dependencies installed")

# 2. Test the setup
print("\n2. Testing Mistral-7B loading...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    model_path = "/Users/keerthana/gen-ai-OWASP-rag/models/mistral-7b-instruct-v3"
    
    # Load tokenizer with proper settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True  # Use fast tokenizer
    )
    print("✅ Tokenizer loaded")
    
    # Test model loading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print("✅ Model loaded")
    
    print("\n3. Your RAG pipeline is ready to use Mistral-7B!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTry removing the forced offline mode in llm_router.py:")
    print('Comment out: os.environ["HF_HUB_OFFLINE"] = "1"')

print("\n" + "=" * 60)
print("HOW YOUR RAG PIPELINE WORKS WITH MISTRAL")
print("=" * 60)

print("""
1. User asks: "What is OWASP?"
2. RAG pipeline:
   - Crawls the website
   - Finds relevant chunks about OWASP
   - Creates context from those chunks
3. Sends to Mistral-7B:
   "Based on this context: [relevant chunks]
    Answer: What is OWASP?"
4. Mistral generates answer based on the context

Your rag_pipeline.py handles all of this automatically!
""")