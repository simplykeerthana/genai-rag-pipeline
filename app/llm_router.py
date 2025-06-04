# app/llm_router.py
from app.config import LLM_MODE, LOCAL_MODEL_PATH, OPENAI_API_KEY, TRANSFORMER_MODEL_PATH
import os
import sys
import json
import requests

# Global variables - models will be loaded only when needed
response_format = "text"
model = None
tokenizer = None
llm = None
device = None

# Remove the initialization code - we'll load models on demand
print(f"[LLM_ROUTER] Initialized with default mode: {LLM_MODE}")

def load_local_model():
    """Load local GGUF model on demand"""
    global llm
    
    if llm is not None:
        return True  # Already loaded
        
    try:
        print("[LLM_ROUTER] Loading local LLaMA model...")
        from llama_cpp import Llama
        llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=2048, n_threads=4)
        print("[LLM_ROUTER] Local LLaMA model loaded successfully")
        return True
    except Exception as e:
        print(f"[LLM_ROUTER] Error loading local model: {e}")
        llm = None
        return False

def load_transformer_model():
    """Load transformer model on demand"""
    global model, tokenizer, device
    
    if model is not None and tokenizer is not None:
        return True  # Already loaded
        
    try:
        print("[LLM_ROUTER] Loading transformer model...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Set up torch
        torch.set_num_threads(4)
        
        # Device selection with MPS support
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        print(f"[LLM_ROUTER] Using device: {device}")
        
        model_path = TRANSFORMER_MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        print(f"[LLM_ROUTER] Loading model from: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("[LLM_ROUTER] Tokenizer loaded")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            low_cpu_mem_usage=True,
        )
        
        # Move model to device
        print(f"[LLM_ROUTER] Moving model to {device}...")
        model = model.to(device)
        model.eval()
        
        # Verify device
        first_param = next(model.parameters())
        actual_device = first_param.device
        print(f"[LLM_ROUTER] Model loaded successfully on: {actual_device}")
        
        return True
        
    except Exception as e:
        print(f"[LLM_ROUTER] Error loading transformer model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        tokenizer = None
        return False

def generate_response(prompt: str, max_length: int = 500, mode: str = None, api_key: str = None) -> str:
    """Generate response using configured LLM"""
    
    current_mode = mode or LLM_MODE
    
    print(f"[LLM_ROUTER] Generating response using mode: {current_mode}")
    
    try:
        if current_mode == "local":
            # Load model on demand
            if not load_local_model():
                return "Error: Failed to load local LLaMA model"
                
            output = llm(prompt, max_tokens=max_length, temperature=0.7)
            return output["choices"][0]["text"].strip()
            
        elif current_mode == "openai":
            import openai
            if api_key:
                openai.api_key = api_key
            elif OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-or-gemini-api-key-here":
                openai.api_key = OPENAI_API_KEY
            else:
                return "Error: OpenAI API key not configured"
                
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=0.7
            )
            return response["choices"][0]["message"]["content"].strip()
            
        elif current_mode == "gemini":
            # Use provided API key or fallback to config
            gemini_key = api_key
            if not gemini_key:
                # Try to get from environment or config
                gemini_key = os.getenv("GEMINI_API_KEY", OPENAI_API_KEY)
            
            if not gemini_key or gemini_key == "your-openai-or-gemini-api-key-here":
                return "Error: Gemini API key not configured"
            
            print(f"[LLM_ROUTER] Using Gemini API (no local model loading needed)")
            
            # Gemini API endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={gemini_key}"
            
            # Prepare request
            headers = {'Content-Type': 'application/json'}
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_length,
                    "temperature": 0.7,
                }
            }
            
            # Make request
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                # Extract text from Gemini response
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            return parts[0]['text'].strip()
                
                return "Error: Unexpected Gemini response format"
            else:
                error_msg = f"Gemini API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error'].get('message', '')}"
                except:
                    error_msg += f" - {response.text}"
                return error_msg
                
        elif current_mode == "transformers":
            # Load model on demand
            if not load_transformer_model():
                return "Error: Failed to load transformer model"
                
            import torch
            
            # Get device from model
            device = next(model.parameters()).device
            
            # Tokenize
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=False
            )
            
            # Move inputs to same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    min_new_tokens=10,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Clean up
            sentences = response.split('. ')
            if sentences and not response.endswith('.'):
                response = '. '.join(sentences[:-1]) + '.'
            
            return response
            
        else:
            return f"Error: Unknown LLM mode '{current_mode}'"
            
    except Exception as e:
        print(f"[LLM_ROUTER] Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

def unload_models():
    """Unload models to free memory (optional utility function)"""
    global model, tokenizer, llm
    
    print("[LLM_ROUTER] Unloading models...")
    
    if model is not None:
        del model
        model = None
        
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
        
    if llm is not None:
        del llm
        llm = None
        
    # Try to free GPU memory if using CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
        
    print("[LLM_ROUTER] Models unloaded")