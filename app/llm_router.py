# app/llm_router.py
from app.config import LLM_MODE, LOCAL_MODEL_PATH, OPENAI_API_KEY, TRANSFORMER_MODEL_PATH
import os
import sys
import json
import requests

# Global variables
response_format = "text"
model = None
tokenizer = None
llm = None

# Initialize LLM based on mode
print(f"[LLM_ROUTER] Initializing with mode: {LLM_MODE}")

if LLM_MODE == "local":
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=2048, n_threads=4)
        print("[LLM_ROUTER] Local LLaMA model loaded successfully")
    except Exception as e:
        print(f"[LLM_ROUTER] Error loading local model: {e}")
        llm = None
    
elif LLM_MODE == "openai":
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        print("[LLM_ROUTER] OpenAI configured")
    except Exception as e:
        print(f"[LLM_ROUTER] Error configuring OpenAI: {e}")
        
elif LLM_MODE == "gemini":
    # Gemini doesn't need initialization, just API key
    print("[LLM_ROUTER] Gemini API configured")
    
elif LLM_MODE == "transformers":
    try:
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
        
    except Exception as e:
        print(f"[LLM_ROUTER] Error loading transformer model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        tokenizer = None

def generate_response(prompt: str, max_length: int = 500, mode: str = None, api_key: str = None) -> str:
    """Generate response using configured LLM"""
    
    current_mode = mode or LLM_MODE
    
    try:
        if current_mode == "local" and llm:
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
                
        elif current_mode == "transformers" and model and tokenizer:
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
            return f"Error: LLM not properly initialized for mode {current_mode}"
            
    except Exception as e:
        print(f"[LLM_ROUTER] Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"