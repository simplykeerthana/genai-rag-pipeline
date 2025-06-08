# app/llm_router.py
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from app.config import LLM_MODE, LOCAL_MODEL_PATH, OPENAI_API_KEY, TRANSFORMER_MODEL_PATH
import os
import sys
import json
import requests
import logging

logger = logging.getLogger(__name__)

# Try to import llama_cpp, but don't fail if it's not available
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
    print("[LLM_ROUTER] llama-cpp-python is available")
except ImportError:
    print("[LLM_ROUTER] llama-cpp-python not available - local GGUF models disabled")
    LLAMA_CPP_AVAILABLE = False
    Llama = None

# Global variables - models will be loaded only when needed
response_format = "text"
model = None
tokenizer = None
llm = None
device = None

# Helper functions from patch
def get_model_type(model_path):
    """Detect model type based on file extension or directory structure"""
    if os.path.isfile(model_path):
        if model_path.endswith('.gguf'):
            return 'gguf'
        elif model_path.endswith('.bin') or model_path.endswith('.safetensors'):
            return 'transformers'
    elif os.path.isdir(model_path):
        # Check for GGUF files in directory
        for file in os.listdir(model_path):
            if file.endswith('.gguf'):
                return 'gguf'
        # Check for transformers model files
        if any(file in os.listdir(model_path) for file in ['config.json', 'pytorch_model.bin', 'model.safetensors']):
            return 'transformers'
    return 'unknown'

def find_gguf_file(model_dir):
    """Find GGUF file in a directory"""
    if os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.gguf'):
                return os.path.join(model_dir, file)
    return None

print(f"[LLM_ROUTER] Initialized with default mode: {LLM_MODE}")

def load_local_model():
    """Load local GGUF model on demand"""
    global llm
    
    # Check if llama-cpp-python is available
    if not LLAMA_CPP_AVAILABLE:
        print("[LLM_ROUTER] Cannot load local model - llama-cpp-python not installed")
        return False
    
    if llm is not None:
        return True  # Already loaded
        
    try:
        print("[LLM_ROUTER] Loading local LLaMA model...")
        
        # Handle both file and directory paths
        model_path = LOCAL_MODEL_PATH
        if os.path.isdir(model_path):
            gguf_file = find_gguf_file(model_path)
            if gguf_file:
                model_path = gguf_file
                print(f"[LLM_ROUTER] Found GGUF file: {model_path}")
            else:
                raise FileNotFoundError(f"No GGUF file found in directory: {model_path}")
        
        llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)
        print("[LLM_ROUTER] Local LLaMA model loaded successfully")
        return True
    except Exception as e:
        print(f"[LLM_ROUTER] Error loading local model: {e}")
        llm = None
        return False

def load_transformer_model():
    """Load transformer model on demand"""
    global model, tokenizer, device, llm
    
    if model is not None and tokenizer is not None:
        return True  # Already loaded
    
    # Enhanced debugging
    print("[LLM_ROUTER] ========== DEBUG INFO ==========")
    print(f"[LLM_ROUTER] TRANSFORMER_MODEL_PATH: {TRANSFORMER_MODEL_PATH}")
    print(f"[LLM_ROUTER] Current working directory: {os.getcwd()}")
    print(f"[LLM_ROUTER] Python path: {sys.path[:3]}...")
    
    # Check /models directory
    if os.path.exists("/models"):
        print("[LLM_ROUTER] /models directory contents:")
        try:
            models_contents = os.listdir("/models")
            for item in models_contents[:5]:  # Show first 5 items
                item_path = os.path.join("/models", item)
                if os.path.isdir(item_path):
                    print(f"[LLM_ROUTER]   📁 {item}/")
                else:
                    print(f"[LLM_ROUTER]   📄 {item}")
        except Exception as e:
            print(f"[LLM_ROUTER]   Error listing /models: {e}")
    else:
        print("[LLM_ROUTER] ❌ /models directory does not exist!")
    
    # Check if the model path exists
    model_path = TRANSFORMER_MODEL_PATH
    print(f"[LLM_ROUTER] Checking model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[LLM_ROUTER] ❌ Model path does not exist: {model_path}")
        
        # Additional debugging
        parent_dir = os.path.dirname(model_path)
        print(f"[LLM_ROUTER] Parent directory: {parent_dir}")
        print(f"[LLM_ROUTER] Parent exists: {os.path.exists(parent_dir)}")
        
        if os.path.exists(parent_dir):
            print(f"[LLM_ROUTER] Contents of {parent_dir}:")
            try:
                for item in os.listdir(parent_dir)[:10]:
                    print(f"[LLM_ROUTER]   - {item}")
            except Exception as e:
                print(f"[LLM_ROUTER]   Error listing parent: {e}")
        
        # Check for alternative paths
        alt_paths = [
            "/models/mistral-7b-instruct-v3",
            "/app/models/mistral-7b-instruct-v3",
            "./models/mistral-7b-instruct-v3",
            "../models/mistral-7b-instruct-v3"
        ]
        
        print("[LLM_ROUTER] Checking alternative paths:")
        for alt_path in alt_paths:
            exists = os.path.exists(alt_path)
            print(f"[LLM_ROUTER]   {alt_path}: {'✅' if exists else '❌'}")
            if exists:
                print(f"[LLM_ROUTER]   Found model at: {alt_path}")
                model_path = alt_path
                break
        else:
            return False
    
    # If we found the model path, check its contents
    if os.path.exists(model_path):
        print(f"[LLM_ROUTER] ✅ Model path exists: {model_path}")
        print("[LLM_ROUTER] Model directory contents:")
        try:
            files = os.listdir(model_path)
            config_files = [f for f in files if f.endswith('.json')]
            model_files = [f for f in files if f.endswith(('.safetensors', '.bin'))]
            
            print(f"[LLM_ROUTER]   Config files ({len(config_files)}): {config_files[:3]}")
            print(f"[LLM_ROUTER]   Model files ({len(model_files)}): {model_files[:3]}")
            
            # Check for required files
            has_config = 'config.json' in files
            has_tokenizer = any('tokenizer' in f for f in files)
            has_model = len(model_files) > 0
            
            print(f"[LLM_ROUTER]   Has config.json: {'✅' if has_config else '❌'}")
            print(f"[LLM_ROUTER]   Has tokenizer files: {'✅' if has_tokenizer else '❌'}")
            print(f"[LLM_ROUTER]   Has model files: {'✅' if has_model else '❌'}")
            
        except Exception as e:
            print(f"[LLM_ROUTER]   Error checking contents: {e}")
    
    print("[LLM_ROUTER] ================================")
    
    # Continue with the rest of the loading code...
    try:
        print("[LLM_ROUTER] Loading transformer model...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Set up torch
        torch.set_num_threads(4)
        
        # Device selection with MPS support
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"[LLM_ROUTER] Using device: {device}")
        
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
            # Check if llama-cpp-python is available
            if not LLAMA_CPP_AVAILABLE:
                print("[LLM_ROUTER] Local mode requested but llama-cpp-python not available")
                return "Error: Local model not available - llama-cpp-python not installed. Please use 'transformers', 'gemini', or 'openai' mode instead."
            
            # Load model on demand
            if not load_local_model():
                return "Error: Failed to load local LLaMA model"
                
            output = llm(prompt, max_tokens=max_length, temperature=0.7)
            return output["choices"][0]["text"].strip()
            
        elif current_mode == "openai":
            try:
                import openai
            except ImportError:
                return "Error: OpenAI package not installed. Please install with: pip install openai"
                
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
            
            # Check if we actually loaded a GGUF model instead
            if llm is not None and model is None:
                # Use GGUF model
                output = llm(prompt, max_tokens=max_length, temperature=0.7)
                return output["choices"][0]["text"].strip()
                
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
                try:
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
                except NotImplementedError as e:
                    if device == "mps":
                        print("[LLM_ROUTER] NotImplementedError on MPS, retrying on CPU...")
                        device = "cpu"
                        model.to(device)
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
                    else:
                        raise
            
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

def get_available_modes():
    """Return list of available LLM modes based on what's installed"""
    available = []
    
    # Check for local mode
    if LLAMA_CPP_AVAILABLE:
        if os.path.exists(LOCAL_MODEL_PATH):
            available.append("local")
        # Also check if transformer path has GGUF
        elif os.path.exists(os.path.dirname(TRANSFORMER_MODEL_PATH)):
            if get_model_type(os.path.dirname(TRANSFORMER_MODEL_PATH)) == 'gguf':
                available.append("local")
    
    # Check for transformers mode
    try:
        import transformers
        import torch
        if os.path.exists(TRANSFORMER_MODEL_PATH):
            available.append("transformers")
        # Check parent directory for any model files
        elif os.path.exists(os.path.dirname(TRANSFORMER_MODEL_PATH)):
            model_type = get_model_type(os.path.dirname(TRANSFORMER_MODEL_PATH))
            if model_type == 'transformers':
                available.append("transformers")
            elif model_type == 'gguf' and LLAMA_CPP_AVAILABLE:
                available.append("transformers")  # Can use GGUF in transformers mode
    except ImportError:
        pass
    
    # API modes are always available if keys are provided
    available.extend(["gemini", "openai"])
    
    return available

def get_mode_info():
    """Get information about available modes"""
    # Check what's actually in the model directory
    model_dir = os.path.dirname(TRANSFORMER_MODEL_PATH)
    detected_type = "unknown"
    available_files = []
    
    if os.path.exists(model_dir):
        available_files = os.listdir(model_dir)
        detected_type = get_model_type(model_dir)
    
    info = {
        "current_mode": LLM_MODE,
        "available_modes": get_available_modes(),
        "model_directory": model_dir,
        "detected_model_type": detected_type,
        "available_files": available_files,
        "mode_details": {
            "local": {
                "available": LLAMA_CPP_AVAILABLE,
                "model_path": LOCAL_MODEL_PATH if LLAMA_CPP_AVAILABLE else None,
                "status": "Available" if LLAMA_CPP_AVAILABLE and (os.path.exists(LOCAL_MODEL_PATH) or detected_type == 'gguf') else "Not available (llama-cpp-python not installed)" if not LLAMA_CPP_AVAILABLE else "Model file not found"
            },
            "transformers": {
                "available": os.path.exists(TRANSFORMER_MODEL_PATH) or detected_type in ['transformers', 'gguf'],
                "model_path": TRANSFORMER_MODEL_PATH,
                "status": "Available" if os.path.exists(TRANSFORMER_MODEL_PATH) else f"Model not found at {TRANSFORMER_MODEL_PATH}, detected {detected_type} in parent directory" if detected_type != 'unknown' else "Model not found"
            },
            "gemini": {
                "available": True,
                "status": "Available (requires API key)"
            },
            "openai": {
                "available": True,
                "status": "Available (requires API key)"
            }
        }
    }
    return info