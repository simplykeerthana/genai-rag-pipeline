import os

# Get the current file's directory (app directory)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the backend directory
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
# Get the project root directory (gen-ai-OWASP-rag)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Mode: 'local', 'openai', 'gemini', or 'transformers'
LLM_MODE = "transformers"

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-or-gemini-api-key-here")

# Local LLM (GGUF for llama.cpp)
LOCAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf")

# Hugging Face Transformers model (like Phi-4)
# The model is likely in the models folder at project root
TRANSFORMER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mistral-7b-instruct-v3")

# Optional: Debug prints
if __name__ == "__main__":
    print(f"Base directory: {BASE_DIR}")
    print(f"Transformer model path: {TRANSFORMER_MODEL_PATH}")
    print(f"Transformer model exists: {os.path.exists(TRANSFORMER_MODEL_PATH)}")
    print(f"Local model path: {LOCAL_MODEL_PATH}")
    print(f"Local model exists: {os.path.exists(LOCAL_MODEL_PATH)}")