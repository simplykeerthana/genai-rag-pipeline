import os

# Use relative paths that work across different environments
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Dynamic model detection - these are fallbacks if auto-detection fails
TRANSFORMER_MODEL_PATH = os.path.join(MODELS_DIR, "mistralai", "Mistral-7B-Instruct-v0.2")
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "llama", "model.gguf")  # Adjust as needed

# Mode: 'local', 'openai', 'gemini', or 'transformers'
LLM_MODE = "transformers"

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-or-gemini-api-key-here")

# Optional: Debug prints
if __name__ == "__main__":
    print(f"Base directory: {BASE_DIR}")
    print(f"Transformer model path: {TRANSFORMER_MODEL_PATH}")
    print(f"Transformer model exists: {os.path.exists(TRANSFORMER_MODEL_PATH)}")
    print(f"Local model path: {LOCAL_MODEL_PATH}")
    print(f"Local model exists: {os.path.exists(LOCAL_MODEL_PATH)}")