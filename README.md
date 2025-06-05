
# 🔍 RAG Knowledge Extraction System

Extract knowledge from websites using Retrieval-Augmented Generation (RAG). Ask questions about any website and get accurate answers with source citations.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- 8GB RAM (minimum)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/gen-ai-OWASP-rag.git
cd gen-ai-OWASP-rag

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download models for local LLM
python download_models.py

# Frontend setup
cd ../frontend
npm install
```

### Run the Application

**Terminal 1:**
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2:**
```bash
cd frontend
npm start
```

Open `http://localhost:3000` in your browser.

## 📸 Screenshots



## 📋 API Usage

### Using Local LLM (No API Key Required)

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main security risks in generative AI?",
    "url": "https://genai.owasp.org"
  }'
```

![image](https://github.com/user-attachments/assets/35ed6429-8297-4d9f-af57-07ba4fe422da)

This run on local llm on mps gpu. 

![image](https://github.com/user-attachments/assets/a6a64c3b-b68c-46c8-a680-5348b0c4f2d7)


Here is UI version of running it on local LLM

![image](https://github.com/user-attachments/assets/08dd3d3b-a603-4ad0-be14-a886c5af67bb)

The results of it, including the chunks it was 



https://github.com/user-attachments/assets/a07538bc-87d7-4d7c-8892-ff45500a0d79






### Using Gemini API (API Key Required)

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main security risks in generative AI?",
    "url": "https://genai.owasp.org",
    "llm_mode": "gemini",
    "api_key": "AIzaSyD7oqlfEIVqA5MN94o36JhvjCDOWKJ4nt8"
  }'
```

![Pasted Graphic copy](https://github.com/user-attachments/assets/66e72124-c82d-4623-9a29-3e7ebc353349)

Here is the UI version of running it on 

<img width="1227" alt="image" src="https://github.com/user-attachments/assets/3ad01be3-96f0-4404-b50a-f106054a04e8" />

The results of it 



https://github.com/user-attachments/assets/107b96d5-caa1-471d-8e30-d0e0883ad61f




### Advanced Options

```bash
# Force refresh cache and limit crawling
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What authentication methods are recommended?",
    "url": "https://cheatsheetseries.owasp.org",
    "force_refresh": true,
    "max_pages": 10,
    "max_depth": 1,
    "top_k": 3
  }'
```

### Check Cache Status

```bash
# Get cache information
curl -X GET "http://localhost:8000/cache/info"

# Clear cache for specific URL
curl -X POST "http://localhost:8000/cache/clear?url=https://genai.owasp.org"

# Clear all cache
curl -X POST "http://localhost:8000/cache/clear"
```

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

## 📊 Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Your question about the website |
| `url` | string | required | Website URL to analyze |
| `api_key` | string | optional | Gemini API key (if provided, uses API mode) |
| `force_refresh` | boolean | false | Bypass cache and re-crawl |
| `max_pages` | integer | 30 | Maximum pages to crawl |
| `max_depth` | integer | 2 | How deep to follow links |
| `top_k` | integer | 5 | Number of relevant chunks to retrieve |
| `llm_mode` | string | "transformers" | LLM to use ("local" or "transformers") |
| `embedding_model` | string | "bge-base" | Embedding model for semantic search |

## 🔧 Configuration

Edit `backend/app/config.py`:

```python
# Default LLM mode (when no API key provided)
LLM_MODE = "transformers"  # or "local" for TinyLlama

# Model paths
LOCAL_MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
TRANSFORMER_MODEL_PATH = "models/mistral-7b-instruct-v3"
```

## 🏗️ How It Works

1. **Crawl** - Discovers and downloads pages from the target website
2. **Chunk** - Splits content into manageable pieces
3. **Embed** - Creates vector embeddings for semantic search
4. **Search** - Finds the most relevant chunks for your query
5. **Generate** - Uses LLM to create a comprehensive answer



## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create a feature branch**  
   ```bash
   git checkout -b feature/amazing-feature

	3.	Commit your changes

git commit -m 'Add amazing feature'


	4.	Push to the branch

git push origin feature/amazing-feature


	5.	Open a Pull Request on GitHub

Thank you for your contributions! 🙌

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
