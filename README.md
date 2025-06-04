# 🔍 RAG Knowledge Extraction System

A powerful web-based system that extracts knowledge from websites using Retrieval-Augmented Generation (RAG). Ask questions about any website and get accurate answers with source citations.

![RAG System](https://img.shields.io/badge/RAG-Knowledge%20Extraction-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![React](https://img.shields.io/badge/React-18.0+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 What Does This Do?

This system allows you to:
1. **Point it at any website** (e.g., documentation, blogs, news sites)
2. **Ask questions** about the content
3. **Get accurate answers** with citations showing exactly where the information came from

### Example Use Cases
- 📚 "What are the security risks mentioned on the OWASP site?"
- 🔧 "How do I implement authentication according to this documentation?"
- 📰 "What are the main points discussed in this article?"
- 💡 "Summarize the key features described on this product page"

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- 8GB RAM (minimum) for local models
- 16GB RAM (recommended) for better performance

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/gen-ai-OWASP-rag.git
cd gen-ai-OWASP-rag
```

### Step 2: Set Up the Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Download Models (For Local LLM)

If you want to use local models (no API key required), you need to download them first:

```bash
# Run the model download script
python download_models.py
```

This script will:
- Create a `models` directory
- Download TinyLlama (1.1B parameters, ~670MB) for fast inference
- Optionally download Mistral 7B for better quality (requires more RAM)

**⚠️ Important**: If you skip this step, you can only use the Gemini API mode (requires API key).

### Step 4: Set Up the Frontend

```bash
cd ../frontend

# Install dependencies
npm install
```

### Step 5: Run the Application

**Terminal 1 - Start the Backend:**
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start the Frontend:**
```bash
cd frontend
npm start
```

### Step 6: Open Your Browser
Navigate to `http://localhost:3000`

## 💡 How to Use

### Using Local Models (No API Key Required)

1. Make sure you've run `download_models.py`
2. In the UI, click "LLM Settings"
3. Select "Local LLM"
4. Choose your model:
   - **TinyLlama**: Fast, lightweight (2-5 seconds response time)
   - **Mistral 7B**: Better quality (10-30 seconds response time)
5. Enter a website URL and your question
6. Click "Submit Query"

### Using Gemini API (Faster, Requires API Key)

1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. In the UI, click "LLM Settings"
3. Select "Gemini API"
4. Paste your API key
5. Enter a website URL and your question
6. Click "Submit Query"

## 📋 Example Queries

Try these examples to see the system in action:

### Example 1: Security Analysis
- **URL**: `https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html`
- **Query**: "What are the best practices for password storage?"

### Example 2: Documentation Search
- **URL**: `https://react.dev/learn`
- **Query**: "How do I manage state in React?"

### Example 3: News Summary
- **URL**: `https://techcrunch.com`
- **Query**: "What are the latest AI announcements?"

## 🛠️ Configuration Options

### Basic Configuration

Edit `backend/app/config.py`:

```python
# Choose default LLM mode
LLM_MODE = "transformers"  # Options: "local", "transformers", "gemini"

# Crawling settings
MAX_PAGES = 30  # Maximum pages to crawl per website
MAX_DEPTH = 2   # How deep to follow links

# Model paths (set by download_models.py)
LOCAL_MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
TRANSFORMER_MODEL_PATH = "models/mistral-7b-instruct-v3"
```

### Advanced Settings

```python
# Embedding model for semantic search
EMBEDDING_MODEL = "bge-base"  # Options: "bge-small" (fast), "bge-large" (accurate)

# Cache settings
CACHE_MAX_AGE_HOURS = 24  # How long to cache website content

# Chunk settings for text splitting
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

## 🏗️ System Architecture

```
User Interface (React)
        ↓
    REST API (FastAPI)
        ↓
    RAG Pipeline
    ├── Web Crawler (BeautifulSoup)
    ├── Text Splitter (LangChain)
    ├── Embeddings (HuggingFace)
    ├── Vector Store (FAISS)
    └── LLM (Local or API)
```

## 🐛 Troubleshooting

### Common Issues

**"Model not found" Error**
- Run `python download_models.py` to download required models
- Check that the `models` directory exists in the backend folder

**"Out of Memory" Error**
- Use TinyLlama instead of Mistral 7B
- Close other applications to free up RAM
- Consider using Gemini API instead

**"Cannot connect to backend"**
- Ensure the backend is running on port 8000
- Check firewall settings
- Try `http://localhost:8000` in your browser

**Slow Response Times**
- First-time model loading takes 10-30 seconds
- Subsequent queries are faster (models stay in memory)
- Use Gemini API for fastest responses

### Debug Mode

Check backend logs in the terminal for detailed information:
```
INFO:     Starting web crawl...
INFO:     Found 15 pages to index
INFO:     Creating embeddings...
INFO:     Searching for relevant content...
INFO:     Generating answer...
```

## 📦 Project Structure

```
gen-ai-OWASP-rag/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI server
│   │   ├── rag_pipeline.py      # Core RAG logic
│   │   ├── llm_router.py        # LLM management
│   │   ├── vector_store.py      # Vector search
│   │   └── config.py            # Configuration
│   ├── models/                  # Downloaded models (created by script)
│   ├── requirements.txt         # Python dependencies
│   └── download_models.py       # Model download script
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # React UI
│   │   └── App.css             # Styles
│   └── package.json            # Node dependencies
└── README.md                   # This file
```

## 🔒 Privacy & Security

- **Local Mode**: All processing happens on your machine. No data is sent to external servers.
- **API Mode**: Queries and website content are sent to Google's Gemini API.
- **Caching**: Crawled content is cached locally to improve performance.
- **No Tracking**: The application doesn't collect or store any usage data.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Hugging Face](https://huggingface.co/) for models and embeddings
- [OWASP](https://owasp.org/) for security best practices

---

**Need Help?** Open an issue on GitHub or check the [Discussions](https://github.com/yourusername/gen-ai-OWASP-rag/discussions) page.
