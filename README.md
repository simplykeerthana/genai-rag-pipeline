# GenAI RAG Pipeline

A complete **Retrieval-Augmented Generation (RAG)** pipeline that scrapes websites, creates vector embeddings, and answers questions using local or API-based LLMs. Built with FastAPI backend and React frontend.

## 🚀 Features

- **Web Scraping**: Automatically crawl websites using LangChain's web loaders
- **Vector Storage**: FAISS vector database for efficient similarity search
- **Multiple LLM Support**: Local Mistral, OpenAI GPT, and Google Gemini
- **Smart Caching**: Avoid re-processing the same URLs
- **Real-time Streaming**: Live updates during processing
- **Modern UI**: React frontend with real-time logs and source tracking
- **Flexible Embeddings**: Support for multiple embedding models

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Design Choices](#design-choices)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Client  │───▶│  FastAPI Server │───▶│ Vector Database │
│   (Frontend)    │    │   (Backend)     │    │    (FAISS)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐               │
         │              │  Web Scraper    │               │
         │              │  (LangChain)    │               │
         │              └─────────────────┘               │
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐               │
         │              │ Embedding Model │               │
         │              │ (BGE/Sentence)  │               │
         │              └─────────────────┘               │
         │                       │                       │
         │                       ▼                       │
         └──────────────▶┌─────────────────┐◀──────────────┘
                         │   LLM Router    │
                         │ Local/API LLMs  │
                         └─────────────────┘
```

### Data Flow

1. **URL Input**: User provides a website URL and question
2. **Web Scraping**: LangChain crawlers extract content from web pages
3. **Text Processing**: Content is chunked and cleaned
4. **Embedding Creation**: Text chunks converted to vector embeddings
5. **Vector Storage**: Embeddings stored in FAISS for fast retrieval
6. **Query Processing**: User question is embedded and matched against stored vectors
7. **Context Retrieval**: Most relevant chunks are retrieved
8. **Answer Generation**: LLM generates response using retrieved context
9. **Response Delivery**: Answer and sources returned to frontend

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- 8GB+ RAM (for local models)
- Git

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/genai-rag-pipeline.git
cd genai-rag-pipeline
```

### 2. Download Models (Automated)

```bash
# Download Mistral-7B-Instruct-v0.2 model
python download_models.py
```

### 3. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Frontend Setup

```bash
cd ../frontend
npm install
```

### 5. Start Services

**Terminal 1 (Backend):**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

### 6. Access Application

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📦 Installation

### Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Key Packages:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `langchain` - LLM orchestration
- `langchain-community` - Community integrations
- `faiss-cpu` - Vector similarity search
- `transformers` - Hugging Face models
- `torch` - PyTorch for model inference
- `sentence-transformers` - Embedding models
- `beautifulsoup4` - HTML parsing
- `requests` - HTTP client

### Frontend Dependencies

```bash
cd frontend
npm install
```

**Key Packages:**
- `react` - UI framework
- `vite` - Build tool
- `axios` - HTTP client
- `react-markdown` - Markdown rendering
- `lucide-react` - Icons

## 🤖 Model Setup

### Automated Download

The easiest way to set up models:

```bash
./scripts/download_models.sh
```

This script:
1. Creates `models/mistralai/` directory
2. Downloads Mistral-7B-Instruct-v0.2 from Hugging Face
3. Sets up proper directory structure
4. Validates model files

### Manual Download

If you prefer manual setup:

```bash
# Install git-lfs
git lfs install

# Create models directory
mkdir -p models/mistralai

# Clone Mistral model
cd models/mistralai
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
```

### Model Directory Structure

```
models/
└── mistralai/
    └── Mistral-7B-Instruct-v0.2/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── pytorch_model.bin (or .safetensors files)
        └── ...
```

### Alternative Models

You can use other models by updating the auto-detection logic or placing them in:
- `models/mistralai/[your-model-name]/`
- The system will auto-detect any folder containing "mistral" in the name

## 🎯 Usage

### Basic Usage

1. **Start the application** (both backend and frontend)
2. **Enter a website URL** (e.g., `https://docs.python.org`)
3. **Ask a question** (e.g., "What is Python used for?")
4. **Select LLM mode**:
   - **Transformers**: Local Mistral model
   - **Gemini**: Google's Gemini API
   - **OpenAI**: OpenAI's GPT models
5. **Click "Ask Question"**

### Advanced Settings

- **Max Pages**: Limit crawling depth (1-50)
- **Top K**: Number of relevant chunks to retrieve (1-10)
- **Force Rebuild**: Ignore cache and re-process URL
- **API Keys**: Required for Gemini/OpenAI modes

### API Usage

```python
import requests

response = requests.post("http://localhost:8000/ask", json={
    "query": "What is machine learning?",
    "url": "https://scikit-learn.org",
    "llm_mode": "transformers",
    "max_pages": 5,
    "top_k": 3
})

print(response.json()["answer"])
```

## 📁 Code Structure

```
genai-rag-pipeline/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI application entry point
│   │   ├── config.py          # Configuration settings
│   │   ├── llm_router.py      # LLM model management
│   │   ├── vector_store.py    # FAISS vector operations
│   │   ├── cached_loader.py   # Web scraping with caching
│   │   └── rag_pipeline.py    # Main RAG orchestration
│   ├── requirements.txt       # Python dependencies
│   └── venv/                  # Virtual environment
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── App.css           # Styling
│   │   ├── main.jsx          # React entry point
│   │   └── index.css         # Global styles
│   ├── package.json          # Node.js dependencies
│   ├── vite.config.js        # Vite configuration
│   └── index.html            # HTML template
├── models/                   # Model storage
│   └── mistralai/           # Mistral models
├── scripts/                 # Utility scripts
│   └── download_models.sh   # Model download script
├── cache/                   # Vector store cache
│   └── vector_stores/       # Cached embeddings
└── README.md               # This file
```

### Key Components

#### Backend Components

**1. `main.py` - FastAPI Application**
- Defines REST API endpoints
- Handles CORS and middleware
- Coordinates requests between frontend and RAG pipeline

**2. `rag_pipeline.py` - Core RAG Logic**
- Orchestrates the entire RAG process
- Manages web scraping → embedding → retrieval → generation
- Handles caching and error recovery

**3. `llm_router.py` - LLM Management**
- Dynamic model loading and switching
- Supports local (Transformers) and API-based (OpenAI, Gemini) models
- Automatic model path detection

**4. `vector_store.py` - Vector Operations**
- FAISS vector database management
- Embedding model loading (BGE, SentenceTransformers)
- Similarity search and retrieval

**5. `cached_loader.py` - Web Scraping**
- LangChain-based web crawling
- Intelligent caching to avoid re-processing
- Content extraction and cleaning

**6. `config.py` - Configuration**
- Environment-specific settings
- Dynamic path resolution
- API key management

#### Frontend Components

**1. `App.jsx` - Main Interface**
- Complete RAG interface with form inputs
- Real-time streaming and progress indicators
- Source display with expand/collapse functionality
- Cache management and settings panel

**2. Styling (`App.css`, `index.css`)**
- Modern, responsive design
- Dark/light theme support
- Loading animations and transitions

## 🎨 Design Choices

### Backend Design

#### 1. **Modular Architecture**
- **Separation of Concerns**: Each component handles a specific responsibility
- **Dependency Injection**: Easy to swap components (e.g., different vector stores)
- **Configuration-Driven**: Behavior controlled via config files

#### 2. **LLM Router Pattern**
```python
# Flexible LLM switching
def generate_response(prompt, mode="transformers", api_key=None):
    if mode == "transformers":
        return use_local_model(prompt)
    elif mode == "openai":
        return use_openai_api(prompt, api_key)
    elif mode == "gemini":
        return use_gemini_api(prompt, api_key)
```

**Benefits:**
- Easy to add new LLM providers
- Consistent interface across different models
- Graceful fallback handling

#### 3. **Smart Caching Strategy**
```python
# Cache key: URL + embedding model + settings hash
cache_key = f"{url_hash}_{embedding_model}_{settings_hash}"
```

**Features:**
- Avoids re-processing identical requests
- Model-specific caching (different embeddings for different models)
- Configurable cache invalidation

#### 4. **Vector Store Choice: FAISS**

**Why FAISS over Chroma/Pinecone:**
- **Performance**: Extremely fast similarity search (Facebook's production-grade)
- **Local Storage**: No external dependencies or API limits
- **Memory Efficient**: Optimized for large-scale vector operations
- **Flexibility**: Supports multiple index types and distance metrics

```python
# FAISS implementation
index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
index.add(embeddings)
scores, indices = index.search(query_embedding, top_k)
```

#### 5. **Embedding Model Selection**

**BGE (BAAI General Embedding) - Primary Choice:**
- **State-of-the-art**: Top performance on MTEB benchmarks
- **Multilingual**: Supports 100+ languages
- **Efficient**: Good balance of quality and speed

**Sentence-Transformers - Fallback:**
- **Reliability**: Well-established and stable
- **Variety**: Multiple model options
- **Community**: Large ecosystem and support

### Frontend Design

#### 1. **Real-time Updates**
```jsx
// Server-Sent Events for live updates
const eventSource = new EventSource('/stream');
eventSource.onmessage = (event) => {
    updateLogs(JSON.parse(event.data));
};
```

#### 2. **Progressive Enhancement**
- **Base Functionality**: Works without JavaScript
- **Enhanced Experience**: Real-time updates, animations
- **Responsive Design**: Mobile-first approach

#### 3. **Error Handling**
```jsx
// Graceful error boundaries
try {
    const response = await api.call();
    handleSuccess(response);
} catch (error) {
    handleError(error);
    showFallbackUI();
}
```

### Technology Stack Rationale

#### Backend: FastAPI
**Pros:**
- **Performance**: Async support, fast execution
- **Developer Experience**: Automatic API docs, type hints
- **Modern**: Built-in validation, serialization
- **Ecosystem**: Excellent Python ML library integration

#### Frontend: React + Vite
**Pros:**
- **Development Speed**: Hot reloading, fast builds
- **Component Architecture**: Reusable, maintainable
- **Ecosystem**: Rich library ecosystem
- **Performance**: Optimized bundling and serving

#### Vector Store: FAISS
**Pros:**
- **Speed**: Millisecond search times
- **Scalability**: Handles millions of vectors
- **No Dependencies**: Self-contained, no external services
- **Production Ready**: Used by Meta, Google, etc.

## 📚 API Reference

### Main Endpoints

#### `POST /ask`
Process a RAG query.

**Request:**
```json
{
  "query": "What is Python?",
  "url": "https://python.org",
  "llm_mode": "transformers",
  "max_pages": 5,
  "top_k": 3,
  "force_rebuild": false,
  "api_key": "optional-for-api-models"
}
```

**Response:**
```json
{
  "answer": "Python is a programming language...",
  "sources": ["Source chunk 1...", "Source chunk 2..."],
  "metadata": {
    "pages_crawled": 5,
    "chunks_created": 127,
    "processing_time": 12.5,
    "cache_hit": false
  },
  "logs": ["Starting crawl...", "Processing..."]
}
```

#### `GET /health`
Service health check.

#### `GET /cache/info`
Get cache statistics.

#### `POST /cache/clear`
Clear cache (optionally filtered by URL/model).

### Error Responses

```json
{
  "detail": "Error message",
  "error_type": "ValidationError|ProcessingError|ModelError",
  "suggestions": ["Try reducing max_pages", "Check your API key"]
}
```

## Examples

### Simple question with local Mistral model

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is gen ai securitye?",
    "url": "https://genai.owasp.org",
    "llm_mode": "transformers",
    "max_pages": 3,
    "top_k": 5
  }'
```

### simple question with gemini api
# Using Google Gemini model
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the benefits of using FastAPI?",
    "url": "https://genai.owasp.org",
    "llm_mode": "gemini",
    "max_pages": 4,
    "top_k": 4,
    "api_key": "your-gemini-api-key-here"
  }'

  ## Results

  

## 🔧 Troubleshooting

### Common Issues

#### 1. **Model Not Found**
```
[LLM_ROUTER] ❌ Model path does not exist
```

**Solutions:**
- Run `./scripts/download_models.sh`
- Check `models/mistralai/` directory exists
- Verify model files are present

#### 2. **Memory Issues**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Use CPU mode: Set `device = "cpu"` in config
- Reduce batch size in model loading
- Use smaller embedding models

#### 3. **Frontend Won't Start**
```
Module not found: lucide-react
```

**Solutions:**
- Run `npm install` in frontend directory
- Delete `node_modules` and reinstall
- Check Node.js version (16+ required)

#### 4. **API Connection Issues**
```
Network Error: Connection refused
```

**Solutions:**
- Ensure backend is running on port 8000
- Check CORS settings in `main.py`
- Verify firewall settings

#### 5. **Slow Processing**
**Optimizations:**
- Reduce `max_pages` for faster crawling
- Use smaller embedding models
- Enable GPU acceleration if available
- Implement result caching

### Performance Tuning

#### For Local Models:
```python
# config.py optimizations
torch.set_num_threads(4)  # Adjust based on CPU cores
model.to("mps")  # Use Apple Silicon GPU
```

#### For Large Websites:
```python
# Limit crawling scope
max_pages = 10
max_depth = 2
chunk_size = 500  # Smaller chunks for faster processing
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup

```bash
# Backend development
cd backend
pip install -e .
python -m pytest tests/

# Frontend development
cd frontend
npm run dev
npm test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the excellent document loading and processing framework
- **Hugging Face** for the transformer models and embeddings
- **Facebook Research** for FAISS vector search
- **Mistral AI** for the open-source Mistral model
- **FastAPI** and **React** communities for the amazing frameworks

## 🔮 Future Enhancements

- [ ] **Multiple Vector Stores**: Support for Chroma, Pinecone, Weaviate
- [ ] **Document Upload**: Support for PDF, Word, text files
- [ ] **Advanced Chunking**: Semantic chunking strategies
- [ ] **Model Fine-tuning**: Domain-specific model adaptation
- [ ] **Multi-modal**: Support for images and videos
- [ ] **Collaborative Features**: Shared knowledge bases
- [ ] **Analytics Dashboard**: Usage metrics and insights
- [ ] **Docker Deployment**: Containerized deployment
- [ ] **Kubernetes**: Production-scale orchestration
- [ ] **Model Quantization**: Smaller, faster models

---

**Built with ❤️ for the AI community**

For questions or support, please open an issue or contact [your-email@example.com](mailto:your-email@example.com).
