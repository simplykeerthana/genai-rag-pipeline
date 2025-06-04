from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_pipeline import run_rag_pipeline
from app.vector_store import clear_cache, get_cache_info
import logging
from datetime import datetime
from typing import List, Optional
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", version="1.0.0")

# Configure CORS properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    url: str
    force_refresh: Optional[bool] = False
    embedding_model: Optional[str] = "bge-base"
    max_pages: Optional[int] = 30
    max_depth: Optional[int] = 2
    top_k: Optional[int] = 5
    llm_model: Optional[str] = None  # New field
    api_key: Optional[str] = None    # New field

# Global log storage for current request
current_logs: List[dict] = []

def capture_log(message: str, level: str = "INFO"):
    """Capture log messages"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    current_logs.append(log_entry)
    
    # Also log to standard logger
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)

@app.get("/")
async def root():
    return {
        "message": "RAG API is running",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /ask - Submit a query",
            "health": "GET /health - Health check",
            "cache_info": "GET /cache/info - Get cache information",
            "clear_cache": "POST /cache/clear - Clear cache",
            "models": "GET /models - Get available models"
        }
    }

@app.get("/models")
async def get_models():
    """Get available LLM models"""
    from app.config import LLM_MODE, TRANSFORMER_MODEL_PATH
    import os
    
    models = {
        "local": [],
        "api": []
    }
    
    # Check local models
    if os.path.exists(TRANSFORMER_MODEL_PATH):
        model_name = os.path.basename(TRANSFORMER_MODEL_PATH)
        models["local"].append({
            "id": "transformers",
            "name": f"Local: {model_name}",
            "description": "Local transformer model using MPS/GPU",
            "requires_api_key": False
        })
    
    # API models
    models["api"] = [
        {
            "id": "gemini",
            "name": "Google Gemini 2.0 Flash",
            "description": "Fast, efficient API model from Google",
            "requires_api_key": True
        },
        {
            "id": "openai",
            "name": "OpenAI GPT-3.5 Turbo",
            "description": "OpenAI's fast chat model",
            "requires_api_key": True
        }
    ]
    
    return {
        "current_mode": LLM_MODE,
        "models": models
    }

@app.post("/ask")
async def ask(request: QueryRequest):
    global current_logs
    current_logs = []  # Reset logs for new request
    
    # Log request details
    logger.info(f"Received request - Query: {request.query}, URL: {request.url}")
    capture_log(f"Starting RAG pipeline for query: {request.query}")
    capture_log(f"Target URL: {request.url}")
    capture_log(f"Settings: max_pages={request.max_pages}, max_depth={request.max_depth}, top_k={request.top_k}")
    capture_log(f"Embedding model: {request.embedding_model}")
    
    # Log LLM model selection
    if request.llm_model:
        capture_log(f"LLM model: {request.llm_model}")
        if request.llm_model in ["gemini", "openai"]:
            capture_log(f"Using API model: {request.llm_model}")
    
    # Handle cache clearing if force refresh requested
    if request.force_refresh:
        capture_log("Force refresh requested - clearing cache for this URL")
        try:
            clear_cache(request.url, request.embedding_model)
            capture_log(f"Successfully cleared cache for {request.url}")
        except Exception as e:
            capture_log(f"Warning: Could not clear cache: {e}", "WARNING")
    
    try:
        # Run the RAG pipeline with model selection
        result = run_rag_pipeline(
            query=request.query,
            url=request.url,
            force_rebuild=request.force_refresh,
            embedding_model=request.embedding_model,
            log_func=capture_log,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            top_k=request.top_k,
            llm_model=request.llm_model,  # Pass model selection
            api_key=request.api_key        # Pass API key if provided
        )
        
        # Add logs to response
        result["logs"] = current_logs
        
        # Log summary
        if "metadata" in result:
            meta = result["metadata"]
            capture_log(
                f"Pipeline completed - Crawled {meta.get('total_pages_crawled', 0)} pages, "
                f"created {meta.get('chunks_found', 0)} chunks, "
                f"found {meta.get('unique_sources', 0)} unique sources"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        capture_log(f"Error: {str(e)}", "ERROR")
        
        # Return error response
        return {
            "answer": f"An error occurred while processing your request: {str(e)}",
            "sources": [],
            "detailed_sources": [],
            "metadata": {
                "total_pages_crawled": 0,
                "chunks_found": 0,
                "base_url": request.url,
                "error": str(e)
            },
            "logs": current_logs
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cache/info")
async def cache_info():
    """Get information about cached vector stores"""
    try:
        info = get_cache_info()
        return {
            "cache_count": len(info),
            "cache_info": info
        }
    except Exception as e:
        return {
            "error": str(e),
            "cache_info": []
        }

@app.post("/cache/clear")
async def clear_cache_endpoint(url: Optional[str] = None, model: Optional[str] = None):
    """Clear cache for a specific URL/model or all caches"""
    try:
        if url:
            clear_cache(url, model)
            message = f"Cleared cache for {url}"
            if model:
                message += f" with model {model}"
        else:
            clear_cache()
            message = "Cleared all caches"
            
        return {"message": message, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

# Handle OPTIONS requests for CORS
@app.options("/ask")
async def options_ask():
    return {"message": "OK"}

@app.options("/cache/clear")
async def options_cache_clear():
    return {"message": "OK"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )