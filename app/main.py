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
    llm_mode: Optional[str] = "transformers"  # New field for LLM selection
    api_key: Optional[str] = None  # New field for API key

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
            "clear_cache": "POST /cache/clear - Clear cache"
        }
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
    capture_log(f"LLM mode: {request.llm_mode}")  # Log LLM mode
    
    # Handle cache clearing if force refresh requested
    if request.force_refresh:
        capture_log("Force refresh requested - clearing cache for this URL")
        try:
            clear_cache(request.url, request.embedding_model)
            capture_log(f"Successfully cleared cache for {request.url}")
        except Exception as e:
            capture_log(f"Warning: Could not clear cache: {e}", "WARNING")
    
    try:
        # Run the RAG pipeline with LLM configuration
        result = run_rag_pipeline(
            query=request.query,
            url=request.url,
            force_rebuild=request.force_refresh,
            embedding_model=request.embedding_model,
            log_func=capture_log,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            top_k=request.top_k,
            llm_mode=request.llm_mode,  # Pass LLM mode
            api_key=request.api_key  # Pass API key
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

@app.post("/debug/ask")
async def debug_ask(request: QueryRequest):
    """Debug endpoint that returns raw data at each stage"""
    global current_logs
    current_logs = []
    
    debug_data = {
        "request": request.dict(),
        "stages": {},
        "logs": []
    }
    
    try:
        # Stage 1: URL Discovery
        capture_log("Stage 1: URL Discovery", "DEBUG")
        from app.rag_pipeline import discover_urls_by_crawling
        
        urls = discover_urls_by_crawling(
            request.url, 
            max_pages=request.max_pages, 
            max_depth=request.max_depth,
            log_func=capture_log
        )
        
        debug_data["stages"]["url_discovery"] = {
            "urls_found": len(urls),
            "urls": urls[:10]  # First 10 URLs
        }
        
        # Stage 2: Content Loading
        capture_log("Stage 2: Content Loading", "DEBUG")
        from langchain_community.document_loaders import WebBaseLoader
        
        if urls:
            loader = WebBaseLoader(urls[:3])  # Load first 3 for debug
            docs = loader.load()
            
            debug_data["stages"]["content_loading"] = {
                "docs_loaded": len(docs),
                "sample_doc": {
                    "url": docs[0].metadata.get('source', '') if docs else '',
                    "content_length": len(docs[0].page_content) if docs else 0,
                    "content_preview": docs[0].page_content[:500] if docs else '',
                    "raw_content_sample": repr(docs[0].page_content[:200]) if docs else '',
                    "metadata": docs[0].metadata if docs else {}
                } if docs else None
            }
        
        # Stage 3: Vector Store Search
        capture_log("Stage 3: Vector Store Search", "DEBUG")
        from app.vector_store import build_or_load_vector_store
        
        # Try to load existing vector store
        vector_store = build_or_load_vector_store(
            docs=[],  # Empty to just load
            url=request.url,
            force_rebuild=False,
            model_key=request.embedding_model,
            log_func=capture_log
        )
        
        if vector_store:
            results = vector_store.similarity_search_with_score(request.query, k=request.top_k)
            
            debug_data["stages"]["vector_search"] = {
                "results_found": len(results),
                "results": [
                    {
                        "score": float(score),
                        "source_url": doc.metadata.get('source_url', ''),
                        "chunk_id": doc.metadata.get('chunk_id', 0),
                        "content_length": len(doc.page_content),
                        "content_preview": doc.page_content[:200],
                        "raw_content": repr(doc.page_content[:100]),
                        "metadata": doc.metadata
                    }
                    for doc, score in results
                ]
            }
            
            # Stage 4: Context Building
            context_parts = []
            for i, (doc, score) in enumerate(results):
                context_parts.append(f"[Source {i+1}: {doc.metadata.get('source_url', 'unknown')}]")
                context_parts.append(doc.page_content)
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            debug_data["stages"]["context_building"] = {
                "context_length": len(context),
                "context_preview": context[:500],
                "raw_context": repr(context[:200])
            }
            
            # Stage 5: LLM Generation
            capture_log("Stage 4: LLM Generation", "DEBUG")
            prompt = f"Based on the context below, answer: {request.query}\n\nContext:\n{context[:2000]}"
            
            debug_data["stages"]["llm_generation"] = {
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:500],
                "prompt_full": prompt,
                "llm_mode": request.llm_mode  # Include LLM mode in debug
            }
            
            from app.llm_router import generate_response
            try:
                # Use LLM mode and API key from request
                answer = generate_response(
                    prompt, 
                    max_length=300,
                    mode=request.llm_mode,
                    api_key=request.api_key
                )
                debug_data["stages"]["llm_generation"]["response"] = {
                    "answer": answer,
                    "answer_length": len(answer),
                    "raw_answer": repr(answer)
                }
            except Exception as e:
                debug_data["stages"]["llm_generation"]["error"] = str(e)
        
        debug_data["logs"] = current_logs
        return debug_data
        
    except Exception as e:
        debug_data["error"] = str(e)
        debug_data["logs"] = current_logs
        return debug_data

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

    @app.post("/test/gemini")
    async def test_gemini(api_key: str):
        """Test endpoint to verify Gemini API"""
        from app.llm_router import generate_response
        
        test_prompt = "Say 'Gemini is working!' if you receive this."
        
        try:
            response = generate_response(
                prompt=test_prompt,
                max_length=100,
                mode="gemini",
                api_key=api_key
            )
            
            return {
                "success": "Gemini is working" in response,
                "response": response,
                "message": "API key is valid!" if "Gemini is working" in response else "Check the response"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "API key may be invalid or there's a connection issue"
            }