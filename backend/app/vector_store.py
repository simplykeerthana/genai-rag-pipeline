from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # <-- updated import
from typing import List, Dict, Optional, Tuple
import os
import pickle
import hashlib
import json
from datetime import datetime, timedelta
import torch

# Cache directory structure
CACHE_DIR = "cache"
CONTENT_CACHE_DIR = os.path.join(CACHE_DIR, "content")  # For raw HTML/text
VECTOR_CACHE_DIR = os.path.join(CACHE_DIR, "vectors")   # For FAISS indices

# Create cache directories
os.makedirs(CONTENT_CACHE_DIR, exist_ok=True)
os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)

# Embedding model options (from best to fastest)
EMBEDDING_MODELS = {
    # Best quality models
    "bge-large": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "description": "Best quality, slower (1024d)",
        "max_length": 512
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5", 
        "dimension": 768,
        "description": "Great quality, balanced (768d)",
        "max_length": 512
    },
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "description": "Good quality, fast (384d)",
        "max_length": 512
    },
    
    # OpenAI-compatible models
    "gte-base": {
        "name": "thenlper/gte-base",
        "dimension": 768,
        "description": "OpenAI alternative, good quality (768d)",
        "max_length": 512
    },
    "gte-small": {
        "name": "thenlper/gte-small",
        "dimension": 384,
        "description": "OpenAI alternative, fast (384d)",
        "max_length": 512
    },
    
    # Specialized for Q&A
    "e5-base": {
        "name": "intfloat/e5-base-v2",
        "dimension": 768,
        "description": "Optimized for Q&A tasks (768d)",
        "max_length": 512
    },
    
    # Original (for compatibility)
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Original model, fast but lower quality (384d)",
        "max_length": 256
    }
}

# Default model selection (balance of quality and speed)
DEFAULT_MODEL = "bge-base"

# Check if CUDA is available for GPU acceleration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(f"[INFO] GPU detected! Using CUDA for embeddings")
else:
    print(f"[INFO] No GPU detected, using CPU for embeddings")

def get_embedding_model(model_key: str = DEFAULT_MODEL, normalize_embeddings: bool = True):
    """
    Get the embedding model with proper configuration
    
    Args:
        model_key: Key from EMBEDDING_MODELS dict
        normalize_embeddings: Whether to normalize embeddings (recommended for similarity search)
    """
    if model_key not in EMBEDDING_MODELS:
        print(f"[WARNING] Unknown model {model_key}, using default {DEFAULT_MODEL}")
        model_key = DEFAULT_MODEL
    
    model_info = EMBEDDING_MODELS[model_key]
    print(f"[INFO] Loading embedding model: {model_info['description']}")
    
    # Model configuration
    model_kwargs = {'device': DEVICE}
    
    # For BGE models, add instruction prefix for better performance
    encode_kwargs = {'normalize_embeddings': normalize_embeddings}
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_info['name'],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embedding_model, model_info

# Initialize default embedding model
embedding_model, current_model_info = get_embedding_model(DEFAULT_MODEL)

def get_url_hash(url: str) -> str:
    """Generate a unique hash for a URL"""
    return hashlib.md5(url.encode()).hexdigest()

def get_cache_paths(url: str, model_key: str) -> Tuple[str, str, str]:
    """Get cache file paths for a given URL and model"""
    url_hash = get_url_hash(url)
    vector_path = os.path.join(VECTOR_CACHE_DIR, f"vector_{url_hash}_{model_key}.faiss")
    meta_path = os.path.join(VECTOR_CACHE_DIR, f"meta_{url_hash}_{model_key}.json")
    embeddings_path = os.path.join(VECTOR_CACHE_DIR, f"embeddings_{url_hash}_{model_key}.pkl")
    
    print(f"[DEBUG] Vector store cache paths:")
    print(f"  Vector: {vector_path}")
    print(f"  Meta: {meta_path}")
    print(f"  Embeddings: {embeddings_path}")
    
    return vector_path, meta_path, embeddings_path

def is_cache_valid(meta_path: str, max_age_hours: int = 24) -> bool:
    try:
        if not os.path.exists(meta_path):
            print(f"[CACHE] Metadata file not found: {meta_path}")
            return False
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            print(f"[CACHE] Loaded metadata: {meta}")
        
        # Check timestamp
        cache_time = datetime.fromisoformat(meta['timestamp'])
        age = datetime.now() - cache_time
        cache_age_hours = age.total_seconds() / 3600
        is_valid = cache_age_hours < max_age_hours
        
        print(f"[CACHE] Cache age: {cache_age_hours:.1f} hours")
        print(f"[CACHE] Max age allowed: {max_age_hours} hours")
        print(f"[CACHE] Cache is {'valid' if is_valid else 'expired'}")
        
        return is_valid
        
    except Exception as e:
        print(f"[CACHE] Validation error: {str(e)}")
        return False

def save_cache_metadata(url: str, meta_path: str, doc_count: int, chunk_count: int, model_key: str):
    """Save metadata about the cached vector store"""
    meta = {
        'url': url,
        'timestamp': datetime.now().isoformat(),
        'doc_count': doc_count,
        'chunk_count': chunk_count,
        'embedding_model': model_key,
        'model_info': EMBEDDING_MODELS[model_key]
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

async def build_or_load_vector_store(
    docs: Optional[List[Dict[str, str]]], 
    url: str, 
    force_rebuild: bool = False,
    max_cache_age_hours: int = 24,
    model_key: str = DEFAULT_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    log_func=print
) -> Optional[FAISS]:
    
    vector_path, meta_path, _ = get_cache_paths(url, model_key)
    
    # Try loading from cache first
    if not force_rebuild and os.path.exists(vector_path) and is_cache_valid(meta_path):
        try:
            embedding_model, _ = get_embedding_model(model_key)
            vector_store = FAISS.load_local(
                vector_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            log_func(f"[CACHE] Loaded vector store from cache: {vector_path}")
            return vector_store
        except Exception as e:
            log_func(f"[ERROR] Failed to load cache: {str(e)}")
    
    # If we get here, we need to build new vector store
    if not docs:
        log_func("[ERROR] No documents provided for building vector store")
        return None
        
    try:
        embedding_model, _ = get_embedding_model(model_key)
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Convert to LangChain documents
        documents = [
            Document(
                page_content=doc["content"],
                metadata={**doc.get("metadata", {}), "source_url": doc["source_url"]}
            )
            for doc in docs
        ]
        
        # Split into chunks
        chunks = text_splitter.split_documents(documents)
        log_func(f"[BUILD] Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embedding_model)
        
        # Save to cache
        vector_store.save_local(vector_path)
        save_cache_metadata(url, meta_path, len(documents), len(chunks), model_key)
        log_func(f"[CACHE] Saved new vector store to: {vector_path}")
        
        return vector_store
        
    except Exception as e:
        log_func(f"[ERROR] Failed to build vector store: {str(e)}")
        return None

def optimize_search_query(query: str, model_key: str) -> str:
    """
    Optimize the search query based on the embedding model
    
    BGE models work better with question prefixes
    """
    if 'bge' in model_key:
        return f"query: {query}"
    elif 'e5' in model_key:
        return f"query: {query}"
    return query

def similarity_search_with_model(
    vector_store: FAISS,
    query: str,
    k: int = 3,
    model_key: str = DEFAULT_MODEL
) -> List[Tuple[Document, float]]:
    """
    Perform similarity search with query optimization for the model
    """
    optimized_query = optimize_search_query(query, model_key)
    return vector_store.similarity_search_with_score(optimized_query, k=k)
def clear_cache(url: Optional[str] = None, model_key: Optional[str] = None):
    """
    Clear cache for a specific URL/model or all caches.
    Handles permission errors gracefully.
    
    Args:
        url: Specific URL to clear cache for, or None to clear all
        model_key: Specific model to clear cache for, or None for all models
    """
    import time
    
    def safe_remove(path):
        """Safely remove a file, handling permission errors"""
        if not os.path.exists(path):
            return True
            
        try:
            os.remove(path)
            return True
        except PermissionError:
            # Try to rename instead of delete
            try:
                backup_path = f"{path}.old_{int(time.time())}"
                os.rename(path, backup_path)
                print(f"Renamed {path} to {backup_path} (couldn't delete due to permissions)")
                return True
            except Exception as e:
                print(f"WARNING: Could not remove or rename {path}: {e}")
                return False
    
    removed_count = 0
    failed_count = 0
    
    if url:
        if model_key:
            # Clear specific URL and model
            vector_path, meta_path, embeddings_path = get_cache_paths(url, model_key)
            for path in [vector_path, meta_path, embeddings_path]:
                if safe_remove(path):
                    removed_count += 1
                else:
                    failed_count += 1
            print(f"Cleared cache for {url} with model {model_key} ({removed_count} files removed, {failed_count} failed)")
        else:
            # Clear all models for this URL
            for mk in EMBEDDING_MODELS.keys():
                vector_path, meta_path, embeddings_path = get_cache_paths(url, mk)
                for path in [vector_path, meta_path, embeddings_path]:
                    if safe_remove(path):
                        removed_count += 1
                    else:
                        failed_count += 1
            print(f"Cleared all caches for {url} ({removed_count} files removed, {failed_count} failed)")
    else:
        # Clear all caches
        try:
            for file in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, file)
                if safe_remove(file_path):
                    removed_count += 1
                else:
                    failed_count += 1
            print(f"Cleared all vector store caches ({removed_count} files removed, {failed_count} failed)")
        except Exception as e:
            print(f"Error accessing cache directory: {e}")
    
    if failed_count > 0:
        print(f"NOTE: {failed_count} files could not be removed due to permissions. They were renamed with .old suffix.")
        print("You may need to manually delete these files or run with appropriate permissions.")
    """
    Clear cache for a specific URL/model or all caches.
    
    Args:
        url: Specific URL to clear cache for, or None to clear all
        model_key: Specific model to clear cache for, or None for all models
    """
    if url:
        if model_key:
            # Clear specific URL and model
            vector_path, meta_path, embeddings_path = get_cache_paths(url, model_key)
            for path in [vector_path, meta_path, embeddings_path]:
                if os.path.exists(path):
                    os.remove(path)
            print(f"Cleared cache for {url} with model {model_key}")
        else:
            # Clear all models for this URL
            for mk in EMBEDDING_MODELS.keys():
                vector_path, meta_path, embeddings_path = get_cache_paths(url, mk)
                for path in [vector_path, meta_path, embeddings_path]:
                    if os.path.exists(path):
                        os.remove(path)
            print(f"Cleared all caches for {url}")
    else:
        # Clear all caches
        for file in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, file))
        print("Cleared all vector store caches")

def get_cache_info() -> List[Dict]:
    """Get information about all cached vector stores"""
    cache_info = []
    
    for file in os.listdir(CACHE_DIR):
        if file.startswith("meta_") and file.endswith(".json"):
            meta_path = os.path.join(CACHE_DIR, file)
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                cached_time = datetime.fromisoformat(meta['timestamp'])
                age = datetime.now() - cached_time
                
                cache_info.append({
                    'url': meta['url'],
                    'timestamp': meta['timestamp'],
                    'age_hours': age.total_seconds() / 3600,
                    'doc_count': meta['doc_count'],
                    'chunk_count': meta['chunk_count'],
                    'model': meta.get('embedding_model', 'unknown'),
                    'model_info': meta.get('model_info', {})
                })
            except:
                continue
    
    return cache_info

# Keep the original functions for backward compatibility
def build_vector_store(docs: List[Dict[str, str]], log_func=None) -> Optional[FAISS]:
    """Legacy function - builds without caching"""
    if not docs:
        return None
    
    # Use a dummy URL for legacy compatibility
    dummy_url = "legacy_build"
    return build_or_load_vector_store(
        docs, 
        dummy_url, 
        force_rebuild=True, 
        model_key=DEFAULT_MODEL,
        log_func=log_func
    )

def load_vector_store(log_func=None) -> Optional[FAISS]:
    """Legacy function - kept for compatibility"""
    return None

def inspect_cache_contents(url: str, model_key: str = DEFAULT_MODEL) -> Dict:
    """
    Inspect the contents of a cached vector store.
    Returns details about the cache and sample contents.
    """
    vector_path, meta_path, embeddings_path = get_cache_paths(url, model_key)
    
    results = {
        "cache_exists": False,
        "metadata": None,
        "vector_store_info": None,
        "error": None
    }
    
    try:
        # Check if cache exists
        if not os.path.exists(vector_path) or not os.path.exists(meta_path):
            print(f"[CACHE] Cache files not found for {url}")
            return results
            
        results["cache_exists"] = True
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            results["metadata"] = meta
            print(f"\n[CACHE] Metadata for {url}:")
            print(f"  Created: {meta['timestamp']}")
            print(f"  Documents: {meta['doc_count']}")
            print(f"  Chunks: {meta['chunk_count']}")
            print(f"  Model: {meta['embedding_model']}")
        
        # Load vector store
        embedding_model, _ = get_embedding_model(meta.get('embedding_model', DEFAULT_MODEL))
        vector_store = FAISS.load_local(
            vector_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Get some sample contents
        samples = vector_store.similarity_search("", k=3)  # Get 3 random documents
        
        results["vector_store_info"] = {
            "sample_chunks": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in samples
            ]
        }
        
        return results
        
    except Exception as e:
        import traceback
        error = f"Error inspecting cache: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error}")
        results["error"] = error
        return results

def check_cache_status(url: str, model_key: str = DEFAULT_MODEL) -> Dict[str, bool]:
    """
    Check status of both content and vector store caches
    
    Args:
        url: URL to check cache for
        model_key: Model used for embeddings
        
    Returns:
        Dict containing cache status information
    """
    url_hash = get_url_hash(url)
    vector_path, meta_path, embeddings_path = get_cache_paths(url, model_key)
    
    # Check vector store cache
    vector_exists = os.path.exists(vector_path)
    meta_exists = os.path.exists(meta_path)
    vector_valid = is_cache_valid(meta_path) if meta_exists else False
    
    print(f"[CACHE] Status for {url}:")
    print(f"  Vector store: {'✓' if vector_exists else '✗'}")
    print(f"  Metadata: {'✓' if meta_exists else '✗'}")
    print(f"  Valid: {'✓' if vector_valid else '✗'}")
    
    return {
        "vector_store_exists": vector_exists,
        "metadata_exists": meta_exists,
        "vector_store_valid": vector_valid
    }