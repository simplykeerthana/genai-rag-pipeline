from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Optional, Tuple
import os
import pickle
import hashlib
import json
from datetime import datetime, timedelta
import torch

# Cache directory
CACHE_DIR = "app/vector_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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
    # Include model in cache path so different models have separate caches
    vector_path = os.path.join(CACHE_DIR, f"vector_{url_hash}_{model_key}.faiss")
    meta_path = os.path.join(CACHE_DIR, f"meta_{url_hash}_{model_key}.json")
    embeddings_path = os.path.join(CACHE_DIR, f"embeddings_{url_hash}_{model_key}.pkl")
    return vector_path, meta_path, embeddings_path

def is_cache_valid(meta_path: str, max_age_hours: int = 24) -> bool:
    """Check if cache is still valid based on age"""
    if not os.path.exists(meta_path):
        return False
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        cached_time = datetime.fromisoformat(meta['timestamp'])
        age = datetime.now() - cached_time
        
        return age < timedelta(hours=max_age_hours)
    except:
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

def build_or_load_vector_store(
    docs: List[Dict[str, str]], 
    url: str, 
    force_rebuild: bool = False,
    max_cache_age_hours: int = 24,
    model_key: str = DEFAULT_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    log_func=None
) -> Optional[FAISS]:
    """
    Build a new vector store or load from cache if available and valid.
    
    Args:
        docs: List of documents to index
        url: The base URL being indexed
        force_rebuild: Force rebuilding even if cache exists
        max_cache_age_hours: Maximum age of cache in hours
        model_key: Which embedding model to use
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        log_func: Optional logging function
    
    Returns:
        FAISS vector store or None if failed
    """
    if log_func is None:
        log_func = print
    
    # Get model-specific paths
    vector_path, meta_path, embeddings_path = get_cache_paths(url, model_key)
    
    # Try to load from cache if not forcing rebuild
    if not force_rebuild and os.path.exists(vector_path) and is_cache_valid(meta_path, max_cache_age_hours):
        try:
            log_func(f"[INFO] Loading vector store from cache for {url}")
            
            # Load metadata to check model
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Load the appropriate embedding model
            cached_model_key = meta.get('embedding_model', 'minilm')
            embedding_model, model_info = get_embedding_model(cached_model_key)
            
            # Load vector store
            vector_store = FAISS.load_local(
                vector_path, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            
            log_func(f"[INFO] Loaded cached vector store: {meta['chunk_count']} chunks from {meta['doc_count']} documents")
            log_func(f"[INFO] Using model: {model_info['description']}")
            log_func(f"[INFO] Cache age: {datetime.now() - datetime.fromisoformat(meta['timestamp'])}")
            
            return vector_store
            
        except Exception as e:
            log_func(f"[WARNING] Failed to load cache: {e}. Rebuilding...")
    
    # Build new vector store
    log_func(f"[INFO] Building new vector store for {url}")
    
    # Get the embedding model
    embedding_model, model_info = get_embedding_model(model_key)
    log_func(f"[INFO] Using embedding model: {model_info['description']}")
    
    # Check if we have any documents
    if not docs:
        log_func("[WARNING] No documents provided to build_vector_store")
        return None
    
    all_documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    valid_doc_count = 0
    total_chunks = 0
    
    for doc in docs:
        # Skip empty documents
        if not doc.get("content", "").strip():
            log_func(f"[WARNING] Skipping empty document from {doc.get('source_url', 'unknown')}")
            continue
        
        valid_doc_count += 1
        
        # Add document metadata to content for better context
        content = doc["content"]
        if doc.get("title"):
            content = f"Title: {doc['title']}\nURL: {doc['source_url']}\n\n{content}"
        
        # For BGE models, add instruction prefix to improve retrieval
        if 'bge' in model_key:
            # BGE models work better with instruction prefixes
            content = f"passage: {content}"
        
        chunks = splitter.split_text(content)
        total_chunks += len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            metadata = {
                "source_url": doc["source_url"],
                "chunk_id": i,
                "chunk_size": len(chunk),
                "depth": doc.get("depth", 0),
                "title": doc.get("title", ""),
                "total_chunks": len(chunks)
            }
            all_documents.append(Document(page_content=chunk, metadata=metadata))
    
    log_func(f"[INFO] Processed {valid_doc_count} documents into {total_chunks} chunks")
    
    # Check if we have any documents after processing
    if not all_documents:
        log_func("[WARNING] No valid documents to index after processing")
        return None
    
    log_func(f"[INFO] Creating vector store with {len(all_documents)} chunks from {valid_doc_count} documents")
    
    try:
        # Create vector store with progress indication
        log_func(f"[INFO] Embedding documents... (this may take a moment)")
        
        # For large document sets, process in batches
        if len(all_documents) > 100:
            log_func(f"[INFO] Processing {len(all_documents)} documents in batches...")
            
            # Process in batches of 50
            batch_size = 50
            embeddings_list = []
            
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                log_func(f"[INFO] Processing batch {i//batch_size + 1}/{(len(all_documents) + batch_size - 1)//batch_size}")
                
                # Create temporary vector store for batch
                if i == 0:
                    vector_store = FAISS.from_documents(batch, embedding_model)
                else:
                    batch_store = FAISS.from_documents(batch, embedding_model)
                    vector_store.merge_from(batch_store)
        else:
            vector_store = FAISS.from_documents(all_documents, embedding_model)
        
        # Save to cache
        vector_store.save_local(vector_path)
        
        # Save metadata
        save_cache_metadata(url, meta_path, valid_doc_count, len(all_documents), model_key)
        
        log_func(f"[INFO] Vector store created and cached successfully")
        return vector_store
        
    except Exception as e:
        log_func(f"[ERROR] Failed to create vector store: {e}")
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