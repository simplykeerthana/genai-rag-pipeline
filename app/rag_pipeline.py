# app/rag_pipeline.py
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.llm_router import generate_response
from app.vector_store import build_or_load_vector_store, get_cache_paths, is_cache_valid
from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
import os
import json
from datetime import datetime

def clean_web_content(text: str) -> str:
    """Clean text content from web pages"""
    if not text:
        return ""
    
    # Fix encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    
    # Replace common unicode characters
    replacements = {
        '\u2019': "'", '\u2018': "'",
        '\u201d': '"', '\u201c': '"',
        '\u2013': '-', '\u2014': '--',
        '\u2026': '...', '\xa0': ' ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def discover_urls_by_crawling(base_url: str, max_pages: int = 30, max_depth: int = 2, log_func=None) -> List[str]:
    """Discover URLs by crawling with depth limit"""
    if log_func is None:
        log_func = print
    
    log_func(f"[DISCOVERY] Mapping site structure of {base_url}")
    
    domain = urlparse(base_url).netloc
    visited = set()
    to_visit = [(base_url, 0)]  # (url, depth)
    discovered_urls = []
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; RAG-Bot/1.0)'
    })

    while to_visit and len(discovered_urls) < max_pages:
        current_url, depth = to_visit.pop(0)
        
        if current_url in visited or depth > max_depth:
            continue

        visited.add(current_url)

        try:
            response = session.get(current_url, timeout=10)
            if response.status_code != 200:
                continue

            discovered_urls.append(current_url)
            log_func(f"[DISCOVERY] Mapped URL at depth {depth}: {current_url}")
            
            # Only crawl deeper if not at max depth
            if depth < max_depth:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = urljoin(current_url, link['href'])
                    parsed = urlparse(href)
                    
                    # Only follow links on same domain
                    if parsed.netloc == domain and href not in visited:
                        # Skip common non-content URLs
                        skip_patterns = ['.pdf', '.jpg', '.png', '.gif', '.zip', '#', 'javascript:', 'mailto:']
                        if not any(pattern in href.lower() for pattern in skip_patterns):
                            to_visit.append((href, depth + 1))
            
            time.sleep(0.5)  # Be polite
            
        except Exception as e:
            log_func(f"[ERROR] Failed to map {current_url}: {e}")
            continue

    log_func(f"[DISCOVERY] Site mapping complete: {len(discovered_urls)} URLs found")
    return discovered_urls

def process_documents(documents: List, log_func=None) -> List[Dict[str, str]]:
    """Process LangChain documents and clean their content"""
    if log_func is None:
        log_func = print
    
    processed_docs = []
    
    for doc in documents:
        try:
            # Clean the content
            cleaned_content = clean_web_content(doc.page_content)
            
            # Skip if content is too short
            if len(cleaned_content) < 100:
                log_func(f"[SKIP] Content too short from {doc.metadata.get('source', 'unknown')}")
                continue
            
            # Extract title from metadata or content
            title = doc.metadata.get('title', '')
            if not title and cleaned_content:
                # Try to extract first heading or first line as title
                lines = cleaned_content.split('\n')
                title = lines[0][:100] if lines else 'Untitled'
            
            processed_docs.append({
                "content": cleaned_content,
                "source_url": doc.metadata.get('source', ''),
                "title": clean_web_content(title),
                "depth": 0,
                "metadata": doc.metadata
            })
            
            log_func(f"[PROCESS] Processed {len(cleaned_content)} chars from {doc.metadata.get('source', 'unknown')}")
            
        except Exception as e:
            log_func(f"[ERROR] Failed to process document: {e}")
            continue
    
    return processed_docs

def check_cache_and_load(url: str, embedding_model: str, log_func) -> Tuple[bool, Optional[object], Optional[dict]]:
    """Check cache status and load if valid - separate function for modularity"""
    vector_path, meta_path, _ = get_cache_paths(url, embedding_model)
    cache_exists = os.path.exists(vector_path) and os.path.exists(meta_path)
    cache_valid = False
    cache_meta = None
    vector_store = None
    
    if cache_exists:
        cache_valid = is_cache_valid(meta_path)
        if cache_valid:
            with open(meta_path, 'r') as f:
                cache_meta = json.load(f)
                cache_age = datetime.now() - datetime.fromisoformat(cache_meta['timestamp'])
                log_func(f"[CACHE] Found valid cache (age: {cache_age})")
                
            # Load vector store directly
            log_func(f"[CACHE] Using cached data - skipping web crawl entirely")
            log_func(f"[CACHE] Loading {cache_meta.get('chunk_count', 0)} pre-indexed chunks from {cache_meta.get('doc_count', 0)} documents")
            
            vector_store = build_or_load_vector_store(
                docs=[],  # Empty docs since we're just loading
                url=url,
                force_rebuild=False,
                model_key=embedding_model,
                log_func=log_func
            )
            
    return cache_valid, vector_store, cache_meta

def crawl_and_index(url: str, max_pages: int, max_depth: int, embedding_model: str, log_func) -> Tuple[Optional[object], Optional[dict]]:
    """Crawl website and build vector index - separate function for modularity"""
    # Discover URLs
    log_func(f"[CRAWL] Starting web crawl of {url} (max_pages={max_pages}, max_depth={max_depth})")
    urls = discover_urls_by_crawling(url, max_pages, max_depth, log_func)
    
    if not urls:
        return None, None

    # Load content using WebBaseLoader
    log_func(f"[DOWNLOAD] Downloading content from {len(urls)} URLs")
    all_documents = []
    
    # Process URLs in smaller batches
    batch_size = 5
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        log_func(f"[DOWNLOAD] Processing batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size}")
        
        try:
            loader = WebBaseLoader(batch_urls)
            loader.requests_per_second = 2  # Rate limit
            
            # Load documents
            docs = loader.load()
            
            # Make sure each doc knows its source URL
            for doc, url in zip(docs, batch_urls):
                doc.metadata['source'] = url
                doc.metadata['source_url'] = url
            
            all_documents.extend(docs)
            
        except Exception as e:
            log_func(f"[ERROR] Failed to load batch: {e}")
            continue

    if not all_documents:
        return None, None

    # Process and clean documents
    log_func(f"[PROCESS] Processing {len(all_documents)} documents")
    processed_docs = process_documents(all_documents, log_func)
    
    if not processed_docs:
        return None, None

    # Build vector store
    log_func(f"[INDEX] Building vector index with {len(processed_docs)} documents")
    vector_store = build_or_load_vector_store(
        docs=processed_docs,
        url=url,
        force_rebuild=True,
        model_key=embedding_model,
        chunk_size=500,
        chunk_overlap=50,
        log_func=log_func
    )

    cache_meta = {
        'doc_count': len(processed_docs),
        'chunk_count': len(vector_store.docstore._dict) if hasattr(vector_store, 'docstore') else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    return vector_store, cache_meta

def search_and_generate(
    vector_store,
    query: str,
    top_k: int,
    llm_mode: str,
    api_key: Optional[str],
    log_func
) -> Tuple[str, List[dict], List[str], set]:
    """Search for relevant chunks and generate answer - separate function for modularity"""
    # Search for relevant chunks
    log_func(f"[SEARCH] Finding relevant content for: {query}")
    results = vector_store.similarity_search_with_score(query, k=top_k)
    
    if not results:
        return "No relevant content found for your query.", [], [], set()

    # Prepare context and sources
    log_func(f"[SEARCH] Found {len(results)} relevant chunks")
    
    context_parts = []
    sources = []
    detailed_sources = []
    unique_sources = set()

    for i, (doc, score) in enumerate(results):
        source_url = doc.metadata.get("source_url", "unknown")
        chunk_content = doc.page_content.strip()
        
        # Build context
        context_parts.append(f"[Source {i+1}: {source_url}]")
        context_parts.append(chunk_content)
        context_parts.append("")
        
        # Track sources
        sources.append(source_url)
        unique_sources.add(source_url)
        
        # Detailed source info
        detailed_sources.append({
            "source_id": i + 1,
            "url": source_url,
            "relevance_score": float(score),
            "chunk_id": doc.metadata.get("chunk_id", 0),
            "title": doc.metadata.get("title", ""),
            "preview": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
            "full_content": chunk_content
        })

    # Generate prompt
    context = "\n".join(context_parts)
    prompt = f"""Based on the following information from web pages, please answer this question: {query}

Information:
{context[:3000]}

Please provide a clear, comprehensive answer based only on the information provided above. If the information doesn't fully answer the question, mention what's missing."""

    # Generate response with proper LLM mode
    log_func(f"[LLM] Generating answer using {llm_mode}")
    try:
        answer = generate_response(prompt, max_length=500, mode=llm_mode, api_key=api_key)
        answer = answer.strip()
        
        # Remove "Answer:" prefix if present
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
            
    except Exception as e:
        log_func(f"[ERROR] LLM generation failed: {e}")
        answer = f"Found relevant content but could not generate answer: {str(e)}"
    
    return answer, detailed_sources, sources, unique_sources

def run_rag_pipeline(
    query: str,
    url: str,
    top_k: int = 5,
    force_rebuild: bool = False,
    embedding_model: str = "bge-base",
    max_pages: int = 30,
    max_depth: int = 2,
    llm_mode: str = "transformers",
    api_key: Optional[str] = None,
    log_func=None
) -> Dict:
    """
    Run the complete RAG pipeline - now more modular
    
    Args:
        query: The user's question
        url: The website to search
        top_k: Number of relevant chunks to retrieve
        force_rebuild: Whether to force rebuilding the index
        embedding_model: Which embedding model to use
        max_pages: Maximum pages to crawl
        max_depth: Maximum crawl depth
        llm_mode: Which LLM to use ('local', 'transformers', 'gemini', 'openai')
        api_key: API key for external LLMs
        log_func: Logging function
    """
    if log_func is None:
        log_func = print

    try:
        # Step 1: Check cache status
        cache_valid, vector_store, cache_meta = check_cache_and_load(url, embedding_model, log_func)
        
        # Step 2: Decide whether to use cache or crawl
        if cache_valid and not force_rebuild and vector_store:
            # FAST PATH: Use cached vector store
            pass  # vector_store already loaded
        else:
            # SLOW PATH: Need to crawl and build vector store
            if force_rebuild:
                log_func(f"[INFO] Force refresh requested - will rebuild cache")
            else:
                log_func(f"[INFO] No valid cache found - will crawl website")
            
            vector_store, cache_meta = crawl_and_index(url, max_pages, max_depth, embedding_model, log_func)
            
            if not vector_store:
                return {
                    "answer": "Failed to build search index.",
                    "sources": [],
                    "detailed_sources": [],
                    "metadata": {"error": "Failed to crawl or index website"}
                }

        # Step 3: Search and generate answer
        answer, detailed_sources, sources, unique_sources = search_and_generate(
            vector_store, query, top_k, llm_mode, api_key, log_func
        )
        
        # Determine which LLM was actually used for display
        llm_display_name = {
            'local': 'TinyLlama (Local)',
            'transformers': 'Mistral 7B (Local)',
            'gemini': 'Gemini API',
            'openai': 'OpenAI API'
        }.get(llm_mode, llm_mode)

        # Step 4: Return results with complete metadata
        return {
            "answer": answer,
            "sources": list(unique_sources),
            "detailed_sources": detailed_sources,
            "metadata": {
                "total_pages_crawled": cache_meta.get('doc_count', 0),
                "chunks_found": cache_meta.get('chunk_count', 0),
                "base_url": url,
                "unique_sources": len(unique_sources),
                "top_k_results": len(detailed_sources),
                "context_length": len("\n".join([d["full_content"] for d in detailed_sources])),
                "cache_used": cache_valid and not force_rebuild,
                "cache_age": str(datetime.now() - datetime.fromisoformat(cache_meta['timestamp'])) if cache_meta else None,
                "embedding_model": embedding_model,
                "crawl_performed": not (cache_valid and not force_rebuild),
                "llm_used": llm_display_name,
                "llm_mode": llm_mode
            }
        }
        
    except Exception as e:
        log_func(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "answer": f"An error occurred: {str(e)}",
            "sources": [],
            "detailed_sources": [],
            "metadata": {"error": str(e)}
        }