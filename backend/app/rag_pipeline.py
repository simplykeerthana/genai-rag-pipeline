# app/rag_pipeline.py
from typing import Dict, List, Optional, Tuple
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from app.llm_router import generate_response
from app.vector_store import get_embedding_model
from app.cached_loader import CachedWebLoader
import os
import json
from datetime import datetime

async def search_and_generate(
    query: str,
    vector_store: Optional[FAISS],
    top_k: int = 5,
    llm_mode: str = "transformers",
    api_key: Optional[str] = None,
    log_func=print
) -> Tuple[str, List[Dict], List[str], set]:
    """Search vector store and generate answer"""
    
    if not vector_store:
        log_func("[ERROR] No vector store available")
        return "Failed to search - no vector store available.", [], [], set()
    
    try:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        
        if not results:
            return "No relevant content found for your query.", [], [], set()

        log_func(f"[SEARCH] Found {len(results)} relevant chunks")
        
        context_parts = []
        sources = []
        detailed_sources = []
        unique_sources = set()

        for i, (doc, score) in enumerate(results):
            source_url = doc.metadata.get("source", "unknown")
            chunk_content = doc.page_content.strip()
            
            context_parts.append(f"[Source {i+1}: {source_url}]")
            context_parts.append(chunk_content)
            context_parts.append("")
            
            sources.append(source_url)
            unique_sources.add(source_url)
            
            detailed_sources.append({
                "source_id": i + 1,
                "url": source_url,
                "score": float(score),
                "title": doc.metadata.get("title", ""),
                "preview": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                "full_content": chunk_content
            })

        context = "\n".join(context_parts)
        prompt = f"""Based on the following information, please answer this question: {query}

Information:
{context[:3000]}

Please provide a clear and comprehensive answer based only on the information above. If the information doesn't fully answer the question, mention what's missing."""

        log_func(f"[LLM] Generating answer using {llm_mode}")
        answer = generate_response(prompt, max_length=500, mode=llm_mode, api_key=api_key)
        
        return answer.strip(), detailed_sources, sources, unique_sources

    except Exception as e:
        log_func(f"[ERROR] Search and generate failed: {e}")
        return "Failed to search and generate answer.", [], [], set()

async def run_rag_pipeline(
    query: str,
    url: str,
    force_rebuild: bool = False,
    embedding_model: str = "bge-base",
    log_func=print,
    max_pages: int = 30,
    max_depth: int = 2,
    top_k: int = 5,
    llm_mode: str = "transformers",
    api_key: Optional[str] = None
) -> Dict:
    """Run the complete RAG pipeline"""
    try:
        # Initialize loader
        loader = CachedWebLoader()
        
        # Load content
        documents = await loader.crawl_and_load(
            base_url=url,
            max_depth=max_depth,
            max_pages=max_pages,
            force_reload=force_rebuild
        )
        
        if not documents:
            return {
                "error": "No content found",
                "sources": []
            }
        
        # Get embedding model
        embedding_model_instance, _ = get_embedding_model(embedding_model)
        
        # Build vector store
        vector_store = loader.build_vector_store(
            documents=documents,
            embedding_model=embedding_model_instance
        )
        
        if not vector_store:
            return {
                "error": "Failed to build vector store",
                "sources": []
            }
        
        # Search and generate response
        answer, detailed_sources, sources, unique_sources = await search_and_generate(
            query=query,
            vector_store=vector_store,
            top_k=top_k,
            llm_mode=llm_mode,
            api_key=api_key,
            log_func=log_func
        )
        
        # Get LLM display name
        llm_display_name = {
            'local': 'TinyLlama (Local)',
            'transformers': 'Mistral 7B (Local)',
            'gemini': 'Gemini API',
            'openai': 'OpenAI API'
        }.get(llm_mode, llm_mode)

        return {
            "answer": answer,
            "sources": list(unique_sources),
            "detailed_sources": detailed_sources,
            "metadata": {
                "documents_processed": len(documents),
                "base_url": url,
                "unique_sources": len(unique_sources),
                "context_used": len("\n".join([d["full_content"] for d in detailed_sources])),
                "embedding_model": embedding_model,
                "llm_used": llm_display_name,
                "llm_mode": llm_mode,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        log_func(f"[ERROR] Pipeline failed: {str(e)}")
        return {"error": str(e), "sources": []}