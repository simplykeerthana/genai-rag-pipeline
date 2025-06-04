#!/usr/bin/env python3
"""
Compare different embedding models for quality and speed
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.vector_store import EMBEDDING_MODELS, get_embedding_model
from app.rag_pipeline import run_rag_pipeline
import numpy as np

def test_embedding_speed(model_key: str, texts: list) -> float:
    """Test embedding speed for a model"""
    print(f"\nTesting {model_key}...")
    
    # Load model
    start = time.time()
    embedding_model, model_info = get_embedding_model(model_key)
    load_time = time.time() - start
    print(f"  Model load time: {load_time:.2f}s")
    
    # Test embedding speed
    start = time.time()
    embeddings = embedding_model.embed_documents(texts)
    embed_time = time.time() - start
    
    print(f"  Embedding time: {embed_time:.2f}s ({len(texts)} texts)")
    print(f"  Speed: {len(texts)/embed_time:.1f} texts/second")
    print(f"  Embedding dimension: {len(embeddings[0])}")
    
    return embed_time

def test_retrieval_quality(url: str, test_queries: list):
    """Test retrieval quality with different models"""
    print(f"\nTesting retrieval quality on {url}")
    print("="*60)
    
    results = {}
    
    for model_key in ["minilm", "bge-small", "bge-base", "e5-base"]:
        print(f"\n\nTesting model: {model_key}")
        print("-"*40)
        
        model_results = []
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            try:
                # Run pipeline with specific model
                result = run_rag_pipeline(
                    query=query,
                    url=url,
                    top_k=3,
                    embedding_model=model_key,
                    log_func=lambda x: None  # Suppress logs
                )
                
                # Show top result
                if result['sources']:
                    top_source = result['sources'][0]
                    print(f"  Top result (relevance: {top_source['relevance_score']:.2%}):")
                    print(f"  {top_source['preview'][:100]}...")
                else:
                    print("  No results found")
                
                model_results.append({
                    'query': query,
                    'found': len(result['sources']) > 0,
                    'top_score': result['sources'][0]['relevance_score'] if result['sources'] else 0
                })
                
            except Exception as e:
                print(f"  Error: {e}")
                model_results.append({
                    'query': query,
                    'found': False,
                    'top_score': 0
                })
        
        results[model_key] = model_results

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for model_key, model_results in results.items():
        success_rate = sum(1 for r in model_results if r['found']) / len(model_results)
        avg_score = np.mean([r['top_score'] for r in model_results if r['found']] or [0])
        
        print(f"\n{model_key}:")
        print(f"  Success rate: {success_rate:.0%}")
        print(f"  Average relevance: {avg_score:.2%}")

def benchmark_all_models():
    """Benchmark all available embedding models"""
    print("EMBEDDING MODEL BENCHMARK")
    print("="*60)
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "RAG combines retrieval with generation for better results.",
        "Vector embeddings represent text in high-dimensional space.",
        "OWASP provides security guidelines for applications."
    ] * 10  # 50 texts total
    
    results = {}
    
    for model_key in EMBEDDING_MODELS.keys():
        embed_time = test_embedding_speed(model_key, test_texts)
        results[model_key] = {
            'time': embed_time,
            'speed': len(test_texts) / embed_time,
            'info': EMBEDDING_MODELS[model_key]
        }
    
    # Summary table
    print("\n\nSUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<15} {'Dimension':<10} {'Speed (texts/s)':<15} {'Description':<40}")
    print("-"*80)
    
    # Sort by speed
    sorted_results = sorted(results.items(), key=lambda x: x[1]['speed'], reverse=True)
    
    for model_key, result in sorted_results:
        info = result['info']
        print(f"{model_key:<15} {info['dimension']:<10} {result['speed']:<15.1f} {info['description']:<40}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument('--benchmark', action='store_true', 
                        help='Run speed benchmark on all models')
    parser.add_argument('--quality', action='store_true',
                        help='Test retrieval quality')
    parser.add_argument('--url', default='https://genai.owasp.org',
                        help='URL to test retrieval quality on')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_all_models()
    
    if args.quality:
        test_queries = [
            "What is OWASP GenAI?",
            "What are the security risks for AI?",
            "How to prevent prompt injection?",
            "Tell me about LLM security",
            "What is retrieval augmented generation?"
        ]
        test_retrieval_quality(args.url, test_queries)
    
    if not args.benchmark and not args.quality:
        print("Please specify --benchmark or --quality")
        parser.print_help()

if __name__ == "__main__":
    main()