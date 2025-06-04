#!/usr/bin/env python3
"""
Test the complete RAG system with GPU support
"""
import requests
import json
import time
import sys
import os

# Add backend to path for direct testing
sys.path.insert(0, '/Users/keerthana/gen-ai-OWASP-rag/backend')

def test_direct_import():
    """Test direct import of modules"""
    print("=" * 60)
    print("TESTING DIRECT IMPORTS")
    print("=" * 60)
    
    try:
        # Test LLM router
        from app import llm_router
        print("✅ LLM router imported successfully")
        
        # Check if model is on MPS
        if hasattr(llm_router, 'model') and llm_router.model:
            import torch
            device = next(llm_router.model.parameters()).device
            print(f"✅ Model loaded on: {device}")
            if str(device) == "mps:0":
                print("🎉 Using Apple Silicon GPU!")
        
        # Test vector store
        from app import vector_store
        print("✅ Vector store imported successfully")
        
        # Test RAG pipeline
        from app import rag_pipeline
        print("✅ RAG pipeline imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoint(base_url="http://localhost:8000"):
    """Test the FastAPI endpoints"""
    print("\n" + "=" * 60)
    print("TESTING API ENDPOINTS")
    print("=" * 60)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        print("Make sure the FastAPI server is running: python main.py")
        return False
    
    # Test cache info
    try:
        response = requests.get(f"{base_url}/cache/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Cache info: {data.get('cache_count', 0)} cached sites")
    except Exception as e:
        print(f"❌ Cache info failed: {e}")
    
    return True

def test_rag_query(base_url="http://localhost:8000"):
    """Test a RAG query"""
    print("\n" + "=" * 60)
    print("TESTING RAG QUERY")
    print("=" * 60)
    
    # Test query
    test_data = {
        "query": "What are the main security vulnerabilities?",
        "url": "https://owasp.org/www-project-top-ten/",
        "force_refresh": False,
        "embedding_model": "bge-base",
        "max_pages": 5,
        "max_depth": 1,
        "top_k": 3
    }
    
    print(f"Query: {test_data['query']}")
    print(f"URL: {test_data['url']}")
    print(f"Settings: max_pages={test_data['max_pages']}, max_depth={test_data['max_depth']}")
    
    try:
        print("\nSending request...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/ask",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Request successful (took {elapsed:.1f}s)")
            print(f"\nAnswer preview: {result.get('answer', '')[:200]}...")
            
            if 'metadata' in result:
                meta = result['metadata']
                print(f"\nMetadata:")
                print(f"  - Pages crawled: {meta.get('total_pages_crawled', 0)}")
                print(f"  - Chunks created: {meta.get('chunks_found', 0)}")
                print(f"  - Unique sources: {meta.get('unique_sources', 0)}")
                print(f"  - Cache used: {meta.get('cache_used', False)}")
            
            if 'sources' in result:
                print(f"\nSources found: {len(result['sources'])}")
                for src in result['sources'][:3]:
                    print(f"  - {src}")
            
            # Show some logs
            if 'logs' in result and result['logs']:
                print(f"\nRecent logs:")
                for log in result['logs'][-5:]:
                    print(f"  [{log['level']}] {log['message']}")
            
            return True
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error making request: {e}")
        return False

def main():
    """Run all tests"""
    print("RAG SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Test 1: Direct imports
    import_ok = test_direct_import()
    
    if not import_ok:
        print("\n⚠️  Direct imports failed. Check your Python path and dependencies.")
        return
    
    # Test 2: API endpoints
    print("\n" + "=" * 60)
    print("Is the FastAPI server running? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        api_ok = test_api_endpoint()
        
        if api_ok:
            # Test 3: RAG query
            print("\nDo you want to test a RAG query? This will crawl a website. (y/n): ", end="")
            response = input().strip().lower()
            
            if response == 'y':
                test_rag_query()
    else:
        print("\nTo start the server, run in another terminal:")
        print("cd /Users/keerthana/gen-ai-OWASP-rag/backend")
        print("python main.py")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()