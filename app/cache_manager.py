#!/usr/bin/env python3
"""
Cache management utilities for the RAG vector store
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.vector_store import clear_cache, get_cache_info
from datetime import datetime
import time

def list_caches():
    """List all cached vector stores"""
    cache_info = get_cache_info()
    
    if not cache_info:
        print("No cached vector stores found.")
        return
    
    print(f"\n{'='*80}")
    print("CACHED VECTOR STORES")
    print(f"{'='*80}")
    
    for info in cache_info:
        print(f"\nURL: {info['url']}")
        print(f"  Cached: {info['timestamp']}")
        print(f"  Age: {info['age_hours']:.1f} hours")
        print(f"  Documents: {info['doc_count']}")
        print(f"  Chunks: {info['chunk_count']}")
        print(f"  Model: {info.get('model', 'unknown')}")
        
    print(f"\nTotal: {len(cache_info)} cached sites\n")

def clear_old_caches(max_age_hours=24):
    """Clear caches older than specified hours"""
    cache_info = get_cache_info()
    cleared = 0
    failed = 0
    
    for info in cache_info:
        if info['age_hours'] > max_age_hours:
            try:
                clear_cache(info['url'])
                cleared += 1
                print(f"Cleared old cache for: {info['url']}")
            except Exception as e:
                failed += 1
                print(f"Failed to clear cache for {info['url']}: {e}")
    
    print(f"\nCleared {cleared} old caches (older than {max_age_hours} hours)")
    if failed > 0:
        print(f"Failed to clear {failed} caches due to permissions")

def clear_specific_url(url):
    """Clear cache for a specific URL"""
    try:
        clear_cache(url)
        print(f"Cleared cache for: {url}")
    except Exception as e:
        print(f"Error clearing cache for {url}: {e}")
        print("Cache files may have been renamed with .old suffix instead")

def clear_all():
    """Clear all caches"""
    confirm = input("Are you sure you want to clear ALL caches? (yes/no): ")
    if confirm.lower() == 'yes':
        try:
            clear_cache()
            print("All caches cleared.")
        except Exception as e:
            print(f"Error clearing caches: {e}")
            print("Some cache files may have been renamed with .old suffix instead")
    else:
        print("Cancelled.")

def show_cache_stats():
    """Show cache statistics"""
    cache_info = get_cache_info()
    
    if not cache_info:
        print("No cached vector stores found.")
        return
    
    total_chunks = sum(info['chunk_count'] for info in cache_info)
    total_docs = sum(info['doc_count'] for info in cache_info)
    avg_age = sum(info['age_hours'] for info in cache_info) / len(cache_info)
    
    print(f"\n{'='*50}")
    print("CACHE STATISTICS")
    print(f"{'='*50}")
    print(f"Total cached sites: {len(cache_info)}")
    print(f"Total documents: {total_docs}")
    print(f"Total chunks: {total_chunks}")
    print(f"Average cache age: {avg_age:.1f} hours")
    print(f"Oldest cache: {max(info['age_hours'] for info in cache_info):.1f} hours")
    print(f"Newest cache: {min(info['age_hours'] for info in cache_info):.1f} hours")
    
    # Show models used
    models = {}
    for info in cache_info:
        model = info.get('model', 'unknown')
        models[model] = models.get(model, 0) + 1
    
    print("\nModels used:")
    for model, count in models.items():
        print(f"  {model}: {count} sites")

def clean_old_files():
    """Clean up old renamed cache files"""
    from app.vector_store import CACHE_DIR
    
    if not os.path.exists(CACHE_DIR):
        print("Cache directory does not exist")
        return
    
    old_files = []
    for file in os.listdir(CACHE_DIR):
        if '.old_' in file:
            old_files.append(file)
    
    if not old_files:
        print("No old cache files found")
        return
    
    print(f"Found {len(old_files)} old cache files:")
    for f in old_files:
        print(f"  {f}")
    
    confirm = input("\nDo you want to delete these old files? (yes/no): ")
    if confirm.lower() == 'yes':
        removed = 0
        for f in old_files:
            try:
                os.remove(os.path.join(CACHE_DIR, f))
                removed += 1
            except Exception as e:
                print(f"Could not remove {f}: {e}")
        print(f"Removed {removed} old files")
    else:
        print("Cancelled")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Vector Store Cache Manager")
    parser.add_argument('command', choices=['list', 'clear', 'clear-old', 'clear-url', 'stats', 'clean-old'],
                        help='Command to execute')
    parser.add_argument('--url', help='URL for clear-url command')
    parser.add_argument('--max-age', type=int, default=24,
                        help='Maximum age in hours for clear-old command (default: 24)')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'list':
            list_caches()
        elif args.command == 'clear':
            clear_all()
        elif args.command == 'clear-old':
            clear_old_caches(args.max_age)
        elif args.command == 'clear-url':
            if not args.url:
                print("Error: --url required for clear-url command")
                sys.exit(1)
            clear_specific_url(args.url)
        elif args.command == 'stats':
            show_cache_stats()
        elif args.command == 'clean-old':
            clean_old_files()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()