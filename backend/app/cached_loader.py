from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Set
import os
import json
import hashlib
from datetime import datetime
import requests
import time

class CachedWebLoader:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.raw_dir = os.path.join(cache_dir, "raw")
        self.vector_dir = os.path.join(cache_dir, "vectors")
        self.visited_urls: Set[str] = set()
        
        # Create cache directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.vector_dir, exist_ok=True)

    def _get_cache_path(self, url: str) -> str:
        """Generate cache path for URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.raw_dir, f"{url_hash}.json")

    async def crawl_and_load(
        self,
        base_url: str,
        max_depth: int = 2,
        max_pages: int = 30,
        force_reload: bool = False
    ) -> List[Dict]:
        """Crawl website and load content with caching"""
        cache_path = self._get_cache_path(base_url)
        
        # Check cache first
        if not force_reload and os.path.exists(cache_path):
            print(f"[CACHE] Loading content from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)

        print(f"[CRAWLER] Starting fresh crawl of {base_url}")
        self.visited_urls.clear()
        documents = []
        domain = urlparse(base_url).netloc
        
        async def crawl_url(url: str, depth: int):
            if (
                url in self.visited_urls or 
                len(self.visited_urls) >= max_pages or 
                depth > max_depth
            ):
                return
                
            self.visited_urls.add(url)
            print(f"[CRAWLER] Depth {depth}: {url}")
            
            try:
                response = requests.get(
                    url,
                    headers={'User-Agent': os.getenv('USER_AGENT', 'RAG-Bot/1.0')},
                    timeout=10
                )
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for tag in ['script', 'style', 'nav', 'footer']:
                    for element in soup.find_all(tag):
                        element.decompose()
                
                # Extract main content
                content = soup.get_text(separator='\n', strip=True)
                
                # Store document
                document = {
                    "url": url,
                    "content": content,
                    "metadata": {
                        "source": url,
                        "title": soup.title.string if soup.title else "",
                        "depth": depth,
                        "crawl_time": datetime.now().isoformat()
                    }
                }
                documents.append(document)
                
                # Find links if not at max depth
                if depth < max_depth:
                    for link in soup.find_all('a', href=True):
                        href = urljoin(url, link['href'])
                        parsed = urlparse(href)
                        if (
                            parsed.netloc == domain and
                            '#' not in href and
                            'mailto:' not in href and
                            href not in self.visited_urls
                        ):
                            await crawl_url(href, depth + 1)
                            
                time.sleep(0.5)  # Be polite
                
            except Exception as e:
                print(f"[ERROR] Failed to crawl {url}: {e}")
        
        # Start crawling from base URL
        await crawl_url(base_url, 0)
        
        # Save to cache if documents were found
        if documents:
            with open(cache_path, 'w') as f:
                json.dump(documents, f)
            print(f"[CACHE] Saved {len(documents)} documents to cache")
        
        return documents

    def build_vector_store(
        self,
        documents: List[Dict],
        embedding_model
    ) -> Optional[FAISS]:
        """Build vector store from documents"""
        if not documents:
            print("[ERROR] No documents to process")
            return None
            
        try:
            # Convert to LangChain format
            docs = [
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                ) for doc in documents
            ]
            
            # Create vector store
            vector_store = FAISS.from_documents(docs, embedding_model)
            print(f"[INDEX] Created vector store with {len(docs)} documents")
            return vector_store
            
        except Exception as e:
            print(f"[ERROR] Failed to build vector store: {str(e)}")
            return None