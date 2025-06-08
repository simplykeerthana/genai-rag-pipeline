# loaders.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import os
import json
import hashlib
import time
import re
import unicodedata
from typing import List, Dict, Optional, Tuple

CACHE_DIR = "cache"
URL_CACHE_PATH = os.path.join(CACHE_DIR, "url_cache.json")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

if os.path.exists(URL_CACHE_PATH):
    with open(URL_CACHE_PATH, "r") as f:
        url_cache = json.load(f)
else:
    url_cache = {}

def get_url_hash(url):
    return hashlib.md5(url.encode()).hexdigest()

def normalize_url(url):
    url = url.split('#')[0].rstrip('/')
    return url

def is_valid_internal_link(url, domain):
    try:
        parsed = urlparse(url)
        return (parsed.netloc.replace('www.', '') == domain.replace('www.', '') and
                parsed.scheme in ["http", "https"] and
                not any(url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.doc', '.docx', '.xls', '.xlsx']))
    except:
        return False

def clean_text(text):
    """Clean text while preserving important content and structure"""
    if not text:
        return ""
    
    # Don't normalize Unicode - preserve original characters
    # This was causing issues with non-ASCII content
    # text = unicodedata.normalize('NFKD', text)
    
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    
    # Keep all non-empty lines (don't filter by length)
    # The previous filter was removing important short lines
    lines = [line for line in lines if line]
    
    return '\n'.join(lines).strip()

def extract_clean_content(soup):
    """Extract and clean content from BeautifulSoup object"""
    # Remove script, style, and other non-content tags
    for tag in soup(['script', 'style', 'noscript', 'iframe', 'meta', 'link']):
        tag.decompose()
    
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
        comment.extract()
    
    # Try to find main content areas
    content = None
    
    # Look for common content containers
    content_selectors = [
        'main',
        'article',
        '[role="main"]',
        '.main-content',
        '#main-content',
        '.content',
        '#content',
        '.post',
        '.entry-content',
        '.page-content'
    ]
    
    for selector in content_selectors:
        if selector.startswith('.') or selector.startswith('#') or selector.startswith('['):
            content = soup.select_one(selector)
        else:
            content = soup.find(selector)
        
        if content:
            break
    
    # If no specific content area found, use body
    if not content:
        content = soup.find('body')
    
    # If still no content, try to get all text
    if not content:
        content = soup
    
    # Extract text with better preservation of structure
    # Use separator to maintain some structure
    text = content.get_text(separator=' ', strip=True)
    
    # Clean the extracted text
    cleaned_text = clean_text(text)
    
    # If we got very little content, try a different approach
    if len(cleaned_text) < 100:
        # Get all paragraph and heading tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th'])
        text_parts = []
        
        for elem in text_elements:
            elem_text = elem.get_text(strip=True)
            if elem_text:
                text_parts.append(elem_text)
        
        alternative_text = ' '.join(text_parts)
        if len(alternative_text) > len(cleaned_text):
            cleaned_text = clean_text(alternative_text)
    
    return cleaned_text

def extract_metadata(soup, url):
    """Extract metadata from the page"""
    metadata = {
        'title': '',
        'description': '',
        'keywords': []
    }
    
    # Try to get title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text(strip=True)
    
    # Try to get meta description
    desc_tag = soup.find('meta', attrs={'name': 'description'})
    if desc_tag:
        metadata['description'] = desc_tag.get('content', '')
    
    # Try to get keywords
    keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
    if keywords_tag:
        keywords = keywords_tag.get('content', '')
        metadata['keywords'] = [k.strip() for k in keywords.split(',') if k.strip()]
    
    return metadata

def save_content(domain, url, html, text, metadata=None):
    """Save content and metadata to cache"""
    try:
        dir_path = os.path.join(CACHE_DIR, domain.replace(':', '_'))
        os.makedirs(dir_path, exist_ok=True)
        url_hash = get_url_hash(url)
        
        # Save text content
        with open(os.path.join(dir_path, f"{url_hash}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        
        # Save HTML
        with open(os.path.join(dir_path, f"{url_hash}.html"), "w", encoding="utf-8") as f:
            f.write(html)
        
        # Save metadata if provided
        if metadata:
            with open(os.path.join(dir_path, f"{url_hash}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return text
    except Exception as e:
        print(f"[ERROR] Saving failed for {url}: {e}")
        return ""

def load_cached_content(domain, url):
    """Load cached content and metadata"""
    try:
        dir_path = os.path.join(CACHE_DIR, domain.replace(':', '_'))
        url_hash = get_url_hash(url)
        
        # Load text content
        with open(os.path.join(dir_path, f"{url_hash}.txt"), "r", encoding="utf-8") as f:
            text = f.read()
        
        # Try to load metadata
        metadata = {}
        meta_path = os.path.join(dir_path, f"{url_hash}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        
        return text, metadata
    except:
        return None, None

def crawl_website(base_url, max_pages=30, max_depth=2, log_func=print):
    """Crawl website and extract content"""
    base_url = normalize_url(base_url)
    if not base_url.startswith('http'):
        base_url = 'https://' + base_url
    
    domain = urlparse(base_url).netloc
    visited = set()
    queue = deque([(base_url, 0)])
    results = []

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })

    while queue and len(results) < max_pages:
        url, depth = queue.popleft()
        
        if url in visited or depth > max_depth:
            continue
            
        visited.add(url)
        url = normalize_url(url)

        # Check cache first
        cached_content, cached_metadata = load_cached_content(domain, url)
        if cached_content:
            log_func(f"[CACHE] Loaded from cache: {url}")
            results.append({
                "source_url": url,
                "content": cached_content,
                "depth": depth,
                "title": cached_metadata.get('title', '') if cached_metadata else '',
                "metadata": cached_metadata or {}
            })
            continue

        try:
            log_func(f"[CRAWL] Fetching: {url} (depth: {depth})")
            response = session.get(url, timeout=10, allow_redirects=True)
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                log_func(f"[SKIP] Non-HTML content: {url} ({content_type})")
                continue
            
            if response.status_code != 200:
                log_func(f"[ERROR] HTTP {response.status_code} for {url}")
                continue
            
            # Ensure we have the correct encoding
            response.encoding = response.apparent_encoding or 'utf-8'
            html = response.text
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract metadata
            metadata = extract_metadata(soup, url)
            
            # Extract content
            text = extract_clean_content(soup)
            
            # Log content extraction results
            log_func(f"[EXTRACT] Extracted {len(text)} characters from {url}")
            
            if len(text) < 50:
                log_func(f"[WARNING] Very little content extracted from {url}")
                # Don't skip - even small content might be valuable
            
            # Save to cache
            final_text = save_content(domain, url, html, text, metadata)
            
            # Add to results
            results.append({
                "source_url": url,
                "content": final_text,
                "depth": depth,
                "title": metadata.get('title', ''),
                "metadata": metadata
            })
            
            # Find and queue internal links
            if depth < max_depth:
                links_found = 0
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(url, link['href'])
                    if is_valid_internal_link(full_url, domain):
                        full_url = normalize_url(full_url)
                        if full_url not in visited:
                            queue.append((full_url, depth + 1))
                            links_found += 1
                
                log_func(f"[LINKS] Found {links_found} internal links on {url}")
            
            # Small delay to be polite
            time.sleep(0.5)
            
        except requests.exceptions.Timeout:
            log_func(f"[ERROR] Timeout while fetching {url}")
        except requests.exceptions.RequestException as e:
            log_func(f"[ERROR] Request failed for {url}: {e}")
        except Exception as e:
            log_func(f"[ERROR] Unexpected error for {url}: {e}")
            import traceback
            log_func(traceback.format_exc())

    # Save URL cache
    with open(URL_CACHE_PATH, "w") as f:
        json.dump(url_cache, f)

    log_func(f"[COMPLETE] Crawled {len(results)} pages from {base_url}")
    return results

def test_extraction(url):
    """Test function to debug content extraction"""
    print(f"Testing content extraction for: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.encoding = response.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract content
        content = extract_clean_content(soup)
        metadata = extract_metadata(soup, url)
        
        print(f"\nTitle: {metadata.get('title', 'N/A')}")
        print(f"Description: {metadata.get('description', 'N/A')}")
        print(f"Content length: {len(content)} characters")
        print(f"\nFirst 500 characters of content:")
        print("-" * 50)
        print(content[:500])
        print("-" * 50)
        
        return content, metadata
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None