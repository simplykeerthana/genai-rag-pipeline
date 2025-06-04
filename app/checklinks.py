#!/usr/bin/env python3
"""
Check what links are on a webpage
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def check_links(url):
    """Check all links on a webpage"""
    print(f"Checking links on {url}")
    print("="*60)
    
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get domain
        domain = urlparse(url).netloc
        
        all_links = []
        internal_links = []
        external_links = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '').strip()
            if not href or href.startswith('#'):
                continue
            
            # Make absolute URL
            full_url = urljoin(url, href)
            all_links.append((href, full_url, link.get_text(strip=True)))
            
            # Check if internal
            parsed = urlparse(full_url)
            if parsed.netloc.replace('www.', '') == domain.replace('www.', ''):
                internal_links.append((full_url, link.get_text(strip=True)))
            else:
                external_links.append((full_url, link.get_text(strip=True)))
        
        print(f"\nTotal links found: {len(all_links)}")
        print(f"Internal links: {len(internal_links)}")
        print(f"External links: {len(external_links)}")
        
        print("\n\nINTERNAL LINKS:")
        print("-"*60)
        for i, (link, text) in enumerate(internal_links[:20]):  # First 20
            print(f"{i+1}. {link}")
            if text:
                print(f"   Text: {text[:50]}...")
        
        if len(internal_links) > 20:
            print(f"\n... and {len(internal_links) - 20} more internal links")
        
        # Check for common patterns
        print("\n\nLINK PATTERNS:")
        print("-"*60)
        patterns = {}
        for link, _ in internal_links:
            parsed = urlparse(link)
            path = parsed.path
            if '/' in path:
                pattern = path.split('/')[1] if len(path.split('/')) > 1 else 'root'
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"{pattern}: {count} links")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://genai.owasp.org"
    check_links(url)