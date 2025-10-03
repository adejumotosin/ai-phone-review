# app.py
import os
import re
import json
import time
import math
import hashlib
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import google.generativeai as genai
# Note: Renaming for consistency with your original approach, 
# but I recommend using 'urllib.parse.quote' instead of 'requests.utils.quote'
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, quote 

# -----------------------------
# Configure Gemini (Google) SDK
# -----------------------------
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Missing Gemini API key. Please set GEMINI_API_KEY in secrets or env vars.")
    st.stop()

genai.configure(api_key=api_key)
# create a model handle
model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Helpers: safe json loader
# -----------------------------
def safe_load_json(text: str):
    """Try to load JSON; if fails, extract first {...} block and try again."""
    if not text: # Added check for None/empty string
        return None
    try:
        # Added stripping of markdown fences for better parsing
        text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{(?:.|\n)*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

# -----------------------------
# Resolve GSMArena product + review URL
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="🔎 Searching GSMArena...")
def resolve_gsmarena_url(product_name: str):
    """Search GSMArena and return (product_url, review_url)."""
    try:
        # RESTORED TO ORIGINAL WORKING LOGIC
        query = str(product_name).strip() # Ensure it's a string and stripped
        if not query:
             return None, None
             
        # Using urllib.parse.quote as a standard
        search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        # This will now use the correct URL with requests.get()
        r = requests.get(search_url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # common selector for results
        link = soup.select_one(".makers a") or soup.select_one(".makers li a")
        if not link:
            # last resort: any php link
            link = soup.select_one("a[href*='.php']")
        if not link or not link.has_attr("href"):
            return None, None

        product_href = link["href"]
        product_url = "https://www.gsmarena.com/" + product_href
        # Build review URL reliably
        review_url = build_review_url(product_url)
        return product_url, review_url
    except Exception as e:
        # Use st.error here since it's a critical failure, and the traceback is already shown
        st.error(f"⚠️ GSMArena search failed: {type(e).__name__} - {e}") 
        return None, None

def build_review_url(product_url: str) -> str:
    """Convert product URL to reviews URL: <base>-reviews-<id>.php"""
    if not product_url or not product_url.endswith(".php"):
        return None
    try:
        base, phone_id_part = product_url.rsplit("-", 1)
        phone_id = phone_id_part.replace(".php", "")
        return f"{base}-reviews-{phone_id}.php"
    except Exception:
        return None

# -----------------------------
# Specs scraper
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="📊 Fetching specs...")
def fetch_gsmarena_specs(url: str):
    # (Rest of the function is unchanged and correct)
    # ...
    specs = {}
    key_map = {
        "Display": ["Display", "Screen", "Size"],
        "Processor": ["Chipset", "CPU", "Processor", "SoC"],
        "RAM": ["Internal", "Memory", "RAM"],
        "Storage": ["Internal", "Storage", "Memory"],
        "Camera": ["Main Camera", "Triple", "Quad", "Dual", "Camera"],
        "Battery": ["Battery"],
        "OS": ["OS", "Android", "iOS"]
    }
    if not url:
        return {k: "Not specified" for k in key_map.keys()}
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Try multiple containers
        rows = soup.select(".article-info table tr") or soup.select("#specs-list table tr") or soup.select("table.specs tr")
        for row in rows:
            th = row.find("td", class_="ttl") or row.find("th") or row.find("td", class_="spec-title")
            td = row.find("td", class_="nfo") or (row.find_all("td")[-1] if row.find_all("td") else None)
            if not th or not td:
                continue
            key = th.get_text(strip=True)
            val = td.get_text(" ", strip=True)
            for field, keywords in key_map.items():
                if any(k.lower() in key.lower() for k in keywords):
                    if field in ("RAM", "Storage"):
                        matches = re.findall(r"(\d+)\s*GB", val, flags=re.IGNORECASE)
                        if matches:
                            # heuristics: first is RAM, last is Storage if both present
                            if field == "RAM":
                                specs["RAM"] = f"{matches[0]}GB RAM"
                                if len(matches) > 1 and "Storage" not in specs:
                                    specs["Storage"] = f"{matches[1]}GB Storage"
                            else:
                                specs["Storage"] = f"{matches[-1]}GB Storage"
                        else:
                            specs[field] = val
                    else:
                        specs[field] = val
                    break
    except Exception as e:
        st.warning(f"⚠️ GSMArena specs fetch failed: {e}")

    # Ensure presence of all keys
    for k in ["Display", "Processor", "RAM", "Storage", "Camera", "Battery", "OS"]:
        specs.setdefault(k, "Not specified")

    return specs
# -----------------------------
# Reviews scraper with better pagination detection
# -----------------------------
@st.cache_data(ttl=21600, show_spinner="💬 Fetching user reviews...")
def fetch_gsmarena_reviews(url: str, limit: int = 1000):
    # (Rest of the function is largely unchanged)
    # ...
    reviews = []
    if not url:
        return reviews

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    page_url = url
    visited_urls = set()
    page_count = 0
    max_pages = 50 

    try:
        while page_url and len(reviews) < limit and page_count < max_pages:
            if page_url in visited_urls:
                st.warning(f"🔄 Breaking pagination loop - already visited: {page_url}")
                break
            
            visited_urls.add(page_url)
            page_count += 1
            
            st.info(f"📖 Scraping page {page_count}: {len(reviews)} reviews so far...")
            
            time.sleep(2)  
            r = requests.get(page_url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # ... (Review block extraction logic is unchanged and correct)
            review_blocks = []
            
            selectors_to_try = [
                ".opin",  
                ".user-opinion", 
                ".uopin",
                ".user-review",
                ".review-item",
                ".review-content",
                ".opinion",
                "div[id*='opin']", 
                "div.thread-item",
                ".thread .post",
                ".user-thread .post"
            ]
            
            for selector in selectors_to_try:
                blocks = soup.select(selector)
                if blocks:
                    review_blocks.extend(blocks)
                    break 
            
            if not review_blocks:
                all_divs = soup.find_all('div')
                for div in all_divs:
                    text = div.get_text(strip=True)
                    if (50 < len(text) < 2000 and 
                        any(word in text.lower() for word in ['phone', 'battery', 'camera', 'display', 'good', 'bad', 'excellent', 'terrible']) and
                        not any(skip in text.lower() for skip in ['gsmarena', 'admin', 'moderator', 'advertisement'])):
                        review_blocks.append(div)

            new_reviews_count = 0
            for block in review_blocks:
                text = block.get_text(" ", strip=True)
                
                if (30 < len(text) < 1500 and 
                    not any(skip in text.lower() for skip in [
                        'gsmarena', 'admin', 'moderator', 'delete', 'report', 
                        'advertisement', 'sponsored', 'click here', 'visit our',
                        'terms of service', 'privacy policy'
                    ]) and
                    any(keyword in text.lower() for keyword in [
                        'phone', 'battery', 'camera', 'display', 'screen', 
                        'performance', 'android', 'ios', 'good', 'bad', 'love', 'hate',
                        'recommend', 'buy', 'excellent', 'terrible', 'amazing', 'awful'
                    ])):
                    
                    if text not in [r[:100] for r in reviews]:  
                        reviews.append(text)
                        new_reviews_count += 1
                        if len(reviews) >= limit:
                            break

            st.info(f"✅ Found {new_reviews_count} new reviews on page {page_count} (total: {len(reviews)})")
            
            if new_reviews_count == 0:
                st.warning(f"⚠️ No new reviews found on page {page_count}, stopping pagination")
                break

            # ... (Pagination detection logic is unchanged and correct)
            next_link = None
            
            next_candidates = [
                soup.find("a", string=re.compile(r"next", re.I)),
                soup.find("a", {"title": re.compile(r"next", re.I)}),
                soup.select_one("a.pages-next"),
                soup.find("a", attrs={"rel": "next"}),
                soup.select_one(".pagination a[href*='page=']:last-child"),
            ]
            
            for candidate in next_candidates:
                if candidate and candidate.has_attr("href"):
                    next_link = candidate
                    break
            
            if not next_link:
                current_page = page_count
                page_links = soup.find_all("a", href=re.compile(r"page=\d+"))
                for link in page_links:
                    href = link["href"]
                    match = re.search(r"page=(\d+)", href)
                    if match and int(match.group(1)) == current_page + 1:
                        next_link = link
                        break
            
            if not next_link and "page=" in page_url:
                try:
                    parsed = urlparse(page_url)
                    query_params = parse_qs(parsed.query)
                    
                    current_page_num = 1
                    if 'page' in query_params:
                        current_page_num = int(query_params['page'][0])
                    
                    query_params['page'] = [str(current_page_num + 1)]
                    new_query = urlencode(query_params, doseq=True)
                    next_page_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"
                    
                    if next_page_url != page_url:
                        page_url = next_page_url
                        continue
                except Exception as e:
                    st.warning(f"Failed to construct next page URL: {e}")
            
            elif not next_link and "page=" not in page_url:
                separator = "&" if "?" in page_url else "?"
                next_page_url = f"{page_url}{separator}page=2"
                page_url = next_page_url
                continue
            
            if next_link:
                href = next_link["href"]
                if href.startswith("http"):
                    page_url = href
                else:
                    # Minor improvement: use the site's base URL for safety
                    page_url = urljoin("https://www.gsmarena.com/", href) 
                st.info(f"🔗 Found next page: {page_url}")
            else:
                st.info("🏁 No more pagination links found")
                break

    except Exception as e:
        st.error(f"⚠️ GSMArena reviews fetch failed: {e}")
        import traceback
        st.error(traceback.format_exc())

    st.success(f"🎉 Successfully collected {len(reviews)} reviews from {page_count} pages")
    return reviews[:limit]

# (The rest of the file: Prompt Builders, Summarization Pipeline, and Streamlit UI remains the same)
