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
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

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
    try:
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
# GSMArena Functions
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="🔎 Searching GSMArena...")
def resolve_gsmarena_url(product_name: str):
    """Search GSMArena and return (product_url, review_url)."""
    try:
        query = product_name.strip()
        search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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
        st.warning(f"⚠️ GSMArena search failed: {e}")
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

@st.cache_data(ttl=86400, show_spinner="📊 Fetching GSMArena specs...")
def fetch_gsmarena_specs(url: str):
    """Scrape key specs from GSMArena product page."""
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

@st.cache_data(ttl=21600, show_spinner="💬 Fetching GSMArena reviews...")
def fetch_gsmarena_reviews(url: str, limit: int = 1000):
    """
    Fetch reviews from GSMArena review pages, following pagination until limit reached.
    Returns list of review texts.
    """
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
    max_pages = 50  # Safety limit

    try:
        while page_url and len(reviews) < limit and page_count < max_pages:
            if page_url in visited_urls:
                st.warning(f"🔄 Breaking GSMArena pagination loop - already visited: {page_url}")
                break
            
            visited_urls.add(page_url)
            page_count += 1
            
            st.info(f"📖 Scraping GSMArena page {page_count}: {len(reviews)} reviews so far...")
            
            time.sleep(2)  # More polite delay
            r = requests.get(page_url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # More comprehensive selectors for review content
            review_blocks = []
            
            # Try different review selectors in order of preference
            selectors_to_try = [
                ".opin",  # Most common
                ".user-opinion", 
                ".uopin",
                ".user-review",
                ".review-item",
                ".review-content",
                ".opinion",
                "div[id*='opin']",  # IDs containing 'opin'
                "div.thread-item",
                ".thread .post",
                ".user-thread .post"
            ]
            
            for selector in selectors_to_try:
                blocks = soup.select(selector)
                if blocks:
                    review_blocks.extend(blocks)
                    break  # Use first successful selector
            
            # If no review blocks found, try fallback approach
            if not review_blocks:
                # Look for divs with specific patterns
                all_divs = soup.find_all('div')
                for div in all_divs:
                    # Check if div contains review-like content
                    text = div.get_text(strip=True)
                    if (50 < len(text) < 2000 and 
                        any(word in text.lower() for word in ['phone', 'battery', 'camera', 'display', 'good', 'bad', 'excellent', 'terrible']) and
                        not any(skip in text.lower() for skip in ['gsmarena', 'admin', 'moderator', 'advertisement'])):
                        review_blocks.append(div)

            # Extract text from found blocks
            new_reviews_count = 0
            for block in review_blocks:
                text = block.get_text(" ", strip=True)
                
                # Better filtering criteria
                if (30 < len(text) < 1500 and  # Reasonable length
                    not any(skip in text.lower() for skip in [
                        'gsmarena', 'admin', 'moderator', 'delete', 'report', 
                        'advertisement', 'sponsored', 'click here', 'visit our',
                        'terms of service', 'privacy policy'
                    ]) and
                    # Must contain phone-related keywords
                    any(keyword in text.lower() for keyword in [
                        'phone', 'battery', 'camera', 'display', 'screen', 
                        'performance', 'android', 'ios', 'good', 'bad', 'love', 'hate',
                        'recommend', 'buy', 'excellent', 'terrible', 'amazing', 'awful'
                    ])):
                    
                    # Avoid duplicates
                    if text not in [r[:100] for r in reviews]:  # Check first 100 chars for similarity
                        reviews.append(text)
                        new_reviews_count += 1
                        if len(reviews) >= limit:
                            break

            st.info(f"✅ Found {new_reviews_count} new GSMArena reviews on page {page_count} (total: {len(reviews)})")
            
            if new_reviews_count == 0:
                st.warning(f"⚠️ No new GSMArena reviews found on page {page_count}, stopping pagination")
                break

            # IMPROVED pagination detection
            next_link = None
            
            # Method 1: Look for specific next page links
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
            
            # Method 2: Look for numbered pagination
            if not next_link:
                current_page = page_count
                page_links = soup.find_all("a", href=re.compile(r"page=\d+"))
                for link in page_links:
                    href = link["href"]
                    match = re.search(r"page=(\d+)", href)
                    if match and int(match.group(1)) == current_page + 1:
                        next_link = link
                        break
            
            # Method 3: Try to construct next page URL manually
            if not next_link and "page=" in page_url:
                try:
                    # Parse current page number and increment
                    parsed = urlparse(page_url)
                    query_params = parse_qs(parsed.query)
                    
                    current_page_num = 1
                    if 'page' in query_params:
                        current_page_num = int(query_params['page'][0])
                    
                    # Construct next page URL
                    query_params['page'] = [str(current_page_num + 1)]
                    new_query = urlencode(query_params, doseq=True)
                    next_page_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"
                    
                    # Verify this page exists by checking if it's different from current
                    if next_page_url != page_url:
                        page_url = next_page_url
                        continue
                except Exception as e:
                    st.warning(f"Failed to construct GSMArena next page URL: {e}")
            
            # Method 4: Add page parameter if none exists
            elif not next_link and "page=" not in page_url:
                separator = "&" if "?" in page_url else "?"
                next_page_url = f"{page_url}{separator}page=2"
                page_url = next_page_url
                continue
            
            # If we found a next link, process it
            if next_link:
                href = next_link["href"]
                if href.startswith("http"):
                    page_url = href
                else:
                    page_url = urljoin("https://www.gsmarena.com/", href)
                st.info(f"🔗 Found GSMArena next page: {page_url}")
            else:
                st.info("🏁 No more GSMArena pagination links found")
                break

    except Exception as e:
        st.error(f"⚠️ GSMArena reviews fetch failed: {e}")
        import traceback
        st.error(traceback.format_exc())

    st.success(f"🎉 Successfully collected {len(reviews)} GSMArena reviews from {page_count} pages")
    return reviews[:limit]

# -----------------------------
# Jumia Functions
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="🔎 Searching Jumia...")
def resolve_jumia_url(product_name: str):
    """Search Jumia Nigeria and return (product_url, review_url)."""
    try:
        query = product_name.strip().replace(" ", "+")
        search_url = f"https://www.jumia.com.ng/catalog/?q={requests.utils.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        
        r = requests.get(search_url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Look for product links in search results
        product_links = soup.select("a[href*='jumia.com.ng/'][href*='.html']")
        
        for link in product_links:
            href = link.get("href", "")
            if href.startswith("/"):
                product_url = "https://www.jumia.com.ng" + href
            elif href.startswith("http"):
                product_url = href
            else:
                continue
                
            # Check if this looks like a phone product
            link_text = link.get_text(strip=True).lower()
            product_keywords = ["samsung", "iphone", "galaxy", "phone", "android", "ios", "mobile"]
            if any(keyword in product_name.lower() for keyword in product_keywords if keyword in link_text):
                # Extract SKU from product URL to build review URL
                review_url = build_jumia_review_url(product_url)
                return product_url, review_url
                
        return None, None
        
    except Exception as e:
        st.warning(f"⚠️ Jumia search failed: {e}")
        return None, None

def build_jumia_review_url(product_url: str) -> str:
    """
    Convert Jumia product URL to reviews URL.
    Example: https://www.jumia.com.ng/samsung-galaxy-a05-6.7-4gb-ram64gb-rom-android-13-black-397109308.html
    To: https://www.jumia.com.ng/catalog/productratingsreviews/sku/SA948MP6AH2XVNAFAMZ/
    """
    if not product_url:
        return None
        
    try:
        # Try to extract SKU from product page HTML (more reliable)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        r = requests.get(product_url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Look for SKU in various places
        sku = None
        
        # Method 1: Look for data-sku attribute
        sku_element = soup.find(attrs={"data-sku": True})
        if sku_element:
            sku = sku_element["data-sku"]
        
        # Method 2: Look in script tags for SKU
        if not sku:
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string:
                    sku_match = re.search(r'"sku":\s*"([^"]+)"', script.string)
                    if sku_match:
                        sku = sku_match.group(1)
                        break
        
        # Method 3: Look for product ID in URL patterns or meta tags
        if not sku:
            # Try to find in meta tags
            meta_sku = soup.find("meta", {"name": "product:retailer_item_id"})
            if meta_sku and meta_sku.get("content"):
                sku = meta_sku["content"]
        
        # Method 4: Extract from URL structure (fallback)
        if not sku:
            # Extract number from end of URL before .html
            url_match = re.search(r"-(\d+)\.html$", product_url)
            if url_match:
                product_id = url_match.group(1)
                # This is a fallback - might not work for all products
                sku = f"JUMIA{product_id}"
        
        if sku:
            review_url = f"https://www.jumia.com.ng/catalog/productratingsreviews/sku/{sku}/"
            return review_url
            
        return None
        
    except Exception as e:
        st.warning(f"⚠️ Failed to build Jumia review URL: {e}")
        return None

@st.cache_data(ttl=21600, show_spinner="💬 Fetching Jumia reviews...")
def fetch_jumia_reviews(review_url: str, limit: int = 500):
    """
    Fetch reviews from Jumia review pages with pagination support.
    Returns list of review texts.
    """
    reviews = []
    if not review_url:
        return reviews

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.jumia.com.ng/",
    }
    
    page = 1
    max_pages = 20  # Safety limit
    
    try:
        while len(reviews) < limit and page <= max_pages:
            # Jumia reviews pagination usually uses ?page= parameter
            if page == 1:
                page_url = review_url
            else:
                separator = "&" if "?" in review_url else "?"
                page_url = f"{review_url}{separator}page={page}"
            
            st.info(f"📖 Scraping Jumia reviews page {page}: {len(reviews)} reviews so far...")
            
            time.sleep(1.5)  # Be respectful to Jumia servers
            r = requests.get(page_url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Look for review containers - Jumia uses various selectors
            review_blocks = []
            
            # Try different selectors for Jumia reviews
            selectors_to_try = [
                ".review-item",
                ".review-content", 
                ".user-review",
                ".comment",
                ".review-text",
                "[data-automation-id*='review']",
                ".ratings-reviews .review",
                ".product-review",
                ".customer-review"
            ]
            
            for selector in selectors_to_try:
                blocks = soup.select(selector)
                if blocks:
                    review_blocks.extend(blocks)
                    break
            
            # Fallback: look for review-like content in divs
            if not review_blocks:
                all_divs = soup.find_all(['div', 'p', 'span'])
                for div in all_divs:
                    text = div.get_text(strip=True)
                    # Look for review patterns
                    if (20 < len(text) < 1000 and 
                        any(word in text.lower() for word in [
                            'product', 'quality', 'delivery', 'good', 'bad', 'excellent', 
                            'phone', 'battery', 'camera', 'screen', 'fast', 'slow',
                            'recommend', 'love', 'hate', 'amazing', 'terrible', 'satisfied'
                        ]) and
                        not any(skip in text.lower() for skip in [
                            'jumia', 'login', 'register', 'cart', 'checkout', 'policy',
                            'terms', 'conditions', 'copyright', 'advertisement'
                        ])):
                        review_blocks.append(div)
            
            new_reviews_count = 0
            for block in review_blocks:
                # Get review text
                review_text = block.get_text(" ", strip=True)
                
                # Filter out non-review content
                if (15 < len(review_text) < 1000 and  # Reasonable length
                    not any(skip in review_text.lower() for skip in [
                        'jumia', 'admin', 'moderator', 'advertisement', 'sponsored',
                        'terms of service', 'privacy policy', 'click here', 'visit',
                        'login', 'register', 'cart', 'checkout', 'add to cart'
                    ]) and
                    # Must look like actual review content
                    any(keyword in review_text.lower() for keyword in [
                        'good', 'bad', 'excellent', 'terrible', 'amazing', 'awful',
                        'love', 'hate', 'recommend', 'quality', 'delivery', 'fast',
                        'slow', 'satisfied', 'disappointed', 'happy', 'phone',
                        'battery', 'camera', 'screen', 'performance'
                    ])):
                    
                    # Avoid duplicates
                    if not any(review_text[:50] in existing_review for existing_review in reviews):
                        reviews.append(review_text)
                        new_reviews_count += 1
                        if len(reviews) >= limit:
                            break
            
            st.info(f"✅ Found {new_reviews_count} new Jumia reviews on page {page} (total: {len(reviews)})")
            
            if new_reviews_count == 0:
                st.warning(f"⚠️ No new Jumia reviews found on page {page}, stopping pagination")
                break
            
            page += 1
            
    except Exception as e:
        st.error(f"⚠️ Jumia reviews fetch failed: {e}")
        import traceback
        st.error(traceback.format_exc())

    st.success(f"🎉 Successfully collected {len(reviews)} Jumia reviews from {page-1} pages")
    return reviews[:limit]

# -----------------------------
# Prompt builders
# -----------------------------
def chunk_prompt(product_name: str, specs: dict, reviews_subset: list, source_info: str = ""):
    """Build a strict chunk prompt for summarizing a subset of reviews."""
    specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
    reviews_context = "\n".join([f"- {r}" for r in reviews_subset]) if reviews_subset else "No reviews found"
    
    source_note = f" (Sources: {source_info})" if source_info else ""

    prompt = f"""
You are an AI Review Summarizer analyzing the {product_name}{source_note}.
Combine official specs with real user reviews to create a concise structured JSON summary.

SPECS:
{specs_context}

REVIEWS SAMPLE:
{reviews_context}

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary).
- Do not split or spell words character-by-character.
- Arrays (pros, cons, user_quotes) must be JSON arrays of full strings, e.g. ["Great battery", "Sharp display"].
- Keep 'verdict' under 30 characters and 'recommendation' under 35 characters.
- Provide 2-6 user quotes (1-2 sentences each).
- Include all phone_specs fields in phone_specs object.
- Keep aspect_sentiments as a list of objects with Aspect, Positive (0-100), Negative (0-100).

Return a JSON object matching this example schema:
{{
  "verdict": "Short verdict",
  "pros": ["..."],
  "cons": ["..."],
  "aspect_sentiments": [
    {{"Aspect":"Camera","Positive":70,"Negative":30}},
    {{"Aspect":"Battery","Positive":80,"Negative":20}}
  ],
  "user_quotes": ["...", "..."],
  "recommendation": "Short target audience",
  "bottom_line": "2-sentence final summary",
  "phone_specs": {{
    "Display": "{specs.get('Display', 'Not specified')}",
    "Processor": "{specs.get('Processor', 'Not specified')}",
    "RAM": "{specs.get('RAM', 'Not specified')}",
    "Storage": "{specs.get('Storage', 'Not specified')}",
    "Camera": "{specs.get('Camera', 'Not specified')}",
    "Battery": "{specs.get('Battery', 'Not specified')}",
    "OS": "{specs.get('OS', 'Not specified')}"
  }}
}}
"""
    return prompt

def final_merge_prompt(product_name: str, specs: dict, partial_texts: list, total_reviews: int, source_breakdown: dict):
    """
    Ask the model to merge multiple partial JSON summaries into one final valid JSON.
    This prompt is strict to avoid character-splitting.
    """
    joined = "\n\n---- PARTIAL SUMMARY ----\n\n".join(partial_texts)
    
    source_summary = ", ".join([f"{count} from {source}" for source, count in source_breakdown.items() if count > 0])
    
    prompt = f"""
You are an AI assistant. You are given multiple JSON partial analyses for the same phone ({product_name}).
Each partial analysis is valid JSON (or near-JSON). Your job: MERGE them into ONE VALID JSON object that exactly matches the schema below.

REVIEW SOURCES: Analyzed {total_reviews} total reviews ({source_summary})

Requirements:
- Return ONLY valid JSON (no explanation).
- Ensure 'pros' and 'cons' are arrays of short phrases (not single characters).
- Deduplicate pros/cons while keeping meaningful phrases.
- Aggregate user quotes (2-6 unique quotes from different sources when possible).
- For aspect_sentiments, combine by taking the arithmetic mean of Positive and Negative across partials for each Aspect (round to nearest integer).
- Ensure phone_specs has all 7 fields; prefer the scraped specs if provided.
- Consider insights from both GSMArena and Jumia reviews for balanced perspective.

Partial analyses:
{joined}

Now output a single JSON object with this schema:
{{
  "verdict": "Short verdict under 30 chars",
  "pros": ["..."],
  "cons": ["..."],
  "aspect_sentiments": [{{"Aspect":"Camera","Positive":70,"Negative":30}},{{"Aspect":"Battery","Positive":80,"Negative":20}}],
  "user_quotes": ["quote1","quote2"],
  "recommendation": "short target audience under 35 chars",
  "bottom_line": "2-3 sentence final summary combining specs & reviews from multiple sources",
  "phone_specs": {{
    "Display":"...","Processor":"...","RAM":"...","Storage":"...","Camera":"...","Battery":"...","OS":"..."
  }}
}}
"""
    return prompt

# -----------------------------
# Enhanced Summarization pipeline with multi-source support
# -----------------------------
def summarize_reviews_chunked(product_name: str, specs: dict, gsmarena_reviews: list, jumia_reviews: list, chunk_size: int = 200, status_container=None, prog_bar=None):
    """
    Summarize reviews by chunking and then merging partial summaries from multiple sources.
    Returns JSON string (final merged JSON) or None
    """
    # Combine reviews with source tagging
    all_reviews = []
    source_breakdown = {"GSMArena": len(gsmarena_reviews), "Jumia": len(jumia_reviews)}
    
    # Add source context to reviews for better analysis
    for review in gsmarena_reviews:
        all_reviews.append(f"[GSMArena] {review}")
    
    for review in jumia_reviews:
        all_reviews.append(f"[Jumia] {review}")
    
    total_reviews = len(all_reviews)
    source_info = f"GSMArena: {len(gsmarena_reviews)}, Jumia: {len(jumia_reviews)}"
    
    if not all_reviews:
        # still ask model to produce a summary from specs only
        try:
            prompt = chunk_prompt(product_name, specs, [], source_info)
            resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            final_text = resp.text.strip()
            return final_text
        except Exception as e:
            if status_container:
                status_container.error(f"Gemini error: {e}")
            return None

    # split reviews into chunks
    chunks = [all_reviews[i:i + chunk_size] for i in range(0, len(all_reviews), chunk_size)]
    partial_texts = []

    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        if status_container:
            status_container.info(f"🤖 Summarizing chunk {idx}/{total} (reviews {((idx-1)*chunk_size)+1} - {min(idx*chunk_size, len(all_reviews))})...")
        if prog_bar:
            prog_bar.progress(int(((idx - 1) / total) * 80) + 35)  # Progress from 35-80%

        prompt = chunk_prompt(product_name, specs, chunk, source_info)
        try:
            resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            text = resp.text.strip()
            # if the model returned fenced code, strip fences
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
            # store raw chunk output (we'll merge later)
            partial_texts.append(text)
        except Exception as e:
            # store failure marker and continue
            partial_texts.append("{}")
            if status_container:
                status_container.warning(f"Chunk {idx} failed: {e}")
        time.sleep(0.2)  # small delay between calls

    # update progress to merging
    if status_container:
        status_container.info("🔀 Merging partial summaries...")
    if prog_bar:
        prog_bar.progress(90)

    # build merge prompt
    merge_prompt = final_merge_prompt(product_name, specs, partial_texts, total_reviews, source_breakdown)
    try:
        merge_resp = model.generate_content(merge_prompt, generation_config={"response_mime_type": "application/json"})
        final_text = merge_resp.text.strip()
        # strip fences if present
        if final_text.startswith("```"):
            final_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", final_text, flags=re.IGNORECASE).strip()
        return final_text
    except Exception as e:
        if status_container:
            status_container.error(f"Final merge failed: {e}")
        return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Phone Review Engine", page_icon="📱", layout="wide")
st.title("📱 AI-Powered Phone Review Engine")
st.markdown("Combine **GSMArena specs** with real user reviews from **GSMArena & Jumia** and summarize with Gemini.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
review_limit_gsmarena = st.sidebar.slider("Max GSMArena reviews", 50, 800, 300, step=50)
review_limit_jumia = st.sidebar.slider("Max Jumia reviews", 50, 500, 200, step=50)
enable_jumia = st.sidebar.checkbox("Enable Jumia reviews", value=True)
chunk_size = 200  # fixed, as requested
show_raw = st.sidebar.checkbox("Show raw AI JSON (expander)", value=False)

# Input + action
phone = st.text_input("Enter phone name (e.g., Samsung Galaxy S24 FE)", value="Samsung Galaxy S24")
analyze = st.button("🔍 Analyze Phone", type="primary")

if analyze and phone:
    status = st.empty()
    prog = st.progress(0)

    # Initialize review lists
    gsmarena_reviews = []
    jumia_reviews = []
    
    # GSMArena Processing
    status.text("🔎 Resolving GSMArena product page...")
    gsmarena_product_url, gsmarena_review_url = resolve_gsmarena_url(phone)
    prog.progress(5)

    if not gsmarena_product_url:
        st.error(f"❌ Could not find '{phone}' on GSMArena.")
        status.empty()
        prog.empty()
        st.stop()

    st.success(f"✅ GSMArena product page: {gsmarena_product_url}")
    if gsmarena_review_url:
        st.success(f"✅ GSMArena reviews page: {gsmarena_review_url}")
    else:
        st.warning("⚠️ Couldn't build GSMArena reviews URL automatically.")

    # Fetch GSMArena specs
    status.text("📊 Fetching GSMArena specs...")
    specs = fetch_gsmarena_specs(gsmarena_product_url)
    prog.progress(10)

    # Fetch GSMArena reviews
    status.text("💬 Collecting GSMArena user reviews...")
    gsmarena_reviews = fetch_gsmarena_reviews(gsmarena_review_url, limit=review_limit_gsmarena) if gsmarena_review_url else []
    prog.progress(20)
    st.info(f"✅ Collected {len(gsmarena_reviews)} GSMArena reviews (cap: {review_limit_gsmarena})")

    # Jumia Processing (if enabled)
    if enable_jumia:
        status.text("🔎 Resolving Jumia product page...")
        jumia_product_url, jumia_review_url = resolve_jumia_url(phone)
        prog.progress(25)

        if jumia_product_url:
            st.success(f"✅ Jumia product page: {jumia_product_url}")
            if jumia_review_url:
                st.success(f"✅ Jumia reviews page: {jumia_review_url}")
            else:
                st.warning("⚠️ Couldn't build Jumia reviews URL automatically.")
            
            # Fetch Jumia reviews
            status.text("💬 Collecting Jumia user reviews...")
            jumia_reviews = fetch_jumia_reviews(jumia_review_url, limit=review_limit_jumia) if jumia_review_url else []
            prog.progress(30)
            st.info(f"✅ Collected {len(jumia_reviews)} Jumia reviews (cap: {review_limit_jumia})")
        else:
            st.warning(f"⚠️ Could not find '{phone}' on Jumia. Proceeding with GSMArena data only.")
            prog.progress(30)
    else:
        st.info("ℹ️ Jumia reviews disabled in settings.")
        prog.progress(30)

    # Summary of data collected
    total_reviews = len(gsmarena_reviews) + len(jumia_reviews)
    st.info(f"📊 **Total data collected:** {len(specs)} specs, {total_reviews} reviews ({len(gsmarena_reviews)} GSMArena + {len(jumia_reviews)} Jumia)")

    if total_reviews == 0:
        st.warning("⚠️ No reviews found from any source. Analysis will be based on specs only.")

    # Summarize with chunking
    status.text("🤖 Summarizing reviews with Gemini in chunks...")
    final_json_text = summarize_reviews_chunked(
        phone, specs, gsmarena_reviews, jumia_reviews, 
        chunk_size=chunk_size, status_container=status, prog_bar=prog
    )
    prog.progress(100)
    status.empty()

    if not final_json_text:
        st.error("⚠️ Failed to produce final summary.")
        st.stop()

    # Parse final JSON
    final_obj = safe_load_json(final_json_text)
    if not final_obj:
        # Try to extract JSON using regex (last attempt)
        m = re.search(r"\{(?:.|\n)*\}", final_json_text)
        if m:
            try:
                final_obj = json.loads(m.group(0))
            except Exception:
                final_obj = None

    if not final_obj:
        st.error("⚠️ Could not parse AI output as JSON. Showing raw output for debugging.")
        with st.expander("Raw AI output"):
            st.text_area("AI output", final_json_text, height=400)
        st.stop()

    # Ensure phone_specs fallback to scraped specs where missing
    phone_specs = final_obj.get("phone_specs", {})
    for k in ["Display", "Processor", "RAM", "Storage", "Camera", "Battery", "OS"]:
        phone_specs.setdefault(k, specs.get(k, "Not specified"))
    final_obj["phone_specs"] = phone_specs

    # Display results in readable UI
    st.markdown("---")
    st.subheader(f"📝 Analysis — {phone}")

    # Metric cards row with source info
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div style='background:#0b0b0b;color:white;padding:12px;border-radius:8px;'><b>⭐ Verdict</b><div style='font-size:18px;margin-top:6px'>{final_obj.get('verdict','N/A')}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='background:#0b0b0b;color:white;padding:12px;border-radius:8px;'><b>🎯 Best For</b><div style='font-size:18px;margin-top:6px'>{final_obj.get('recommendation','N/A')}</div></div>", unsafe_allow_html=True)
    with c3:
        source_text = f"GSMArena: {len(gsmarena_reviews)}"
        if enable_jumia and len(jumia_reviews) > 0:
            source_text += f", Jumia: {len(jumia_reviews)}"
        st.markdown(f"<div style='background:#0b0b0b;color:white;padding:12px;border-radius:8px;'><b>📊 Sources</b><div style='font-size:14px;margin-top:6px'>{source_text}<br/>{len(phone_specs)} specs total</div></div>", unsafe_allow_html=True)

    # Bottom line
    st.markdown("### 🎯 Bottom Line")
    st.info(final_obj.get("bottom_line", "No summary available."))

    # Two column main
    left, right = st.columns([0.6, 0.4])
    with left:
        # Specs table
        st.markdown("### 🔧 Technical Specifications")
        df_specs = pd.DataFrame(list(phone_specs.items()), columns=["Component", "Details"])
        st.table(df_specs)

        # Aspect sentiments chart (if present)
        if final_obj.get("aspect_sentiments"):
            df_aspects = pd.DataFrame(final_obj["aspect_sentiments"])
            if not df_aspects.empty and "Aspect" in df_aspects.columns:
                df_chart = df_aspects.set_index("Aspect")
                # ensure Positive/Negative present
                if "Positive" in df_chart.columns and "Negative" in df_chart.columns:
                    st.markdown("### 📊 User Sentiment (Combined Sources)")
                    st.bar_chart(df_chart[["Positive", "Negative"]], height=320)

    with right:
        st.markdown("### ✅ Strengths")
        for p in final_obj.get("pros", []):
            st.success(p)
        st.markdown("### ⚠️ Weaknesses")
        for c in final_obj.get("cons", []):
            st.error(c)

    # User quotes with improved display
    if final_obj.get("user_quotes"):
        st.markdown("### 💬 What Users Are Saying")
        for i, quote in enumerate(final_obj.get("user_quotes", []), 1):
            # Try to identify source from quote if it has source tags
            if quote.startswith("[GSMArena]"):
                source_icon = "🔧"
                clean_quote = quote.replace("[GSMArena]", "").strip()
            elif quote.startswith("[Jumia]"):
                source_icon = "🛒"
                clean_quote = quote.replace("[Jumia]", "").strip()
            else:
                source_icon = "👤"
                clean_quote = quote
            
            st.info(f"{source_icon} **User {i}:** {clean_quote}")

    # Source breakdown info
    if enable_jumia and (len(gsmarena_reviews) > 0 or len(jumia_reviews) > 0):
        st.markdown("### 📈 Review Sources Breakdown")
        source_data = {
            "Source": ["GSMArena", "Jumia"],
            "Reviews Collected": [len(gsmarena_reviews), len(jumia_reviews)],
            "Percentage": [
                f"{len(gsmarena_reviews)/max(total_reviews,1)*100:.1f}%" if total_reviews > 0 else "0%",
                f"{len(jumia_reviews)/max(total_reviews,1)*100:.1f}%" if total_reviews > 0 else "0%"
            ]
        }
        df_sources = pd.DataFrame(source_data)
        st.table(df_sources)

    # Download & raw
    st.markdown("---")
    
    # Add metadata to JSON for download
    final_obj["_metadata"] = {
        "phone_name": phone,
        "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "gsmarena_reviews": len(gsmarena_reviews),
            "jumia_reviews": len(jumia_reviews),
            "total_reviews": total_reviews
        },
        "gsmarena_urls": {
            "product": gsmarena_product_url,
            "reviews": gsmarena_review_url
        }
    }
    
    if enable_jumia:
        final_obj["_metadata"]["jumia_urls"] = {
            "product": jumia_product_url,
            "reviews": jumia_review_url
        }
    
    st.download_button(
        "📥 Download Complete Analysis JSON", 
        json.dumps(final_obj, indent=2, ensure_ascii=False), 
        f"{phone.replace(' ','_')}_analysis_{time.strftime('%Y%m%d')}.json", 
        "application/json"
    )
    
    if show_raw:
        with st.expander("📄 Raw AI JSON"):
            st.json(final_obj)