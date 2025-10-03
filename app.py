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
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # First, strip non-JSON output (like "```json" fences)
        text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        
        # Then, try to find the first complete JSON object
        m = re.search(r"\{(?:.|\n)*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                # If regex-extracted block still fails, return None
                return None
        return None

# -----------------------------
# Resolve GSMArena product + review URL - FIX APPLIED HERE
# -----------------------------
# Changed caching key to ensure robustness against environment
def get_cache_key_resolve_gsmarena(product_name: str):
    """Generates a cache key based on the sanitized product name."""
    return product_name.strip().lower().replace(" ", "-")

@st.cache_data(ttl=86400, show_spinner="🔎 Searching GSMArena...", show_hash=False)
def resolve_gsmarena_url(product_name: str):
    """Search GSMArena and return (product_url, review_url)."""
    try:
        # **CRITICAL FIX**: Explicitly cast and strip the input to prevent MissingSchema/InvalidSchema error
        query = str(product_name).strip()
        if not query:
             return None, None
             
        # Use urllib.parse.quote for URL encoding (more standard)
        encoded_query = quote(query)
        search_url = f"[https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName=](https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName=){encoded_query}"
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        # **CRITICAL CHECK**: Ensure URL starts with http/https
        if not search_url.startswith("http"):
            raise ValueError(f"URL schema is missing or invalid: {search_url}")
            
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
        product_url = "[https://www.gsmarena.com/](https://www.gsmarena.com/)" + product_href
        # Build review URL reliably
        review_url = build_review_url(product_url)
        return product_url, review_url
    except Exception as e:
        # The URL that failed is already printed in the traceback, we just print the error
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

# -----------------------------
# Reviews scraper with better pagination detection
# -----------------------------
@st.cache_data(ttl=21600, show_spinner="💬 Fetching user reviews...")
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
                st.warning(f"🔄 Breaking pagination loop - already visited: {page_url}")
                break
            
            visited_urls.add(page_url)
            page_count += 1
            
            st.info(f"📖 Scraping page {page_count}: {len(reviews)} reviews so far...")
            
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

            st.info(f"✅ Found {new_reviews_count} new reviews on page {page_count} (total: {len(reviews)})")
            
            if new_reviews_count == 0:
                st.warning(f"⚠️ No new reviews found on page {page_count}, stopping pagination")
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
                    st.warning(f"Failed to construct next page URL: {e}")
            
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
                    # Use urljoin to handle relative paths reliably
                    page_url = urljoin(url, href)
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

# -----------------------------
# Prompt builders (No changes here, already robust)
# -----------------------------
def chunk_prompt(product_name: str, specs: dict, reviews_subset: list):
    """Build a strict chunk prompt for summarizing a subset of reviews."""
    # Ensure specs are used only if available
    specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
    reviews_context = "\n".join([f"- {r}" for r in reviews_subset]) if reviews_subset else "No reviews found"

    # Use dict comprehension for dynamic spec insertion into the schema example
    phone_specs_schema = ",\n".join([
        f'    "{k}": "{{specs.get(\'{k}\', \'Not specified\')}}"' for k in 
        ["Display", "Processor", "RAM", "Storage", "Camera", "Battery", "OS"]
    ])
    
    prompt = f"""
You are an AI Review Summarizer analyzing the {product_name}.
Combine GSMArena official specs with real user reviews to create a concise structured JSON summary.

SPECS:
{specs_context}

REVIEWS SAMPLE:
{reviews_context}

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary, no backticks).
- Do not split or spell words character-by-character.
- Arrays (pros, cons, user_quotes) must be JSON arrays of full strings, e.g. ["Great battery", "Sharp display"].
- Keep 'verdict' under 30 characters and 'recommendation' under 35 characters.
- Provide 2-6 user quotes (1-2 sentences each).
- Include all phone_specs fields in phone_specs object.
- Keep aspect_sentiments as a list of objects with Aspect, Positive (0-100), Negative (0-100).

Return a JSON object matching this example schema:
{{
  "verdict": "Short verdict",
  "pros": ["...", "..."],
  "cons": ["...", "..."],
  "aspect_sentiments": [
    {{"Aspect":"Camera","Positive":70,"Negative":30}},
    {{"Aspect":"Battery","Positive":80,"Negative":20}}
  ],
  "user_quotes": ["...", "..."],
  "recommendation": "Short target audience",
  "bottom_line": "2-sentence final summary",
  "phone_specs": {{
{phone_specs_schema}
  }}
}}
"""
    return prompt

def final_merge_prompt(product_name: str, specs: dict, partial_texts: list):
    """
    Ask the model to merge multiple partial JSON summaries into one final valid JSON.
    This prompt is strict to avoid character-splitting.
    """
    joined = "\n\n---- PARTIAL SUMMARY ----\n\n".join(partial_texts)
    
    # Use dict comprehension for dynamic spec insertion into the schema example
    phone_specs_schema = ",\n".join([
        f'    "{k}": "{specs.get(k, "Not specified")}"' for k in 
        ["Display", "Processor", "RAM", "Storage", "Camera", "Battery", "OS"]
    ])
    
    prompt = f"""
You are an AI assistant. You are given multiple JSON partial analyses for the same phone ({product_name}).
Each partial analysis is valid JSON (or near-JSON). Your job: MERGE them into ONE VALID JSON object that exactly matches the schema below.

Requirements:
- Return ONLY valid JSON (no explanation, no commentary, no backticks).
- Ensure 'pros' and 'cons' are arrays of short phrases (not single characters).
- Deduplicate pros/cons while keeping meaningful phrases.
- Aggregate user quotes (2-6 unique quotes).
- For aspect_sentiments, combine by taking the arithmetic mean of Positive and Negative across partials for each Aspect (round to nearest integer).
- Ensure phone_specs has all 7 fields; use the provided scraped specs as the primary source for these values.

Partial analyses:
{joined}

Now output a single JSON object with this schema:
{{
  "verdict": "Short verdict under 30 chars",
  "pros": ["...","..."],
  "cons": ["...","..."],
  "aspect_sentiments": [{{"Aspect":"Camera","Positive":70,"Negative":30}},{{"Aspect":"Battery","Positive":80,"Negative":20}}],
  "user_quotes": ["quote1","quote2"],
  "recommendation": "short target audience under 35 chars",
  "bottom_line": "2-3 sentence final summary combining specs & reviews",
  "phone_specs": {{
{phone_specs_schema}
  }}
}}
"""
    return prompt

# -----------------------------
# Summarization pipeline (chunked) with progress - RETRIES INCLUDED
# -----------------------------
def summarize_reviews_chunked(product_name: str, specs: dict, reviews: list, chunk_size: int = 200, status_container=None, prog_bar=None, max_retries=3):
    """
    Summarize reviews by chunking and then merging partial summaries.
    INCLUDES RETRY LOGIC for API stability.
    Returns JSON string (final merged JSON) or None
    """
    
    # ------------------
    # Case 1: No Reviews (Specs only)
    # ------------------
    if not reviews:
        status_container.info("🤖 Generating summary from specs only (No reviews found)...")
        for attempt in range(max_retries):
            try:
                prompt = chunk_prompt(product_name, specs, [])
                resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                final_text = resp.text.strip()
                if safe_load_json(final_text):
                    return final_text
                else:
                    raise ValueError("Model did not return valid JSON despite request.")
            except Exception as e:
                if status_container:
                    status_container.warning(f"⚠️ Specs-only summary failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt + 1 == max_retries:
                    return None
                time.sleep(2 * (attempt + 1))  # Exponential backoff
        return None

    # ------------------
    # Case 2: Chunked Summarization
    # ------------------
    chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
    partial_texts = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        if prog_bar:
            # Progress from 35-90% during chunk processing
            prog_bar.progress(int(((idx - 1) / total) * 55) + 35) 

        for attempt in range(max_retries):
            status_container.info(f"🤖 Summarizing chunk {idx}/{total} (Attempt {attempt + 1}/{max_retries})...")
            prompt = chunk_prompt(product_name, specs, chunk)
            try:
                resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                text = resp.text.strip()
                
                # Check and clean output
                final_output = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
                
                # Check if output is valid JSON
                if safe_load_json(final_output):
                    partial_texts.append(final_output)
                    break # Success, move to next chunk
                else:
                    raise ValueError("Model did not return valid JSON despite request.")
                    
            except Exception as e:
                if status_container:
                    status_container.warning(f"Chunk {idx} failed (Attempt {attempt + 1}): {e}")
                if attempt + 1 == max_retries:
                    # Append failure marker after all retries fail
                    partial_texts.append("{}") 
                    break 
                time.sleep(2 * (attempt + 1)) # Exponential backoff

    # ------------------
    # Case 3: Final Merge
    # ------------------
    if prog_bar:
        prog_bar.progress(90)

    # Filter out empty failure markers (only keep successful/non-empty results)
    valid_partials = [p for p in partial_texts if safe_load_json(p) is not None and safe_load_json(p) != {}]
    
    if not valid_partials:
        status_container.error("❌ All summarization chunks failed. Cannot proceed to merge.")
        return None
        
    status_container.info(f"🔀 Merging {len(valid_partials)} successful partial summaries...")
    
    # build merge prompt
    merge_prompt = final_merge_prompt(product_name, specs, valid_partials)
    
    for attempt in range(max_retries):
        try:
            merge_resp = model.generate_content(merge_prompt, generation_config={"response_mime_type": "application/json"})
            final_text = merge_resp.text.strip()
            
            # Check and clean output
            final_output = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", final_text, flags=re.IGNORECASE | re.DOTALL).strip()
            
            # Final validation check
            if safe_load_json(final_output):
                 return final_output
            else:
                 raise ValueError("Final merge output was not valid JSON.")
                 
        except Exception as e:
            if status_container:
                status_container.error(f"⚠️ Final merge failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt + 1 == max_retries:
                return None
            time.sleep(5 * (attempt + 1)) # Longer backoff for the critical merge step
            
    return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Phone Review Engine", page_icon="📱", layout="wide")
st.title("📱 AI-Powered Phone Review Engine")
st.markdown("Combine GSMArena specs with **real user reviews** and summarize with Gemini.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
review_limit = st.sidebar.slider("Max reviews to analyze", 50, 1000, 400, step=50)
chunk_size = 200  # fixed, as requested
show_raw = st.sidebar.checkbox("Show raw AI JSON (expander)", value=False)

# Input + action
phone = st.text_input("Enter phone name (e.g., Samsung Galaxy S24 FE)", value="Samsung Galaxy S24")
analyze = st.button("🔍 Analyze Phone", type="primary")

if analyze and phone:
    status = st.empty()
    prog = st.progress(0)

    status.text("🔎 Resolving product page...")
    # Pass the product name string directly to the function
    product_url, review_url = resolve_gsmarena_url(phone)
    prog.progress(5)

    if not product_url:
        # Error already printed inside the function
        status.empty()
        prog.empty()
        st.stop()

    st.success(f"✅ Product page: {product_url}")
    if review_url:
        st.success(f"✅ Reviews page: {review_url}")
    else:
        st.warning("⚠️ Couldn't build reviews URL automatically. Proceeding with specs only.")

    # Fetch specs
    status.text("📊 Fetching specs...")
    specs = fetch_gsmarena_specs(product_url)
    prog.progress(15)

    # Fetch reviews
    status.text("💬 Collecting user reviews (this can take a while for many pages)...")
    reviews = fetch_gsmarena_reviews(review_url, limit=review_limit) if review_url else []
    prog.progress(35)
    st.info(f"✅ Collected {len(reviews)} reviews (cap: {review_limit})")

    # Summarize with chunking
    status.text("🤖 Summarizing reviews with Gemini in chunks...")
    final_json_text = summarize_reviews_chunked(phone, specs, reviews, chunk_size=chunk_size, status_container=status, prog_bar=prog)
    prog.progress(100)
    status.empty()

    if not final_json_text:
        st.error("⚠️ Failed to produce final summary. Check the logs/warnings above for API errors.")
        st.stop()

    # Parse final JSON
    final_obj = safe_load_json(final_json_text)
    
    if not final_obj:
        st.error("⚠️ Could not parse AI output as JSON. Showing raw output for debugging.")
        with st.expander("Raw AI output"):
            st.text_area("AI output", final_json_text, height=400)
        st.stop()

    # Ensure phone_specs fallback to scraped specs where missing
    phone_specs = final_obj.get("phone_specs", {})
    for k in ["Display", "Processor", "RAM", "Storage", "Camera", "Battery", "OS"]:
        # Fallback to scraped specs if AI output didn't populate them or used generic text
        if phone_specs.get(k) in [None, "Not specified", "..."]:
            phone_specs[k] = specs.get(k, "Not specified")
        
    final_obj["phone_specs"] = phone_specs

    # Display results in readable UI
    st.markdown("---")
    st.subheader(f"📝 Analysis — {phone}")

    # Metric cards row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div style='background:#0b0b0b;color:white;padding:12px;border-radius:8px;'><b>⭐ Verdict</b><div style='font-size:18px;margin-top:6px'>{final_obj.get('verdict','N/A')}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='background:#0b0b0b;color:white;padding:12px;border-radius:8px;'><b>🎯 Best For</b><div style='font-size:18px;margin-top:6px'>{final_obj.get('recommendation','N/A')}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div style='background:#0b0b0b;color:white;padding:12px;border-radius:8px;'><b>📊 Data</b><div style='font-size:18px;margin-top:6px'>{len(phone_specs)} specs, {len(reviews)} reviews</div></div>", unsafe_allow_html=True)

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
                    # calculate a net sentiment score for sorting/coloring
                    df_chart['Net'] = df_chart['Positive'] - df_chart['Negative']
                    df_chart = df_chart.sort_values(by='Net', ascending=False).drop(columns=['Net'])
                    
                    st.markdown("### 📊 User Sentiment")
                    st.bar_chart(df_chart[["Positive", "Negative"]], height=320)

    with right:
        st.markdown("### ✅ Strengths")
        for p in final_obj.get("pros", []):
            st.success(p)
        st.markdown("### ⚠️ Weaknesses")
        for c in final_obj.get("cons", []):
            st.error(c)

    # User quotes
    if final_obj.get("user_quotes"):
        st.markdown("### 💬 What Users Are Saying")
        for i, q in enumerate(final_obj.get("user_quotes", []), 1):
            st.info(f"**User {i}:** {q}")

    # Download & raw
    st.markdown("---")
    st.download_button("📥 Download JSON", json.dumps(final_obj, indent=2, ensure_ascii=False), f"{phone.replace(' ','_')}_summary.json", "application/json")
    if show_raw:
        with st.expander("📄 Raw AI JSON"):
            st.json(final_obj)
