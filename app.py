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
from groq import Groq
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

# -----------------------------
# Configure Groq API
# -----------------------------
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("❌ Missing Groq API key. Please set GROQ_API_KEY in secrets or env vars.")
    st.stop()

client = Groq(api_key=api_key)

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
# Resolve GSMArena product + review URL
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
# Reviews scraper
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
            
            time.sleep(2)
            r = requests.get(page_url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

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

# -----------------------------
# Prompt builders
# -----------------------------
def chunk_prompt(product_name: str, specs: dict, reviews_subset: list):
    """Build a strict chunk prompt for summarizing a subset of reviews."""
    specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
    reviews_context = "\n".join([f"- {r}" for r in reviews_subset]) if reviews_subset else "No reviews found"

    prompt = f"""
You are an AI Review Summarizer analyzing the {product_name}.
Combine GSMArena official specs with real user reviews to create a concise structured JSON summary.

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

def final_merge_prompt(product_name: str, specs: dict, partial_texts: list):
    """
    Ask the model to merge multiple partial JSON summaries into one final valid JSON.
    """
    joined = "\n\n---- PARTIAL SUMMARY ----\n\n".join(partial_texts)
    prompt = f"""
You are an AI assistant. You are given multiple JSON partial analyses for the same phone ({product_name}).
Each partial analysis is valid JSON (or near-JSON). Your job: MERGE them into ONE VALID JSON object that exactly matches the schema below.

Requirements:
- Return ONLY valid JSON (no explanation).
- Ensure 'pros' and 'cons' are arrays of short phrases (not single characters).
- Deduplicate pros/cons while keeping meaningful phrases.
- Aggregate user quotes (2-6 unique quotes).
- For aspect_sentiments, combine by taking the arithmetic mean of Positive and Negative across partials for each Aspect (round to nearest integer).
- Ensure phone_specs has all 7 fields; prefer the scraped specs if provided.

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
  "bottom_line": "2-3 sentence final summary combining specs & reviews",
  "phone_specs": {{
    "Display":"...","Processor":"...","RAM":"...","Storage":"...","Camera":"...","Battery":"...","OS":"..."
  }}
}}
"""
    return prompt

# -----------------------------
# Summarization with Groq
# -----------------------------
def summarize_reviews_chunked(product_name: str, specs: dict, reviews: list, chunk_size: int = 200, status_container=None, prog_bar=None):
    """
    Summarize reviews by chunking and then merging partial summaries using Groq.
    """
    if not reviews:
        try:
            prompt = chunk_prompt(product_name, specs, [])
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            final_text = response.choices[0].message.content.strip()
            return final_text
        except Exception as e:
            if status_container:
                status_container.error(f"Groq error (no reviews): {e}")
            st.error(f"Full error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
    partial_texts = []

    total = len(chunks)
    failed_chunks = 0
    
    for idx, chunk in enumerate(chunks, start=1):
        if status_container:
            status_container.info(f"🤖 Summarizing chunk {idx}/{total} (reviews {((idx-1)*chunk_size)+1} - {min(idx*chunk_size, len(reviews))})...")
        if prog_bar:
            prog_bar.progress(int(((idx - 1) / total) * 45) + 35)

        prompt = chunk_prompt(product_name, specs, chunk)
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
            partial_texts.append(text)
            if status_container:
                status_container.success(f"✅ Chunk {idx} completed")
        except Exception as e:
            failed_chunks += 1
            partial_texts.append("{}")
            if status_container:
                status_container.warning(f"⚠️ Chunk {idx} failed: {e}")
            st.warning(f"Chunk {idx} error details: {str(e)}")
        time.sleep(0.5)

    if status_container:
        status_container.info(f"📊 Processed {total} chunks ({failed_chunks} failed)")

    if failed_chunks > total * 0.5:
        if status_container:
            status_container.error(f"❌ Too many chunks failed ({failed_chunks}/{total}). Cannot proceed.")
        return None

    if status_container:
        status_container.info("🔀 Merging partial summaries...")
    if prog_bar:
        prog_bar.progress(90)

    merge_prompt = final_merge_prompt(product_name, specs, partial_texts)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": merge_prompt}],
            temperature=0.5,
            max_tokens=2048
        )
        final_text = response.choices[0].message.content.strip()
        if final_text.startswith("```"):
            final_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", final_text, flags=re.IGNORECASE).strip()
        if status_container:
            status_container.success("✅ Final merge completed")
        return final_text
    except Exception as e:
        if status_container:
            status_container.error(f"❌ Final merge failed: {e}")
        st.error(f"Merge error details: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Phone Review Engine", page_icon="📱", layout="wide")
st.title("📱 AI-Powered Phone Review Engine")
st.markdown("Combine GSMArena specs with **real user reviews** and summarize with Groq AI.")

st.sidebar.header("⚙️ Settings")
review_limit = st.sidebar.slider("Max reviews to analyze", 50, 1000, 400, step=50)
chunk_size = 200
show_raw = st.sidebar.checkbox("Show raw AI JSON (expander)", value=False)

phone = st.text_input("Enter phone name (e.g., Samsung Galaxy S24 FE)", value="Samsung Galaxy S24")
analyze = st.button("🔍 Analyze Phone", type="primary")

if analyze and phone:
    status = st.empty()
    prog = st.progress(0)

    status.text("🔎 Resolving product page...")
    product_url, review_url = resolve_gsmarena_url(phone)
    prog.progress(5)

    if not product_url:
        st.error(f"❌ Could not find '{phone}' on GSMArena.")
        status.empty()
        prog.empty()
        st.stop()

    st.success(f"✅ Product page: {product_url}")
    if review_url:
        st.success(f"✅ Reviews page: {review_url}")
    else:
        st.warning("⚠️ Couldn't build reviews URL automatically. Proceeding with specs only.")

    status.text("📊 Fetching specs...")
    specs = fetch_gsmarena_specs(product_url)
    prog.progress(15)

    status.text("💬 Collecting user reviews...")
    reviews = fetch_gsmarena_reviews(review_url, limit=review_limit) if review_url else []
    prog.progress(35)
    st.info(f"✅ Collected {len(reviews)} reviews (cap: {review_limit})")

    status.text("🤖 Summarizing reviews with Groq AI...")
    final_json_text = summarize_reviews_chunked(phone, specs, reviews, chunk_size=chunk_size, status_container=status, prog_bar=prog)
    prog.progress(100)
    status.empty()

    if not final_json_text:
        st.error("⚠️ Failed to produce final summary.")
        st.stop()

    final_obj = safe_load_json(final_json_text)
    if not final_obj:
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

    phone_specs = final_obj.get("phone_specs", {})
    for k in ["Display", "Processor", "RAM", "Storage", "Camera", "Battery", "OS"]:
        phone_specs.setdefault(k, specs.get(k, "Not specified"))
    final_obj["phone_specs"] = phone_specs

    st.markdown("---")
    st.subheader(f"📝 Analysis — {phone}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div style='background:#0b0b0b;color:white;padding:12px;border-radius:8px;'><b>⭐ Verdict</b><div style='font-size:18px;margin-top:6px'>{final_obj.get('verdict','N/A')}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='background:#0b0b0b;color:white;padding:12px;border-radius:8px;'><b>🎯 Best For</b><div style='font-size:18px;margin-top:6px'>{final_obj.get('recommendation','N/A')}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div style='background:#0b0b0b;color:white;padding:12px;border-radius:8px;'><b>📊 Data</b><div style='font-size:18px;margin-top:6px'>{len(phone_specs)} specs, {len(reviews)} reviews</div></div>", unsafe_allow_html=True)

    st.markdown("### 🎯 Bottom Line")
    st.info(final_obj.get("bottom_line", "No summary available."))

    left, right = st.columns([0.6, 0.4])
    with left:
        st.markdown("### 🔧 Technical Specifications")
        df_specs = pd.DataFrame(list(phone_specs.items()), columns=["Component", "Details"])
        st.table(df_specs)

        if final_obj.get("aspect_sentiments"):
            df_aspects = pd.DataFrame(final_obj["aspect_sentiments"])
            if not df_aspects.empty and "Aspect" in df_aspects.columns:
                df_chart = df_aspects.set_index("Aspect")
                if "Positive" in df_chart.columns and "Negative" in df_chart.columns:
                    st.markdown("### 📊 User Sentiment")
                    st.bar_chart(df_chart[["Positive", "Negative"]], height=320)

    with right:
        st.markdown("### ✅ Strengths")
        for p in final_obj.get("pros", []):
            st.success(p)
        st.markdown("### ⚠️ Weaknesses")
        for c in final_obj.get("cons", []):
            st.error(c)

    if final_obj.get("user_quotes"):
        st.markdown("### 💬 What Users Are Saying")
        for i, q in enumerate(final_obj.get("user_quotes", []), 1):
            st.info(f"**User {i}:** {q}")

    st.markdown("---")
    st.download_button("📥 Download JSON", json.dumps(final_obj, indent=2, ensure_ascii=False), f"{phone.replace(' ','_')}_summary.json", "application/json")
    if show_raw:
        with st.expander("📄 Raw AI JSON"):
            st.json(final_obj)