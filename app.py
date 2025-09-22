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
# Resolve GSMArena product + review URL
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="🔎 Searching GSMArena...")
def resolve_gsmarena_url(product_name: str):
    """Search GSMArena and return (product_url, review_url)."""
    try:
        query = product_name.strip()
        search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(search_url, headers=headers, timeout=10)
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
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
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
# Reviews scraper (pagination aware)
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

    headers = {"User-Agent": "Mozilla/5.0"}
    page_url = url

    try:
        while page_url and len(reviews) < limit:
            time.sleep(1)  # polite pause
            r = requests.get(page_url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Try common selectors
            blocks = []
            blocks.extend(soup.select(".opin"))
            blocks.extend(soup.select(".user-opinion"))
            blocks.extend(soup.select(".uopin"))
            blocks.extend(soup.select(".user-review"))
            blocks.extend(soup.select(".review-item"))
            # fallback paragraphs from comments area
            if not blocks:
                blocks = soup.select("#user-comments p, .thread p, .user-thread p, .post p, p")

            for blk in blocks:
                text = blk.get_text(" ", strip=True)
                if 30 < len(text) < 1200:
                    low = text.lower()
                    if not any(skip in low for skip in ["gsmarena", "admin", "moderator", "delete", "report"]):
                        reviews.append(text)
                        if len(reviews) >= limit:
                            break

            # detect next link (a.pages-next, rel=next, or 'next' text)
            next_link = soup.select_one("a.pages-next") or soup.find("a", attrs={"rel": "next"}) or soup.find("a", string=re.compile(r"next", re.I))
            if next_link and next_link.has_attr("href"):
                href = next_link["href"]
                page_url = href if href.startswith("http") else "https://www.gsmarena.com/" + href.lstrip("/")
            else:
                # try incrementing ?page= if present
                m = re.search(r"([?&]page=)(\d+)", page_url)
                if m:
                    current = int(m.group(2))
                    page_url = re.sub(r"([?&]page=)\d+", r"\g<1>%d" % (current + 1), page_url)
                else:
                    page_url = None
    except Exception as e:
        st.warning(f"⚠️ GSMArena reviews fetch failed: {e}")

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
    This prompt is strict to avoid character-splitting.
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
# Summarization pipeline (chunked) with progress
# -----------------------------
def summarize_reviews_chunked(product_name: str, specs: dict, reviews: list, chunk_size: int = 200, status_container=None, prog_bar=None):
    """
    Summarize reviews by chunking and then merging partial summaries.
    status_container: optional st.empty() for status messages
    prog_bar: optional st.progress() to update chunk progress
    Returns JSON string (final merged JSON) or None
    """
    if not reviews:
        # still ask model to produce a summary from specs only
        try:
            prompt = chunk_prompt(product_name, specs, [])
            resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            final_text = resp.text.strip()
            return final_text
        except Exception as e:
            if status_container:
                status_container.error(f"Gemini error: {e}")
            return None

    # split reviews into chunks
    chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
    partial_texts = []

    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        if status_container:
            status_container.info(f"🤖 Summarizing chunk {idx}/{total} (reviews {((idx-1)*chunk_size)+1} - {min(idx*chunk_size, len(reviews))})...")
        if prog_bar:
            prog_bar.progress(int(((idx - 1) / total) * 100))

        prompt = chunk_prompt(product_name, specs, chunk)
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
    merge_prompt = final_merge_prompt(product_name, specs, partial_texts)
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