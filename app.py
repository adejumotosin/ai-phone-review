# ==================================================
# 📱 AI-Powered Phone Review Engine
# Combines GSMArena specs + user reviews + Gemini AI
# ==================================================

# -----------------------------
# 📦 Imports
# -----------------------------
import os
import re
import json
import time
import hashlib
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import google.generativeai as genai

# -----------------------------
# 🔑 Configure Gemini API Key
# -----------------------------
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Missing Gemini API key. Please set GEMINI_API_KEY in secrets or env vars.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# 🔗 Helpers: build review URL
# -----------------------------
def build_review_url(product_url: str) -> str:
    """
    Convert GSMArena product URL into its reviews URL.
    Example:
      input:  https://www.gsmarena.com/samsung_galaxy_s24_fe-13262.php
      output: https://www.gsmarena.com/samsung_galaxy_s24_fe-reviews-13262.php
    """
    if not product_url or not product_url.endswith(".php"):
        return None
    try:
        base, phone_id = product_url.rsplit("-", 1)
        phone_id = phone_id.replace(".php", "")
        return f"{base}-reviews-{phone_id}.php"
    except Exception:
        return None

# -----------------------------
# 🔎 Resolve GSMArena product URL
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="🔗 Finding GSMArena page...")
def resolve_gsmarena_url(product_name: str):
    """
    Search GSMArena for a product and return (product_url, review_url).
    """
    try:
        query = product_name.replace(" ", "+")
        search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={query}"
        headers = {"User-Agent": "Mozilla/5.0"}

        r = requests.get(search_url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Try common selectors
        link = soup.select_one(".makers a") or soup.select_one(".makers li a") or soup.select_one(".section-body .makers a")
        if not link:
            # as last resort try first php link on page
            link = soup.select_one("a[href*='.php']")

        if not link or "href" not in link.attrs:
            return None, None

        product_url = "https://www.gsmarena.com/" + link["href"]
        review_url = build_review_url(product_url)
        return product_url, review_url

    except Exception as e:
        st.warning(f"⚠️ GSMArena search failed: {e}")
        return None, None

# -----------------------------
# 📊 Fetch GSMArena specs
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="📊 Fetching specs...")
def fetch_gsmarena_specs(url: str):
    """
    Scrape phone specs from GSMArena and normalize main fields.
    Returns dict with keys like Display, Processor, RAM, Storage, Camera, Battery, OS.
    """
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
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Candidate containers/selectors
        spec_containers = [
            ".article-info table tr",
            "#specs-list table tr",
            "table.specs tr",
            ".specs-brief-accent tr",
        ]

        spec_rows = []
        for sel in spec_containers:
            spec_rows = soup.select(sel)
            if spec_rows:
                break

        for row in spec_rows:
            th = row.find("td", class_="ttl") or row.find("th") or row.find("td", class_="spec-title")
            td = row.find("td", class_="nfo") or (row.find_all("td")[-1] if row.find_all("td") else None)
            if not th or not td:
                continue
            key = th.get_text(strip=True)
            val = td.get_text(" ", strip=True)

            # Map to canonical fields
            for field, keywords in key_map.items():
                if any(k.lower() in key.lower() for k in keywords):
                    if field in ["RAM", "Storage"]:
                        # extract GB numbers (handles "8GB/256GB" and similar)
                        matches = re.findall(r"(\d+)\s*GB", val, flags=re.IGNORECASE)
                        if matches:
                            if field == "RAM":
                                specs["RAM"] = f"{matches[0]}GB RAM"
                                # if multiple numbers present, next is probably storage
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

    # Ensure keys exist (fill with 'Not specified' if missing)
    for k in ["Display", "Processor", "RAM", "Storage", "Camera", "Battery", "OS"]:
        specs.setdefault(k, "Not specified")
    return specs

# -----------------------------
# 💬 Fetch GSMArena reviews (pagination-aware)
# -----------------------------
@st.cache_data(ttl=21600, show_spinner="💬 Fetching GSMArena reviews...")
def fetch_gsmarena_reviews(url: str, limit: int = 50):
    """
    Fetch user reviews from GSMArena review pages.
    Automatically follows pagination via 'next' link until 'limit' reviews are collected.
    """
    reviews = []
    if not url:
        return reviews

    headers = {"User-Agent": "Mozilla/5.0"}
    page_url = url

    try:
        while page_url and len(reviews) < limit:
            time.sleep(1)  # be polite
            r = requests.get(page_url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Try common review selectors (desktop & mobile)
            review_blocks = []
            # check for opinion blocks
            review_blocks.extend(soup.select(".opin"))
            review_blocks.extend(soup.select(".user-opinion"))
            review_blocks.extend(soup.select(".uopin"))
            review_blocks.extend(soup.select(".user-review"))
            review_blocks.extend(soup.select(".review-item"))
            # fallback to paragraphs in comments area
            if not review_blocks:
                review_blocks = soup.select("#user-comments p, .thread p, .user-thread p, .post p, p")

            for block in review_blocks:
                text = block.get_text(" ", strip=True)
                if 30 < len(text) < 1000:
                    low_text = text.lower()
                    if not any(w in low_text for w in ["gsmarena", "admin", "moderator", "delete", "report"]):
                        reviews.append(text)
                        if len(reviews) >= limit:
                            break

            # Look for pagination 'next' link
            next_link = soup.select_one("a.pages-next") or soup.find("a", string=re.compile(r"next", re.I))
            if next_link and next_link.has_attr("href"):
                href = next_link["href"]
                # sometimes href is absolute or relative
                if href.startswith("http"):
                    page_url = href
                else:
                    page_url = "https://www.gsmarena.com/" + href.lstrip("/")
            else:
                # try another pagination pattern: page param
                # if url already has ?page=, increment it, else stop
                m = re.search(r"[?&]page=(\d+)", page_url)
                if m:
                    current = int(m.group(1))
                    page_url = re.sub(r"([?&]page=)\d+", r"\g<1>%d" % (current + 1), page_url)
                else:
                    # No next link and no page param -> stop
                    page_url = None

    except Exception as e:
        st.warning(f"⚠️ GSMArena reviews fetch failed: {e}")

    return reviews[:limit]

# -----------------------------
# 🤖 Summarize with Gemini (safe JSON)
# -----------------------------
@st.cache_data(ttl=43200, show_spinner="🤖 Summarizing with Gemini...")
def summarize_reviews(product_name: str, specs: dict, reviews: list):
    """
    Ask Gemini to produce a structured JSON summary combining specs and reviews.
    Returns string (JSON text) or None on failure.
    """
    try:
        specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
        reviews_context = "\n".join([f"- {r[:200]}..." if len(r) > 200 else f"- {r}" for r in reviews[:20]]) if reviews else "No reviews found"

        prompt = f"""
You are an AI Review Summarizer analyzing the {product_name}.
Combine GSMArena official specs with real user reviews.

SPECS:
{specs_context}

REVIEWS:
{reviews_context}

Return valid JSON ONLY. Structure:
{{
  "verdict": "Brief rating (max 30 chars)",
  "pros": ["..."],
  "cons": ["..."],
  "aspect_sentiments": [
    {{"Aspect": "Camera", "Positive": 75, "Negative": 25}},
    {{"Aspect": "Battery", "Positive": 80, "Negative": 20}},
    {{"Aspect": "Performance", "Positive": 70, "Negative": 30}},
    {{"Aspect": "Display", "Positive": 85, "Negative": 15}},
    {{"Aspect": "Build Quality", "Positive": 65, "Negative": 35}}
  ],
  "user_quotes": ["quote1", "quote2", "quote3"],
  "recommendation": "Target audience (max 35 chars)",
  "bottom_line": "2-3 sentence final summary combining specs and user feedback",
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

        # Request JSON mime type; model may still include formatting—will clean after receiving.
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        text = response.text.strip()

        # Remove triple-backtick fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

        return text

    except Exception as e:
        st.error(f"⚠️ Gemini API error: {e}")
        return None

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Review Engine", page_icon="📱", layout="wide")

# Custom CSS metric-card (keeps it compact)
st.markdown("""
<style>
.metric-card { background-color: #0b0b0b; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; color: white; }
.metric-title-box { color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; display: inline-block; margin-bottom: 0.5rem; }
.metric-verdict .metric-title-box { background-color: #ff6b35; }
.metric-best-for .metric-title-box { background-color: #1f77b4; }
.metric-data-found .metric-title-box { background-color: #28a745; }
.metric-card p { word-wrap: break-word; overflow-wrap: break-word; margin: 0; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📱 AI-Powered Phone Review Engine")
st.markdown("Get a **comprehensive analysis** combining GSMArena specs with real user reviews.")

# Sidebar
st.sidebar.header("⚙️ Settings")
review_limit = st.sidebar.slider("Max reviews to analyze", 10, 200, 50, step=10)
include_raw = st.sidebar.checkbox("Show raw AI output", value=False)

# Input
phone = st.text_input("Enter phone name", value="Samsung Galaxy S24")
analyze_btn = st.button("🔍 Analyze Phone", type="primary")

# Main flow
if analyze_btn and phone:
    progress = st.progress(0)
    status = st.empty()

    status.text("🔎 Resolving GSMArena URLs...")
    product_url, review_url = resolve_gsmarena_url(phone)
    progress.progress(10)

    if not product_url:
        st.error(f"❌ Could not find '{phone}' on GSMArena. Try a more specific model name.")
        status.empty()
        progress.empty()
        st.stop()

    st.success(f"✅ Found product page: {product_url}")
    if review_url:
        st.success(f"✅ Using reviews page: {review_url}")
    else:
        st.warning("⚠️ Could not construct review URL automatically; proceeding with specs only.")

    status.text("📊 Fetching specs...")
    specs = fetch_gsmarena_specs(product_url)
    progress.progress(30)

    status.text("💬 Fetching user reviews (may take a few seconds)...")
    reviews = fetch_gsmarena_reviews(review_url, limit=review_limit) if review_url else []
    progress.progress(60)
    if reviews:
        st.success(f"✅ Collected {len(reviews)} reviews")
    else:
        st.info("ℹ️ No reviews collected; analysis will rely mainly on specs.")

    status.text("🤖 Generating AI summary...")
    summary_text = summarize_reviews(phone, specs, reviews)
    progress.progress(90)

    if not summary_text:
        st.error("⚠️ Failed to generate AI summary. See logs or try again.")
        status.empty()
        progress.empty()
        st.stop()

    # Try to parse JSON safely
    try:
        summary = json.loads(summary_text)
    except json.JSONDecodeError:
        # Some models include trailing text — try to extract JSON object via regex
        try:
            match = re.search(r"\{(?:.*\n*)*\}", summary_text)
            if match:
                summary = json.loads(match.group(0))
            else:
                raise
        except Exception as e:
            st.error("⚠️ Could not parse AI JSON output.")
            if include_raw:
                with st.expander("Raw AI Output"):
                    st.text_area("AI Output", summary_text, height=300)
            status.empty()
            progress.empty()
            st.stop()

    progress.progress(100)
    status.empty()
    progress.empty()

    # Display results
    st.markdown("---")
    st.subheader(f"📝 Analysis: {phone}")

    # Metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-verdict">
            <span class="metric-title-box">⭐ Verdict</span>
            <p style="font-size:1.1rem;font-weight:600;">{summary.get('verdict','N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-best-for">
            <span class="metric-title-box">🎯 Best For</span>
            <p style="font-size:1.1rem;font-weight:600;">{summary.get('recommendation','N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-data-found">
            <span class="metric-title-box">📊 Data Found</span>
            <p style="font-size:1.1rem;font-weight:600;">{len(specs)} specs, {len(reviews)} reviews</p>
        </div>
        """, unsafe_allow_html=True)

    # Bottom line
    st.markdown("### 🎯 Bottom Line")
    st.info(summary.get("bottom_line", "No summary available."))

    # Main two-column content
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        if "phone_specs" in summary and summary["phone_specs"]:
            st.markdown("### 🔧 Technical Specifications")
            spec_df = pd.DataFrame(list(summary["phone_specs"].items()), columns=["Component", "Details"])
            st.table(spec_df)

        if "aspect_sentiments" in summary and summary["aspect_sentiments"]:
            st.markdown("### 📊 User Sentiment Analysis")
            df_aspects = pd.DataFrame(summary["aspect_sentiments"])
            if not df_aspects.empty and "Aspect" in df_aspects.columns:
                df_chart = df_aspects.set_index("Aspect")[["Positive", "Negative"]]
                st.bar_chart(df_chart, height=320)

    with c2:
        st.markdown("### ✅ Strengths")
        for p in summary.get("pros", []):
            st.success(f"✓ {p}")

        st.markdown("### ⚠️ Weaknesses")
        for con in summary.get("cons", []):
            st.error(f"✗ {con}")

    # Quotes
    if summary.get("user_quotes"):
        st.markdown("### 💬 What Users Are Saying")
        for i, q in enumerate(summary.get("user_quotes", []), 1):
            st.info(f"**User {i}:** {q}")

    # Footer / Downloads
    st.markdown("---")
    st.markdown("**Data Sources:** GSMArena specifications & user reviews | **AI Analysis:** Google Gemini")
    st.download_button("📥 Download JSON", json.dumps(summary, indent=2), f"{phone.replace(' ', '_')}_analysis.json", "application/json")

    if include_raw:
        with st.expander("Raw AI Output"):
            st.text_area("AI Output", summary_text, height=250)