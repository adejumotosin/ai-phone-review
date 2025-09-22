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
# 🔗 Resolve GSMArena URL
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="🔗 Finding GSMArena page...")
def resolve_gsmarena_url(product_name: str):
    """
    Search GSMArena for a product and return:
      - Product page URL
      - Reviews page URL (if available)
    """
    try:
        query = product_name.replace(" ", "+")
        search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.124 Safari/537.36"
        }

        r = requests.get(search_url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Flexible selectors
        selectors = [".makers a", ".makers li a", ".section-body .makers a", "a[href*='.php']"]
        link = None
        for selector in selectors:
            link = soup.select_one(selector)
            if link:
                break
        if not link:
            return None, None

        product_url = "https://www.gsmarena.com/" + link["href"]

        # Extract phone ID
        href = link["href"]
        if "-" in href and ".php" in href:
            phone_id = href.split("-")[-1].replace(".php", "")
            phone_name = href.replace(f"-{phone_id}.php", "").replace("_", " ")

            # Candidate review URLs
            review_urls = [
                f"https://www.gsmarena.com/{href.replace('.php', '-reviews.php')}",
                f"https://m.gsmarena.com/{href.replace('_', ' ').replace('.php', '')}-reviews-{phone_id}.php",
                f"https://www.gsmarena.com/reviews.php3?idPhone={phone_id}",
                f"https://m.gsmarena.com/{phone_name.replace(' ', '_')}-reviews-{phone_id}.php",
            ]

            # Pick first valid one
            for review_url in review_urls:
                try:
                    resp = requests.get(review_url, headers=headers, stream=True, timeout=5)
                    if resp.status_code == 200:
                        return product_url, review_url
                except Exception:
                    continue

        return product_url, product_url.replace(".php", "-reviews.php")

    except Exception as e:
        st.warning(f"⚠️ GSMArena search failed: {e}")
        return None, None


# -----------------------------
# 📊 Fetch GSMArena Specs
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="📊 Fetching specs...")
def fetch_gsmarena_specs(url: str):
    """
    Scrape phone specs from GSMArena.
    Returns a dictionary with cleaned fields.
    """
    specs = {}
    key_map = {
        "Display": ["Display", "Screen", "Size"],
        "Processor": ["Chipset", "CPU", "Processor", "SoC"],
        "RAM": ["Internal", "Memory", "RAM"],
        "Storage": ["Internal", "Storage", "Memory"],
        "Camera": ["Main Camera", "Triple", "Quad", "Dual", "Camera"],
        "Battery": ["Battery"],
        "OS": ["OS", "Android", "iOS"],
    }

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Candidate containers
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

            for field, keywords in key_map.items():
                if any(k.lower() in key.lower() for k in keywords):
                    if field in ["RAM", "Storage"]:
                        matches = re.findall(r"(\d+)\s*GB", val)
                        if matches:
                            if field == "RAM":
                                specs["RAM"] = f"{matches[0]}GB RAM"
                            elif field == "Storage":
                                specs["Storage"] = f"{matches[-1]}GB Storage"
                        else:
                            specs[field] = val
                    else:
                        specs[field] = val
                    break

    except Exception as e:
        st.warning(f"⚠️ GSMArena specs fetch failed: {e}")

    return specs


# -----------------------------
# 💬 Fetch GSMArena Reviews
# -----------------------------
@st.cache_data(ttl=21600, show_spinner="💬 Fetching GSMArena reviews...")
def fetch_gsmarena_reviews(url: str, limit: int = 20):
    """
    Scrape user reviews from GSMArena (desktop & mobile).
    Includes pagination (first 3 pages).
    """
    reviews = []
    if not url:
        return reviews

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        for page in range(1, 4):  # First 3 pages
            page_url = f"{url}?page={page}" if page > 1 else url
            r = requests.get(page_url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Desktop selectors
            review_blocks = soup.select(".opin, .user-review, .review-item, #user-comments p")
            for block in review_blocks:
                text = block.get_text(strip=True)
                if 35 < len(text) < 1000:
                    if not any(w in text.lower() for w in ["gsmarena", "admin", "moderator"]):
                        reviews.append(text)
                        if len(reviews) >= limit:
                            return reviews
            time.sleep(1)

    except Exception as e:
        st.warning(f"⚠️ GSMArena reviews fetch failed: {e}")

    return reviews[:limit]


# -----------------------------
# 🤖 Summarize with Gemini
# -----------------------------
@st.cache_data(ttl=43200, show_spinner="🤖 Summarizing with Gemini...")
def summarize_reviews(product_name, specs, reviews):
    """
    Use Gemini to summarize reviews + specs into structured JSON.
    """
    try:
        specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
        reviews_context = "\n".join([f"- {r[:200]}..." if len(r) > 200 else f"- {r}" for r in reviews[:20]]) or "No reviews found"

        prompt = f"""
        You are an AI Review Summarizer analyzing the {product_name}.
        Combine GSMArena official specs with real user reviews.

        SPECS:
        {specs_context}

        REVIEWS:
        {reviews_context}

        Return valid JSON with this structure:
        {{
          "verdict": "...",
          "pros": ["..."],
          "cons": ["..."],
          "aspect_sentiments": [
            {{"Aspect": "Camera", "Positive": 75, "Negative": 25}},
            {{"Aspect": "Battery", "Positive": 80, "Negative": 20}}
          ],
          "user_quotes": ["...", "..."],
          "recommendation": "...",
          "bottom_line": "...",
          "phone_specs": {{
            "Display": "...",
            "Processor": "...",
            "RAM": "...",
            "Storage": "...",
            "Camera": "...",
            "Battery": "...",
            "OS": "..."
          }}
        }}
        """

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Cleanup if needed
        if text.startswith("```"):
            text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()

        return text

    except Exception as e:
        st.error(f"⚠️ Gemini API error: {e}")
        return None


# -----------------------------
# 🎨 Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Review Engine", page_icon="📱", layout="wide")

st.title("📱 AI-Powered Phone Review Engine")
st.markdown("Get a **comprehensive analysis** combining GSMArena specs with real user reviews.")


# Sidebar
st.sidebar.header("⚙️ Settings")
review_limit = st.sidebar.slider("Max reviews to analyze", 5, 50, 20)


# Input
phone = st.text_input("Enter phone name", value="Samsung Galaxy S24")
analyze_btn = st.button("🔍 Analyze Phone", type="primary", use_container_width=True)


# Main process
if analyze_btn and phone:
    progress = st.progress(0)

    st.info("🔎 Searching GSMArena database...")
    product_url, review_url = resolve_gsmarena_url(phone)
    if not product_url:
        st.error(f"❌ Could not find '{phone}' on GSMArena.")
        st.stop()

    progress.progress(20)
    st.success(f"✅ Found product page: {product_url}")

    st.info("📊 Extracting specifications...")
    specs = fetch_gsmarena_specs(product_url)
    progress.progress(40)

    st.info("💬 Collecting user reviews...")
    reviews = fetch_gsmarena_reviews(review_url, review_limit) if review_url else []
    progress.progress(60)

    st.info("🤖 Generating AI summary...")
    summary_text = summarize_reviews(phone, specs, reviews)
    progress.progress(100)

    if not summary_text:
        st.error("⚠️ Failed to generate summary.")
        st.stop()

    try:
        summary = json.loads(summary_text)
    except Exception:
        st.error("⚠️ Could not parse AI response.")
        with st.expander("Raw Output"):
            st.text_area("AI Output", summary_text, height=200)
        st.stop()

    # Results
    st.subheader(f"📝 Analysis: {phone}")

    # Metric cards
    col1, col2, col3 = st.columns(3)
    col1.metric("⭐ Verdict", summary.get("verdict", "N/A"))
    col2.metric("🎯 Best For", summary.get("recommendation", "N/A"))
    col3.metric("📊 Data", f"{len(specs)} specs, {len(reviews)} reviews")

    # Bottom line
    st.markdown("### 🎯 Bottom Line")
    st.info(summary.get("bottom_line", "No summary."))

    # Specs
    if "phone_specs" in summary:
        st.markdown("### 🔧 Specifications")
        df_specs = pd.DataFrame(summary["phone_specs"].items(), columns=["Component", "Details"])
        st.table(df_specs)

    # Sentiment chart
    if "aspect_sentiments" in summary:
        st.markdown("### 📊 User Sentiment")
        df_sent = pd.DataFrame(summary["aspect_sentiments"])
        if not df_sent.empty and "Aspect" in df_sent:
            st.bar_chart(df_sent.set_index("Aspect")[["Positive", "Negative"]])

    # Pros & Cons
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Strengths")
        for p in summary.get("pros", []):
            st.success(f"✓ {p}")
    with col2:
        st.markdown("### ⚠️ Weaknesses")
        for c in summary.get("cons", []):
            st.error(f"✗ {c}")

    # Quotes
    if summary.get("user_quotes"):
        st.markdown("### 💬 User Quotes")
        for i, q in enumerate(summary["user_quotes"], 1):
            st.info(f"**User {i}:** {q}")

    # Download
    st.download_button("⬇️ Download JSON", json.dumps(summary, indent=2), f"{phone}_analysis.json", "application/json")