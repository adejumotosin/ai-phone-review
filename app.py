import os
import re
import json
import hashlib
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from google import genai

# -----------------------------
# 1️⃣ Configure Gemini API Key
# -----------------------------
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Missing Gemini API key. Please set GEMINI_API_KEY in secrets or env vars.")
    st.stop()

client = genai.Client(api_key=api_key)

# -----------------------------
# 2️⃣ GSMArena URL Resolver
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="🔗 Finding GSMArena page...")
def resolve_gsmarena_url(product_name):
    try:
        query = product_name.replace(" ", "+")
        search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(search_url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        link = soup.select_one(".makers a")
        if not link:
            return None, None

        product_url = "https://www.gsmarena.com/" + link["href"]
        review_url = product_url.replace(".php", "-reviews.php")
        return product_url, review_url
    except Exception as e:
        st.warning(f"⚠️ GSMArena search failed: {e}")
        return None, None

# -----------------------------
# 3️⃣ Specs Scraper
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="📊 Fetching specs...")
def fetch_gsmarena_specs(url):
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

        spec_table = soup.select(".specs-list tr")
        for row in spec_table:
            th = row.find("td", class_="ttl")
            td = row.find("td", class_="nfo")
            if not th or not td:
                continue
            key = th.get_text(strip=True)
            val = td.get_text(" ", strip=True)

            for field, keywords in key_map.items():
                if any(k.lower() in key.lower() for k in keywords):
                    if field in ["RAM", "Storage"] and "GB" in val:
                        parts = val.split(",")
                        ram = [p for p in parts if "RAM" in p or re.search(r"\d+GB", p)]
                        storage = [p for p in parts if "ROM" in p or "storage" in p.lower()]
                        if field == "RAM" and ram:
                            specs["RAM"] = ram[0].strip()
                        elif field == "Storage" and storage:
                            specs["Storage"] = storage[0].strip()
                        else:
                            specs[field] = val
                    else:
                        specs[field] = val
    except Exception as e:
        st.warning(f"⚠️ GSMArena specs fetch failed: {e}")
    return specs

# -----------------------------
# 4️⃣ Reviews Scraper
# -----------------------------
@st.cache_data(ttl=21600, show_spinner="💬 Fetching GSMArena reviews...")
def fetch_gsmarena_reviews(url, limit=20):
    reviews = []
    if not url:
        return reviews
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        review_blocks = soup.select(".user-thread .opin")
        for block in review_blocks[:limit]:
            reviews.append(block.get_text(strip=True))
    except Exception as e:
        st.warning(f"⚠️ GSMArena reviews fetch failed: {e}")
    return reviews

# -----------------------------
# 5️⃣ Summarizer
# -----------------------------
@st.cache_data(ttl=43200, show_spinner="🤖 Summarizing with Gemini...")
def summarize_reviews(product_name, specs, reviews):
    reviews_text = "\n".join(reviews[:50])
    specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
    reviews_context = "\n".join([f"- {r}" for r in reviews[:20]]) if reviews else "No reviews found"

    prompt = f"""
    You are an AI Review Summarizer.
    Combine **GSMArena official specs** with **real user reviews** for {product_name}.

    Specs:
    {specs_context}

    User Reviews:
    {reviews_context}

    ⚠️ Always include all 7 spec fields: Display, Processor, RAM, Storage, Camera, Battery, OS.
    If missing, infer or set "N/A".
    Return ONLY valid JSON.

    JSON:
    {{
      "verdict": "...",
      "pros": ["..."],
      "cons": ["..."],
      "aspect_sentiments": [
        {{"Aspect": "Camera", "Positive": 80, "Negative": 20}}
      ],
      "user_quotes": ["..."],
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

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    return response.text

# -----------------------------
# 6️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Review Engine", page_icon="📱", layout="wide")
st.title("📱 AI-Powered Phone Review Engine (GSMArena Specs + Reviews)")

col1, col2 = st.columns([0.7, 0.3])
with col1:
    phone = st.text_input("Enter a phone name", "Samsung Galaxy A17")
with col2:
    generate_button = st.button("Get Review", type="primary", use_container_width=True)

if generate_button and phone:
    product_url, review_url = resolve_gsmarena_url(phone)
    specs = fetch_gsmarena_specs(product_url) if product_url else {}
    reviews = fetch_gsmarena_reviews(review_url, limit=20) if review_url else []

    if not reviews and not specs:
        st.error("❌ No GSMArena data found. Try another phone.")
        st.stop()

    try:
        summary_text = summarize_reviews(phone, specs, reviews)
        summary_data = json.loads(summary_text)
    except Exception as e:
        st.error(f"⚠️ Error parsing summary: {e}")
        st.text_area("Raw Output", summary_text, height=300)
        st.stop()

    # Cards for Verdict, Best For, Data Found
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 15px; border-radius: 10px;
                    border-left: 5px solid #4caf50; margin-bottom: 10px">
            <h4>⭐ Verdict</h4>
            <p>{summary_data.get("verdict", "N/A")}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 15px; border-radius: 10px;
                    border-left: 5px solid #2196f3; margin-bottom: 10px">
            <h4>🎯 Best For</h4>
            <p>{summary_data.get("recommendation", "N/A")}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 15px; border-radius: 10px;
                    border-left: 5px solid #ff9800; margin-bottom: 10px">
            <h4>📊 Data Found</h4>
            <p>{len(specs)} specs, {len(reviews)} reviews</p>
        </div>
        """, unsafe_allow_html=True)

    # Bottom line
    st.markdown("### 🎯 Bottom Line")
    st.info(summary_data.get("bottom_line", "No summary available."))

    # Two-column layout
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        if "phone_specs" in summary_data:
            st.markdown("### 🔧 Technical Specifications")
            specs_df = pd.DataFrame(summary_data["phone_specs"].items(), columns=["Component", "Details"])
            st.table(specs_df)

        if "aspect_sentiments" in summary_data:
            st.markdown("### 📊 User Sentiment Analysis")
            df = pd.DataFrame(summary_data["aspect_sentiments"])
            if not df.empty and "Aspect" in df.columns:
                st.bar_chart(df.set_index("Aspect")[["Positive", "Negative"]])
    with col2:
        st.markdown("### ✅ Strengths")
        for pro in summary_data.get("pros", []):
            st.success(f"✓ {pro}")

        st.markdown("### ⚠️ Weaknesses")
        for con in summary_data.get("cons", []):
            st.error(f"✗ {con}")

    if summary_data.get("user_quotes"):
        st.markdown("### 💬 What Users Are Saying")
        for i, quote in enumerate(summary_data.get("user_quotes", []), 1):
            st.info(f"**User {i}:** {quote}")