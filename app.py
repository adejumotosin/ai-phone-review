# app.py
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
    """Search GSMArena for a product and return its base + reviews URL"""
    try:
        query = product_name.replace(" ", "+")
        search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={query}"
        r = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        # ✅ get first result link
        # Based on your image, this selector is still correct.
        link = soup.select_one(".makers a")
        if not link:
            return None, None

        product_url = "https://www.gsmarena.com/" + link["href"]

        # ✅ build reviews URL
        review_url = product_url.replace(".php", "-reviews.php")
        return product_url, review_url
    except Exception as e:
        st.warning(f"⚠️ GSMArena search failed: {e}")
        return None, None

# -----------------------------
# 3️⃣ Focused Specs Scraper
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="📊 Fetching specs...")
def fetch_gsmarena_specs(url):
    specs = {}
    key_map = {
        "Display": ["Display"],
        "Processor": ["Chipset"],
        "RAM": ["Internal"],
        "Storage": ["Internal"],
        "Camera": ["Main Camera", "Triple", "Quad", "Dual"],
        "Battery": ["Battery"],
        "OS": ["OS"]
    }

    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        # ✅ PATCHED: Use the new container .article-info and find the table rows.
        spec_table = soup.select(".article-info table tr")
        
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
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        # ✅ PATCHED: Use the new ID for the main comment container.
        review_container = soup.find(id="user-comments")
        if not review_container:
            return reviews
        
        # Now, find the individual reviews within that container. The class might still be 'opin'.
        review_blocks = review_container.find_all("div", class_="opin")
        if not review_blocks:
            # Fallback for a common alternative structure
            review_blocks = review_container.find_all("p") 

        for block in review_blocks[:limit]:
            reviews.append(block.get_text(strip=True))
            
    except Exception as e:
        st.warning(f"⚠️ GSMArena reviews fetch failed: {e}")
    return reviews
    
# -----------------------------
# 5️⃣ Summarizer with Gemini (spec completion)
# -----------------------------
@st.cache_data(ttl=43200, show_spinner="🤖 Summarizing with Gemini...")
def summarize_reviews(product_name, specs, reviews):
    reviews_text = "\n".join(reviews[:50])
    reviews_hash = hashlib.md5((reviews_text + str(specs)).encode("utf-8")).heigest()

    specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
    reviews_context = "\n".join([f"- {r}" for r in reviews[:20]]) if reviews else "No reviews found"

    prompt = f"""
    You are an AI Review Summarizer.
    Combine **GSMArena official specs** with **real user reviews** for {product_name}.

    Specs:
    {specs_context}

    User Reviews:
    {reviews_context}

    ⚠️ IMPORTANT:
    - Always include all 7 spec fields: Display, Processor, RAM, Storage, Camera, Battery, OS.
    - If missing, infer conservatively or set to "N/A".
    - Return ONLY valid JSON (no markdown, no commentary).

    Output JSON:
    {{
      "verdict": "...",
      "pros": ["..."],
      "cons": ["..."],
      "aspect_sentiments": [
        {{"Aspect": "Camera", "Positive": 80, "Negative": 20}},
        {{"Aspect": "Battery", "Positive": 70, "Negative": 30}}
      ],
      "user_quotes": ["short real quotes from reviews"],
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

    return response.text, reviews_hash

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
    st.info(f"🔎 Collecting GSMArena specs and reviews for {phone}...")

    # Resolve URLs
    product_url, review_url = resolve_gsmarena_url(phone)

    # Fetch specs + reviews
    specs = fetch_gsmarena_specs(product_url) if product_url else {}
    reviews = fetch_gsmarena_reviews(review_url, limit=20) if review_url else []

    if not reviews and not specs:
        st.error("❌ No GSMArena data found. Try another phone.")
        st.stop()

    st.success(f"✅ Fetched {len(specs)} specs and {len(reviews)} reviews")

    # Summarize
    with st.spinner("🤖 Summarizing with Gemini..."):
        try:
            summary_text, _ = summarize_reviews(phone, specs, reviews)
            summary_data = json.loads(summary_text)
        except Exception as e:
            st.error(f"⚠️ Error parsing summary: {e}")
            st.text_area("Raw Output", summary_text, height=300)
            st.stop()

    # Display results
    st.markdown("---")
    st.subheader(f"📝 Review Summary for {phone}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("⭐ Verdict", summary_data.get("verdict", "N/A"))
    with col2:
        st.metric("🎯 Recommendation", summary_data.get("recommendation", "N/A"))

    st.markdown("### 📋 Bottom Line")
    st.info(summary_data.get("bottom_line", "No summary provided."))

    # Specs Table
    if "phone_specs" in summary_data:
        st.markdown("### 🔧 Key Phone Specifications")
        specs_df = pd.DataFrame(summary_data["phone_specs"].items(), columns=["Spec", "Details"])
        st.table(specs_df)

    # Pros & Cons
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Pros")
        for pro in summary_data.get("pros", []):
            st.write(f"- {pro}")
    with col2:
        st.markdown("### ⚠️ Cons")
        for con in summary_data.get("cons", []):
            st.write(f"- {con}")

    # Aspect Sentiments
    if "aspect_sentiments" in summary_data:
        st.markdown("### 📊 Aspect Sentiments")
        df = pd.DataFrame(summary_data["aspect_sentiments"])
        if "Aspect" in df.columns:
            st.bar_chart(df.set_index("Aspect")[["Positive", "Negative"]])

    # User Quotes
    st.markdown("### 👥 Real User Quotes")
    for i, quote in enumerate(summary_data.get("user_quotes", []), 1):
        st.info(f"💬 {quote}")
