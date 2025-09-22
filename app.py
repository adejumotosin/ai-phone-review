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

# ==============================
# 1️⃣ Configure Gemini API Key
# ==============================
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Missing Gemini API key. Please set GEMINI_API_KEY in secrets or env vars.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# ==============================
# 2️⃣ GSMArena URL Builder
# ==============================
def resolve_gsmarena_url(product_name: str) -> str:
    query = product_name.strip().lower().replace(" ", "_")
    hash_id = hashlib.md5(query.encode()).hexdigest()[:5]  # fallback hash
    return f"https://www.gsmarena.com/{query}-reviews-{hash_id}.php"

# ==============================
# 3️⃣ Review Scraper with Pagination
# ==============================
def fetch_gsmarena_reviews(url: str, limit: int = 1000):
    """
    Fetch reviews from GSMArena, following pagination (?p=2, ?p=3, ...).
    Returns up to `limit` reviews as a list of strings.
    """
    reviews = []
    if not url:
        return reviews

    headers = {"User-Agent": "Mozilla/5.0"}
    page = 1

    try:
        while len(reviews) < limit:
            page_url = url if page == 1 else f"{url}?p={page}"
            r = requests.get(page_url, headers=headers, timeout=10)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.text, "html.parser")

            # Detect review blocks
            blocks = []
            blocks.extend(soup.select(".opin"))
            blocks.extend(soup.select(".user-opinion"))
            blocks.extend(soup.select(".uopin"))
            blocks.extend(soup.select(".user-review"))
            blocks.extend(soup.select(".review-item"))

            if not blocks:
                # fallback paragraphs inside comments
                blocks = soup.select("#user-comments p, .user-thread p, .post p")

            found = 0
            for blk in blocks:
                text = blk.get_text(" ", strip=True)
                if 30 < len(text) < 1200:
                    low = text.lower()
                    if not any(skip in low for skip in ["gsmarena", "admin", "moderator", "delete", "report"]):
                        reviews.append(text)
                        found += 1
                        if len(reviews) >= limit:
                            break

            if found == 0:
                # no more reviews on this page → stop
                break

            page += 1
            time.sleep(1)  # polite pause

    except Exception as e:
        st.warning(f"⚠️ GSMArena reviews fetch failed: {e}")

    return reviews[:limit]

# ==============================
# 4️⃣ Prompt Builder
# ==============================
def build_prompt(product_name, specs, reviews_subset):
    specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
    reviews_context = "\n".join([f"- {r}" for r in reviews_subset]) if reviews_subset else "No reviews found"

    return f"""
    You are an AI Review Summarizer analyzing the {product_name}.
    Combine GSMArena official specs with real user reviews to create a comprehensive analysis.

    OFFICIAL SPECS:
    {specs_context}

    USER REVIEWS SAMPLE:
    {reviews_context}

    OUTPUT RULES:
    - Return ONLY valid JSON, no markdown
    - Do not split words into characters
    - Use full sentences or short phrases only
    - Arrays like pros/cons must be JSON string lists, e.g. ["Great battery", "Smooth display"]

    JSON SCHEMA (fill all keys):
    {{
      "verdict": "string, short summary under 30 chars",
      "pros": ["list of positive aspects as full phrases"],
      "cons": ["list of negative aspects as full phrases"],
      "aspect_sentiments": [
        {{"aspect": "Camera", "sentiment": "positive", "detail": "clear photos"}},
        {{"aspect": "Battery", "sentiment": "negative", "detail": "drains quickly"}}
      ],
      "user_quotes": ["2-6 real user quotes (1-2 sentences each)"],
      "recommendation": "string under 35 chars",
      "bottom_line": "1-2 sentences wrap-up",
      "phone_specs": {{
        "Display": "string",
        "Processor": "string",
        "RAM": "string",
        "Storage": "string",
        "Camera": "string",
        "Battery": "string",
        "OS": "string"
      }}
    }}
    """

# ==============================
# 5️⃣ Summarizer with Chunking
# ==============================
def summarize_reviews(product_name, specs, reviews, chunk_size=200):
    summaries = []
    for i in range(0, len(reviews), chunk_size):
        chunk = reviews[i:i+chunk_size]
        prompt = build_prompt(product_name, specs, chunk)
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            summaries.append(text)
        except Exception as e:
            st.error(f"Gemini summarization failed: {e}")

    if not summaries:
        return None

    final_prompt = f"""
    Merge the following partial JSON analyses into one valid JSON object:
    {summaries}
    """

    try:
        response = model.generate_content(final_prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Final merge failed: {e}")
        return None

# ==============================
# 6️⃣ Streamlit App
# ==============================
st.set_page_config(page_title="AI Phone Review Summarizer", layout="wide")

st.title("📱 AI Phone Review Summarizer")

product_name = st.text_input("Enter phone model (e.g. Samsung Galaxy S24 FE)")

if product_name:
    review_url = resolve_gsmarena_url(product_name)
    reviews = fetch_gsmarena_reviews(review_url, limit=1000)

    st.info(f"Fetched {len(reviews)} reviews for {product_name}")

    summary_json = summarize_reviews(product_name, {}, reviews)

    if summary_json:
        try:
            summary = json.loads(summary_json)

            st.subheader(f"📊 AI Summary for {product_name}")

            st.metric("Verdict", summary.get("verdict", "N/A"))
            st.write("**Recommendation:**", summary.get("recommendation", "N/A"))

            col1, col2 = st.columns(2)
            with col1:
                st.write("✅ **Pros**")
                for p in summary.get("pros", []):
                    st.write(f"- {p}")
            with col2:
                st.write("⚠️ **Cons**")
                for c in summary.get("cons", []):
                    st.write(f"- {c}")

            st.write("💡 **User Quotes**")
            for q in summary.get("user_quotes", []):
                st.info(q)

            st.write("📌 **Bottom Line**")
            st.success(summary.get("bottom_line", "N/A"))

            with st.expander("📑 Full JSON (debug)"):
                st.json(summary)

        except Exception as e:
            st.error(f"❌ JSON parsing failed: {e}")
            st.text(summary_json)
    else:
        st.error("Failed to generate summary.")