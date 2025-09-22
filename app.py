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
# 1️⃣ GSMArena URL Resolver
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

        link = soup.select_one(".makers a") or soup.select_one(".makers li a")
        if not link:
            return None, None

        product_url = "https://www.gsmarena.com/" + link["href"]
        return product_url, build_review_url(product_url)
    except Exception as e:
        st.warning(f"⚠️ GSMArena search failed: {e}")
        return None, None

# -----------------------------
# 2️⃣ Build Review URL
# -----------------------------
def build_review_url(product_url: str) -> str:
    if not product_url or not product_url.endswith(".php"):
        return None
    base, phone_id = product_url.rsplit("-", 1)
    phone_id = phone_id.replace(".php", "")
    return f"{base}-reviews-{phone_id}.php"

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

        spec_rows = soup.select(".article-info table tr") or soup.select("#specs-list table tr")
        for row in spec_rows:
            th = row.find("td", class_="ttl") or row.find("th")
            td = row.find("td", class_="nfo") or (row.find_all("td")[-1] if row.find_all("td") else None)
            if not th or not td:
                continue
            key = th.get_text(strip=True)
            val = td.get_text(" ", strip=True)
            for field, keywords in key_map.items():
                if any(k.lower() in key.lower() for k in keywords):
                    if field in ["RAM", "Storage"] and "GB" in val:
                        gb_matches = re.findall(r'(\\d+)\\s*GB', val)
                        if gb_matches:
                            if field == "RAM":
                                specs["RAM"] = f"{gb_matches[0]}GB RAM"
                            elif field == "Storage":
                                specs["Storage"] = f"{gb_matches[-1]}GB Storage"
                    else:
                        specs[field] = val
                    break
    except Exception as e:
        st.warning(f"⚠️ GSMArena specs fetch failed: {e}")
    return specs

# -----------------------------
# 4️⃣ Reviews Scraper (pagination)
# -----------------------------
@st.cache_data(ttl=21600, show_spinner="💬 Fetching GSMArena reviews...")
def fetch_gsmarena_reviews(url: str, limit: int = 1000):
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

            review_blocks = soup.select(".opin, .user-opinion, .uopin, .user-review, .review-item")
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

            # Pagination
            next_link = soup.select_one("a.pages-next") or soup.find("a", string=re.compile(r"next", re.I))
            if next_link and next_link.has_attr("href"):
                href = next_link["href"]
                page_url = href if href.startswith("http") else "https://www.gsmarena.com/" + href.lstrip("/")
            else:
                page_url = None

    except Exception as e:
        st.warning(f"⚠️ GSMArena reviews fetch failed: {e}")

    return reviews[:limit]

# -----------------------------
# 5️⃣ Summarizer with Chunking
# -----------------------------
@st.cache_data(ttl=43200, show_spinner="🤖 Summarizing with Gemini...")
def summarize_reviews(product_name, specs, reviews, chunk_size=200):
    try:
        if not reviews:
            reviews = []

        def build_prompt(product_name, specs, reviews_subset):
            specs_context = "\\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
            reviews_context = "\\n".join([f"- {r}" for r in reviews_subset]) if reviews_subset else "No reviews found"
            return f"""
            You are an AI Review Summarizer analyzing the {product_name}.
            Combine GSMArena official specs with real user reviews to create a comprehensive analysis.

            OFFICIAL SPECS:
            {specs_context}

            USER REVIEWS SAMPLE:
            {reviews_context}

            CRITICAL REQUIREMENTS:
            - Always include all 7 spec fields: Display, Processor, RAM, Storage, Camera, Battery, OS
            - If missing, mark as "Not specified"
            - Verdict <30 chars, Recommendation <35 chars
            - Extract real user quotes (2-4 sentences max each)
            - Output ONLY valid JSON, no markdown

            Expected JSON keys: verdict, pros, cons, aspect_sentiments, user_quotes, recommendation, bottom_line, phone_specs
            """

        chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
        partial_summaries = []
        for chunk in chunks:
            prompt = build_prompt(product_name, specs, chunk)
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            try:
                partial_json = json.loads(response.text)
                partial_summaries.append(partial_json)
            except Exception:
                continue

        if not partial_summaries:
            return None, None

        merged = {
            "verdict": partial_summaries[0].get("verdict", "N/A"),
            "pros": list({p for s in partial_summaries for p in s.get("pros", [])}),
            "cons": list({c for s in partial_summaries for c in s.get("cons", [])}),
            "aspect_sentiments": partial_summaries[0].get("aspect_sentiments", []),
            "user_quotes": [q for s in partial_summaries for q in s.get("user_quotes", [])][:6],
            "recommendation": partial_summaries[0].get("recommendation", "N/A"),
            "bottom_line": " ".join(s.get("bottom_line", "") for s in partial_summaries)[:500],
            "phone_specs": specs or partial_summaries[0].get("phone_specs", {})
        }

        reviews_hash = hashlib.md5(("".join(reviews) + str(specs)).encode("utf-8")).hexdigest()
        return json.dumps(merged, ensure_ascii=False), reviews_hash

    except Exception as e:
        st.error(f"⚠️ Gemini API error: {e}")
        return None, None

# -----------------------------
# 6️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="📱 AI Phone Review Summarizer", layout="wide")
st.title("📱 AI Phone Review Summarizer")

product_name = st.text_input("Enter phone model (e.g. Samsung Galaxy S24 FE)")
review_limit = st.sidebar.slider("Max reviews to analyze", 50, 1000, 200, step=50)

if product_name:
    product_url, review_url = resolve_gsmarena_url(product_name)
    specs = fetch_gsmarena_specs(product_url) if product_url else {}
    reviews = fetch_gsmarena_reviews(review_url, limit=review_limit) if review_url else []

    summary_json, _ = summarize_reviews(product_name, specs, reviews)
    if summary_json:
        summary = json.loads(summary_json)
        st.subheader(f"📊 AI Summary for {product_name}")
        st.write(summary)
    else:
        st.error("Failed to generate summary.")