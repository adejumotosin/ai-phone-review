# app.py
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
# 2️⃣ GSMArena URL Resolver
# ==============================
def resolve_gsmarena_url(product_name: str):
    """Search GSMArena for a product and return its base + reviews URL"""
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

        href = link["href"]
        if "-" in href and ".php" in href:
            phone_id = href.split("-")[-1].replace(".php", "")
            review_url = f"https://www.gsmarena.com/{href.replace('.php', '-reviews.php')}"
            return product_url, review_url

        return product_url, None

    except Exception as e:
        st.warning(f"⚠️ GSMArena search failed: {e}")
        return None, None

# ==============================
# 3️⃣ Specs Scraper
# ==============================
def fetch_gsmarena_specs(url: str):
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

        rows = soup.select("#specs-list table tr")
        for row in rows:
            th = row.find("td", class_="ttl")
            td = row.find("td", class_="nfo")
            if not th or not td:
                continue
            key = th.get_text(strip=True)
            val = td.get_text(" ", strip=True)

            for field, keywords in key_map.items():
                if any(k.lower() in key.lower() for k in keywords):
                    specs[field] = val
                    break

    except Exception as e:
        st.warning(f"⚠️ Specs fetch failed: {e}")

    return specs

# ==============================
# 4️⃣ Reviews Scraper with Pagination
# ==============================
def fetch_gsmarena_reviews(url: str, limit: int = 1000):
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

            blocks = []
            blocks.extend(soup.select(".opin"))
            blocks.extend(soup.select(".user-opinion"))
            blocks.extend(soup.select(".uopin"))
            blocks.extend(soup.select(".user-review"))
            blocks.extend(soup.select(".review-item"))

            if not blocks:
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
                break

            page += 1
            time.sleep(1)

    except Exception as e:
        st.warning(f"⚠️ Reviews fetch failed: {e}")

    return reviews[:limit]

# ==============================
# 5️⃣ Summarizer with Chunking
# ==============================
def build_prompt(product_name, specs, reviews_subset):
    specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
    reviews_context = "\n".join([f"- {r}" for r in reviews_subset]) if reviews_subset else "No reviews found"

    return f"""
    You are an AI Review Summarizer analyzing the {product_name}.
    Combine GSMArena official specs with real user reviews.

    SPECS:
    {specs_context}

    REVIEWS:
    {reviews_context}

    RULES:
    - Output ONLY valid JSON
    - No markdown, no commentary
    - Keep verdict <30 chars
    - Keep recommendation <35 chars

    JSON SCHEMA:
    {{
      "verdict": "string",
      "pros": ["list of pros"],
      "cons": ["list of cons"],
      "aspect_sentiments": [
        {{"Aspect": "Camera", "Positive": 70, "Negative": 30}},
        {{"Aspect": "Battery", "Positive": 80, "Negative": 20}}
      ],
      "user_quotes": ["2-6 real quotes"],
      "recommendation": "string",
      "bottom_line": "string",
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

def summarize_reviews(product_name, specs, reviews, chunk_size=200):
    summaries = []
    for i in range(0, len(reviews), chunk_size):
        chunk = reviews[i:i+chunk_size]
        prompt = build_prompt(product_name, specs, chunk)
        try:
            response = model.generate_content(prompt)
            summaries.append(response.text.strip())
        except Exception as e:
            st.error(f"Gemini summarization failed: {e}")

    if not summaries:
        return None

    final_prompt = f"""
    Merge these partial JSON objects into one valid JSON:
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
st.set_page_config(page_title="AI Phone Review Engine", layout="wide")

st.title("📱 AI-Powered Phone Review Engine")

phone = st.text_input("Enter phone model (e.g. Samsung Galaxy S24 FE)")

if phone:
    product_url, review_url = resolve_gsmarena_url(phone)

    if not product_url:
        st.error("❌ Could not find this phone on GSMArena.")
        st.stop()

    specs = fetch_gsmarena_specs(product_url)
    reviews = fetch_gsmarena_reviews(review_url, limit=1000)

    st.success(f"✅ Found {len(specs)} specs and {len(reviews)} reviews")

    summary_json = summarize_reviews(phone, specs, reviews)

    if summary_json:
        try:
            summary = json.loads(summary_json)

            st.subheader(f"📊 AI Summary for {phone}")
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

            if "aspect_sentiments" in summary:
                df = pd.DataFrame(summary["aspect_sentiments"])
                if not df.empty and "Aspect" in df.columns:
                    chart_data = df.set_index("Aspect")[["Positive", "Negative"]]
                    st.bar_chart(chart_data, height=300)

            with st.expander("📑 Full JSON (debug)"):
                st.json(summary)

        except Exception as e:
            st.error(f"❌ JSON parsing failed: {e}")
            st.text(summary_json)
    else:
        st.error("Failed to generate summary.")