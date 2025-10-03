# app.py
import re
import time
import json
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from textblob import TextBlob
from urllib.parse import urljoin

# -----------------------------
# Helpers
# -----------------------------
def safe_load_json(text: str):
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
# GSMArena URL resolution
# -----------------------------
@st.cache_data(ttl=86400)
def resolve_gsmarena_url(product_name: str):
    try:
        query = product_name.strip()
        search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(search_url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        link = soup.select_one(".makers a") or soup.select_one(".makers li a") or soup.select_one("a[href*='.php']")
        if not link or not link.has_attr("href"):
            return None, None
        product_href = link["href"]
        product_url = "https://www.gsmarena.com/" + product_href
        review_url = build_review_url(product_url)
        return product_url, review_url
    except Exception:
        return None, None

def build_review_url(product_url: str) -> str:
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
@st.cache_data(ttl=86400)
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
    if not url:
        return {k: "Not specified" for k in key_map.keys()}
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
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
    except Exception:
        pass
    for k in ["Display", "Processor", "RAM", "Storage", "Camera", "Battery", "OS"]:
        specs.setdefault(k, "Not specified")
    return specs

# -----------------------------
# Reviews scraper
# -----------------------------
@st.cache_data(ttl=21600)
def fetch_gsmarena_reviews(url: str, limit: int = 1000):
    reviews = []
    if not url:
        return reviews
    headers = {"User-Agent": "Mozilla/5.0"}
    page_url = url
    visited_urls = set()
    page_count = 0
    max_pages = 50
    try:
        while page_url and len(reviews) < limit and page_count < max_pages:
            if page_url in visited_urls:
                break
            visited_urls.add(page_url)
            page_count += 1
            time.sleep(1)
            r = requests.get(page_url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            review_blocks = soup.select(".opin, .user-opinion, .uopin") or soup.select(".review-item, .review-content")
            for block in review_blocks:
                text = block.get_text(" ", strip=True)
                if 30 < len(text) < 1500:
                    reviews.append(text)
                    if len(reviews) >= limit:
                        break
            next_link = soup.find("a", string=re.compile(r"next", re.I))
            if next_link and next_link.has_attr("href"):
                href = next_link["href"]
                page_url = href if href.startswith("http") else urljoin("https://www.gsmarena.com/", href)
            else:
                break
    except Exception:
        pass
    return reviews[:limit]

# -----------------------------
# Sentiment analyzer
# -----------------------------
def analyze_review_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    label = "POSITIVE" if polarity >= 0 else "NEGATIVE"
    return label, polarity

# -----------------------------
# Aspect-specific pros/cons
# -----------------------------
ASPECT_KEYWORDS = {
    "Camera": ["camera", "photo", "picture", "selfie", "video", "lens"],
    "Battery": ["battery", "charge", "power", "last", "duration"],
    "Display": ["display", "screen", "resolution", "brightness", "touch"],
    "Performance": ["performance", "speed", "lag", "fast", "slow", "smooth"],
    "OS": ["os", "android", "ios", "software", "ui", "update"]
}

def extract_aspect_pros_cons(reviews, aspects=ASPECT_KEYWORDS):
    aspect_summary = {}
    for aspect, keywords in aspects.items():
        pros, cons = [], []
        for review in reviews:
            if any(k.lower() in review.lower() for k in keywords):
                label, _ = analyze_review_sentiment(review)
                if label == "POSITIVE":
                    pros.append(review)
                else:
                    cons.append(review)
        aspect_summary[aspect] = {"pros": pros[:5], "cons": cons[:5]}
    return aspect_summary

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Phone Review Engine", page_icon="📱", layout="wide")
st.title("📱 Phone Review Engine (Data Science Version)")

st.sidebar.header("⚙️ Settings")
review_limit = st.sidebar.slider("Max reviews to analyze", 50, 1000, 400, step=50)

phone = st.text_input("Enter phone name (e.g., Samsung Galaxy S24 FE)", value="Samsung Galaxy S24")
analyze = st.button("🔍 Analyze Phone")

if analyze and phone:
    status = st.empty()
    status.text("🔎 Resolving GSMArena product page...")
    product_url, review_url = resolve_gsmarena_url(phone)

    if not product_url:
        st.error(f"❌ Could not find '{phone}' on GSMArena.")
        st.stop()

    st.success(f"✅ Product page: {product_url}")
    if review_url:
        st.success(f"✅ Reviews page: {review_url}")
    else:
        st.warning("⚠️ Couldn't build reviews URL automatically. Proceeding with specs only.")

    status.text("📊 Fetching specs...")
    specs = fetch_gsmarena_specs(product_url)
    st.info(f"✅ Specs fetched.")

    status.text("💬 Collecting reviews...")
    reviews = fetch_gsmarena_reviews(review_url, limit=review_limit) if review_url else []
    st.info(f"✅ Collected {len(reviews)} reviews.")

    status.text("🔍 Extracting aspect-specific pros/cons...")
    aspect_pros_cons = extract_aspect_pros_cons(reviews)
    st.info(f"✅ Extracted pros/cons for {len(aspect_pros_cons)} aspects.")

    # Display results
    st.subheader("📊 Phone Specs")
    for k, v in specs.items():
        st.write(f"**{k}:** {v}")

    if reviews:
        st.subheader("💡 Aspect-specific Pros/Cons")
        for aspect, vals in aspect_pros_cons.items():
            st.write(f"### {aspect}")
            st.write("**Pros:**")
            for p in vals["pros"]:
                st.write(f"- {p}")
            st.write("**Cons:**")
            for c in vals["cons"]:
                st.write(f"- {c}")