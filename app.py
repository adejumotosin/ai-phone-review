import os
import re
import time
import json
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Summarization
try:
    from summa import summarizer
    SUMMA_AVAILABLE = True
except:
    SUMMA_AVAILABLE = False

# -----------------------------
# SUPABASE DB HELPERS
# -----------------------------
from supabase import create_client, Client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_reviews_to_db(phone_model: str, reviews: List[str]):
    if not reviews:
        return
    rows = [{"phone_name": phone_model, "review_text": r[:2000]} for r in reviews]
    supabase.table("reviews").insert(rows).execute()

@st.cache_data(ttl=3600)
def get_reviews_from_db(phone_model: str, limit: int = 500) -> List[str]:
    response = supabase.table("reviews") \
                       .select("review_text") \
                       .eq("phone_name", phone_model) \
                       .order("id", desc=True) \
                       .limit(limit) \
                       .execute()
    if response.data:
        return [r["review_text"] for r in response.data]
    return []

def clear_reviews_from_db(phone_model: str):
    supabase.table("reviews").delete().eq("phone_name", phone_model).execute()

# -----------------------------
# SCRAPER (GSMArena reviews)
# -----------------------------
def fetch_gsmarena_reviews(review_page_url: str, max_reviews: int = 200) -> List[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    reviews = []
    try:
        r = requests.get(review_page_url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        selectors = [".opin", ".user-opinion", ".uopin", ".review-content", ".opinion", ".user-review"]
        blocks = []
        for sel in selectors:
            found = soup.select(sel)
            if found:
                blocks = found
                break
        if not blocks:
            container = soup.select_one("#review")
            if container:
                blocks = container.find_all("p")
            else:
                blocks = soup.find_all("p")
        for b in blocks:
            text = b.get_text(" ", strip=True)
            if text and len(text) > 30:
                reviews.append(text)
                if len(reviews) >= max_reviews:
                    break
    except Exception as e:
        st.warning(f"Scrape failed: {e}")
    time.sleep(1)
    return reviews

# -----------------------------
# SENTIMENT MODEL
# -----------------------------
MODEL_PATH = "sentiment_model.joblib"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    X_train = [
        "Battery life is excellent, lasts all day",
        "Amazing camera and photos are crisp",
        "Screen is vibrant and bright",
        "Phone is snappy and fast",
        "Battery drains quickly, terrible",
        "Camera is awful, pictures are noisy",
        "Screen flickers and is dim",
        "Phone lags and freezes sometimes",
    ]
    y_train = ["POSITIVE", "POSITIVE", "POSITIVE", "POSITIVE",
               "NEGATIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE"]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

model_pipeline = load_or_train_model()

def predict_sentiment(texts: List[str]) -> List[Tuple[str, float]]:
    if not texts:
        return []
    preds = model_pipeline.predict(texts)
    probs = model_pipeline.predict_proba(texts)
    label_index = {lab: i for i, lab in enumerate(model_pipeline.classes_)}
    results = []
    for i, lab in enumerate(preds):
        prob = float(probs[i, label_index[lab]])
        results.append((lab, prob))
    return results

ASPECT_KEYWORDS = {
    "Battery": ["battery", "charge", "charging", "power", "life", "mah"],
    "Camera": ["camera", "photo", "picture", "video", "selfie", "lens"],
    "Display": ["display", "screen", "resolution", "brightness", "oled", "lcd"],
    "Performance": ["performance", "speed", "lag", "slow", "fps", "processor", "chip", "cpu"]
}

def aspect_based_sentiment(reviews: List[str]) -> Dict[str, float]:
    aspect_scores = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        hits = [r for r in reviews if any(k in r.lower() for k in keywords)]
        if not hits:
            aspect_scores[aspect] = 0.0
            continue
        preds = predict_sentiment(hits)
        mapped = []
        for lab, score in preds:
            if lab.upper().startswith("POS"):
                mapped.append(score)
            else:
                mapped.append(-score)
        aspect_scores[aspect] = sum(mapped) / len(mapped) if mapped else 0.0
    return aspect_scores

# -----------------------------
# SUMMARY
# -----------------------------
def summarize_reviews(reviews: List[str]) -> str:
    text = " ".join(reviews)
    if not text.strip():
        return ""
    if SUMMA_AVAILABLE:
        try:
            s = summarizer.summarize(text, ratio=0.03)
            if s and len(s) > 30:
                return s
        except:
            pass
    sentences = re.split(r'(?<=[.!?])\s+', text)
    long_sents = [s for s in sentences if len(s) > 40]
    return " ".join(long_sents[:3]) if long_sents else (sentences[0] if sentences else "")

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Phone Review Analyzer", layout="wide")
st.title("📱 Phone Review Analyzer (Supabase + scikit-learn)")

with st.sidebar:
    phone = st.text_input("Phone model", value="Samsung Galaxy S24")
    gsmarena_url = st.text_input("Optional: GSMArena review page URL")
    max_scrape = st.slider("Max reviews to scrape", 10, 500, 150, step=10)
    clear_db = st.button("⚠️ Clear cached reviews")

if clear_db:
    clear_reviews_from_db(phone)
    st.success(f"Cleared reviews for {phone} from Supabase.")

if st.button("🔎 Load / Analyze"):
    reviews = get_reviews_from_db(phone)
    if not reviews and gsmarena_url:
        st.info("No cached reviews found. Scraping...")
        scraped = fetch_gsmarena_reviews(gsmarena_url, max_reviews=max_scrape)
        if scraped:
            save_reviews_to_db(phone, scraped)
            reviews = get_reviews_from_db(phone)
            st.success(f"Scraped & saved {len(scraped)} reviews.")
        else:
            st.warning("Scraper failed or no reviews found.")
    elif not reviews:
        st.warning("No reviews in DB. Provide a GSMArena URL.")

    if reviews:
        st.subheader("Sample reviews")
        for i, r in enumerate(reviews[:5], 1):
            st.write(f"**{i}.** {r[:400]}{'...' if len(r)>400 else ''}")

        st.subheader("🔬 Analysis results")
        aspect_scores = aspect_based_sentiment(reviews)
        df_aspect = pd.DataFrame(list(aspect_scores.items()), columns=["Aspect", "Score"])
        st.bar_chart(df_aspect.set_index("Aspect"))

        summary = summarize_reviews(reviews)
        st.subheader("📝 Summary")
        st.write(summary or "No summary available.")

        avg_score = sum(aspect_scores.values()) / max(len(aspect_scores), 1)
        verdict = "Positive" if avg_score > 0.1 else ("Negative" if avg_score < -0.1 else "Mixed")
        st.info(f"📣 Verdict: **{verdict}** (avg = {avg_score:.2f})")

        out = {
            "phone_model": phone,
            "num_reviews": len(reviews),
            "aspect_scores": aspect_scores,
            "summary": summary,
            "verdict": verdict
        }
        st.download_button("📥 Download JSON", json.dumps(out, indent=2), file_name=f"{phone}_summary.json")