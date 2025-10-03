# app.py
import os
import re
import time
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Text summarization (optional; fallback available)
try:
    from summa import summarizer
    SUMMA_AVAILABLE = True
except Exception:
    SUMMA_AVAILABLE = False

# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = Path("reviews.db")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
VECT_PATH = MODEL_DIR / "vectorizer.joblib"  # optional, but pipeline saved in joblib anyway

ASPECT_KEYWORDS = {
    "Battery": ["battery", "charge", "charging", "power", "life", "mah"],
    "Camera": ["camera", "photo", "picture", "video", "selfie", "lens"],
    "Display": ["display", "screen", "resolution", "brightness", "oled", "lcd"],
    "Performance": ["performance", "speed", "lag", "slow", "fps", "processor", "chip", "cpu"]
}

# -----------------------------
# DB HELPERS (SQLite)
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_model TEXT,
            review_text TEXT,
            review_date TEXT,
            rating INTEGER,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

def save_reviews_to_db(phone_model: str, reviews: List[str]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for r in reviews:
        cur.execute(
            "INSERT INTO reviews (phone_model, review_text) VALUES (?, ?)",
            (phone_model, r[:2000])  # limit length for safety
        )
    conn.commit()
    conn.close()

@st.cache_data(ttl=3600)
def get_reviews_from_db(phone_model: str, limit: int = 500) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT review_text FROM reviews WHERE phone_model = ? ORDER BY id DESC LIMIT ?", (phone_model, limit))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

# -----------------------------
# SIMPLE GSMARENA SCRAPER
# User can paste a GSMArena "reviews" page URL.
# The function is intentionally conservative (polite delay + robust selectors).
# -----------------------------
def fetch_gsmarena_reviews(review_page_url: str, max_reviews: int = 200) -> List[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; bot/0.1; +https://example.com/bot)"
    }
    reviews = []
    try:
        r = requests.get(review_page_url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # GSMArena sometimes uses divs with class "uopin" or ".opin"
        selectors = [".opin", ".user-opinion", ".uopin", ".review-content", ".opinion", ".user-review"]
        blocks = []
        for sel in selectors:
            found = soup.select(sel)
            if found:
                blocks = found
                break

        # fallback to paragraphs under reviews container
        if not blocks:
            container = soup.select_one("#review")
            if container:
                blocks = container.find_all("p")
            else:
                blocks = soup.find_all("p")

        for b in blocks:
            text = b.get_text(" ", strip=True)
            if text and len(text) > 30:  # filter out tiny bits
                reviews.append(text)
                if len(reviews) >= max_reviews:
                    break

    except Exception as e:
        st.warning(f"Scrape failed: {e}")

    time.sleep(1)  # polite pause for future calls
    return reviews

# -----------------------------
# SENTIMENT MODEL (TF-IDF -> Logistic Regression)
# - If a saved model exists, load it.
# - Otherwise, auto-train a tiny starter model (NOT production quality).
#   Replace starter data with your labeled dataset for best results.
# -----------------------------
@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception:
            st.warning("Saved model exists but failed to load. Re-training.")
    # Starter training data (very small). Replace with your labeled dataset.
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
    y_train = ["POSITIVE", "POSITIVE", "POSITIVE", "POSITIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

model_pipeline = load_or_train_model()

# -----------------------------
# PREDICTION & ASPECT ANALYSIS
# -----------------------------
def predict_sentiment(texts: List[str]) -> List[Tuple[str, float]]:
    """
    Returns list of (label, score) for each text.
    Score is the predicted probability for the chosen label.
    """
    if not texts:
        return []
    preds = model_pipeline.predict(texts)
    if hasattr(model_pipeline, "predict_proba"):
        probs = model_pipeline.predict_proba(texts)
        # find probability for predicted class
        label_index = {lab: i for i, lab in enumerate(model_pipeline.classes_)}
        results = []
        for i, lab in enumerate(preds):
            prob = float(probs[i, label_index[lab]])
            results.append((lab, prob))
        return results
    else:
        # fallback: no probabilities
        return [(str(lab), 0.0) for lab in preds]

def aspect_based_sentiment(reviews: List[str]) -> Dict[str, float]:
    """
    For each aspect, collect sentences that mention aspect keywords,
    run sentiment classifier, then compute mean polarity score (-1 to +1).
    """
    aspect_scores = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        hits = []
        for r in reviews:
            low = r.lower()
            if any(k in low for k in keywords):
                hits.append(r)
        if not hits:
            aspect_scores[aspect] = 0.0
            continue
        preds = predict_sentiment(hits)
        # map POSITIVE -> +1, NEGATIVE -> -1
        mapped = []
        for lab, score in preds:
            if lab.upper().startswith("POS"):
                mapped.append(1.0 * (score if score>0 else 1.0))
            else:
                mapped.append(-1.0 * (score if score>0 else 1.0))
        # average, normalize to -1..+1
        aspect_scores[aspect] = float(sum(mapped) / len(mapped)) if mapped else 0.0
    return aspect_scores

# -----------------------------
# SUMMARY & PROS/CONS EXTRACTION
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
        except Exception:
            pass
    # fallback: return first 3 longish sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    long_sents = [s for s in sentences if len(s) > 40]
    return " ".join(long_sents[:3]) if long_sents else (sentences[0] if sentences else "")

def extract_pros_cons(reviews: List[str], top_n: int = 3) -> Tuple[List[str], List[str]]:
    pos = []
    neg = []
    preds = predict_sentiment(reviews)
    for (lab, score), review in zip(preds, reviews):
        if lab.upper().startswith("POS"):
            pos.append(review)
        else:
            neg.append(review)
    return pos[:top_n], neg[:top_n]

# -----------------------------
# STREAMLIT APP UI
# -----------------------------
st.set_page_config(page_title="Phone Review Analyzer (SQLite + scikit-learn)", layout="wide")
st.title("📱 Phone Review Analyzer — full DS (SQLite + scikit-learn)")

# Initialize DB
init_db()

with st.sidebar:
    st.header("Options")
    phone = st.text_input("Phone model (identifier stored in DB)", value="Samsung Galaxy S24")
    gsmarena_url = st.text_input("Optional: GSMArena review page URL (paste full reviews page URL)")
    max_scrape = st.slider("Max reviews to scrape (if scraping)", 10, 500, 150, step=10)
    clear_db = st.button("⚠️ Clear cached reviews for this phone")

if clear_db:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM reviews WHERE phone_model = ?", (phone,))
    conn.commit()
    conn.close()
    st.success("Cleared reviews for phone from DB.")

if st.button("🔎 Load / Analyze"):
    if not phone:
        st.error("Please enter a phone model identifier (any string).")
    else:
        # 1) Try DB
        reviews = get_reviews_from_db(phone)
        if not reviews and gsmarena_url:
            st.info("No cached reviews found. Scraping provided GSMArena URL...")
            scraped = fetch_gsmarena_reviews(gsmarena_url, max_reviews=max_scrape)
            if scraped:
                save_reviews_to_db(phone, scraped)
                reviews = get_reviews_from_db(phone)
                st.success(f"Scraped & saved {len(scraped)} reviews.")
            else:
                st.warning("Scraper didn't find reviews or failed. Check the URL.")
        elif not reviews:
            st.warning("No reviews in DB for this phone. Provide a GSMArena review URL to scrape or add reviews manually.")
        else:
            st.success(f"Loaded {len(reviews)} cached reviews from DB (latest first).")

        if reviews:
            # Basic preview and download
            st.subheader("Sample reviews (most recent)")
            for i, r in enumerate(reviews[:5], 1):
                st.write(f"**{i}.** {r[:400]}{'...' if len(r)>400 else ''}")

            if st.button("Download all reviews as CSV"):
                df = pd.DataFrame({"review_text": reviews})
                st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{phone}_reviews.csv", mime="text/csv")

            # 2) Sentiment analysis + ABSA
            st.subheader("🔬 Analysis results")
            with st.spinner("Running aspect-based sentiment..."):
                aspect_scores = aspect_based_sentiment(reviews)
            df_aspect = pd.DataFrame(list(aspect_scores.items()), columns=["Aspect", "Score"])
            st.table(df_aspect)

            # bar chart
            st.bar_chart(df_aspect.set_index("Aspect"))

            # pros & cons
            pros, cons = extract_pros_cons(reviews, top_n=5)
            st.subheader("✅ Pros (sample)")
            if pros:
                for p in pros:
                    st.success(p)
            else:
                st.write("No positive sentences detected.")

            st.subheader("❌ Cons (sample)")
            if cons:
                for c in cons:
                    st.error(c)
            else:
                st.write("No negative sentences detected.")

            # summary & verdict
            summary = summarize_reviews(reviews)
            st.subheader("📝 Summary (extractive)")
            st.write(summary or "No summary available.")

            # Simple aggregate verdict
            avg_score = sum(aspect_scores.values()) / max(len(aspect_scores), 1)
            verdict = "Positive overall" if avg_score > 0.1 else ("Negative overall" if avg_score < -0.1 else "Mixed / Neutral")
            st.info(f"📣 Verdict: **{verdict}** (avg aspect score = {avg_score:.2f})")

            # JSON output
            out = {
                "phone_model": phone,
                "num_reviews": len(reviews),
                "aspect_scores": aspect_scores,
                "pros_sample": pros,
                "cons_sample": cons,
                "summary": summary,
                "verdict": verdict
            }
            st.download_button("📥 Download JSON summary", json.dumps(out, indent=2, ensure_ascii=False), file_name=f"{phone}_summary.json", mime="application/json")