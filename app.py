import os import re import json import time import hashlib import requests import pandas as pd import streamlit as st from bs4 import BeautifulSoup import google.generativeai as genai

-----------------------------

1️⃣ Configure Gemini API Key

-----------------------------

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") if not api_key: st.error("❌ Missing Gemini API key. Please set GEMINI_API_KEY in secrets or env vars.") st.stop()

genai.configure(api_key=api_key) model = genai.GenerativeModel("gemini-1.5-flash")

-----------------------------

2️⃣ Enhanced GSMArena URL Resolver

-----------------------------

@st.cache_data(ttl=86400, show_spinner="🔗 Finding GSMArena page...") def resolve_gsmarena_url(product_name): try: query = product_name.replace(" ", "+") search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={query}"

headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(search_url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    selectors = [".makers a", ".makers li a", ".section-body .makers a", "a[href*='.php']"]

    link = None
    for selector in selectors:
        link = soup.select_one(selector)
        if link:
            break
    if not link:
        return None, None

    product_url = "https://www.gsmarena.com/" + link["href"]

    href = link["href"]
    if "-" in href and ".php" in href:
        phone_id = href.split("-")[-1].replace(".php", "")
        phone_name = href.replace(f"-{phone_id}.php", "").replace("_", " ")

        review_urls = [
            f"https://www.gsmarena.com/{href.replace('.php', '-reviews.php')}",
            f"https://m.gsmarena.com/{href.replace('_', ' ').replace('.php', '')}-reviews-{phone_id}.php",
            f"https://www.gsmarena.com/reviews.php3?idPhone={phone_id}",
            f"https://m.gsmarena.com/{phone_name.replace(' ', '_')}-reviews-{phone_id}.php"
        ]

        valid_review_url = None
        for review_url in review_urls:
            try:
                test_response = requests.get(review_url, headers=headers, timeout=5, stream=True, allow_redirects=True)
                if test_response.status_code == 200:
                    valid_review_url = review_url
                    test_response.close()
                    break
                test_response.close()
            except:
                continue
        return product_url, valid_review_url
    else:
        return product_url, product_url.replace(".php", "-reviews.php")

except Exception as e:
    st.warning(f"⚠️ GSMArena search failed: {e}")
    return None, None

-----------------------------

3️⃣ Enhanced Specs Scraper

-----------------------------

@st.cache_data(ttl=86400, show_spinner="📊 Fetching specs...") def fetch_gsmarena_specs(url): specs = {} key_map = { "Display": ["Display", "Screen", "Size"], "Processor": ["Chipset", "CPU", "Processor", "SoC"], "RAM": ["Internal", "Memory", "RAM"], "Storage": ["Internal", "Storage", "Memory"], "Camera": ["Main Camera", "Triple", "Quad", "Dual", "Camera"], "Battery": ["Battery"], "OS": ["OS", "Android", "iOS"] } try: headers = {"User-Agent": "Mozilla/5.0"} r = requests.get(url, headers=headers, timeout=10) r.raise_for_status() soup = BeautifulSoup(r.text, "html.parser")

spec_containers = [".article-info table tr", "#specs-list table tr", "table.specs tr", ".specs-brief-accent tr"]
    spec_rows = []
    for container_selector in spec_containers:
        spec_rows = soup.select(container_selector)
        if spec_rows:
            break

    for row in spec_rows:
        th = row.find("td", class_="ttl") or row.find("th") or row.find("td", class_="spec-title")
        td = row.find("td", class_="nfo") or (row.find_all("td")[-1] if row.find_all("td") else None)
        if not th or not td:
            continue
        key = th.get_text(strip=True)
        val = td.get_text(" ", strip=True) if hasattr(td, 'get_text') else str(td)

        for field, keywords in key_map.items():
            if any(k.lower() in key.lower() for k in keywords):
                if field in ["RAM", "Storage"]:
                    gb_matches = re.findall(r'(\d+)\s*GB', val)
                    if gb_matches:
                        if field == "RAM":
                            specs["RAM"] = f"{gb_matches[0]}GB RAM"
                            if len(gb_matches) > 1:
                                specs["Storage"] = f"{gb_matches[1]}GB Storage"
                        elif field == "Storage":
                            specs["Storage"] = f"{gb_matches[-1]}GB Storage"
                    else:
                        specs[field] = val
                else:
                    specs[field] = val
                break
except Exception as e:
    st.warning(f"⚠️ GSMArena specs fetch failed: {e}")
return specs

-----------------------------

4️⃣ Enhanced Reviews Scraper with Pagination

-----------------------------

@st.cache_data(ttl=21600, show_spinner="💬 Fetching GSMArena reviews...") def fetch_gsmarena_reviews(url, limit=20): reviews = [] if not url: return reviews

headers = {"User-Agent": "Mozilla/5.0"}

try:
    page = 1
    while len(reviews) < limit and page <= 3:  # fetch up to 3 pages
        time.sleep(1)
        paged_url = url if page == 1 else f"{url}?page={page}"
        r = requests.get(paged_url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        review_blocks = soup.select(".opin, .user-opinion, .uopin, .user-review, .review-item, .user-thread p")
        for block in review_blocks:
            review_text = block.get_text(strip=True)
            if 35 < len(review_text) < 1000:
                if not any(word in review_text.lower() for word in ["gsmarena", "admin", "moderator", "delete", "report"]):
                    reviews.append(review_text)
                    if len(reviews) >= limit:
                        break
        page += 1
except Exception as e:
    st.warning(f"⚠️ GSMArena reviews fetch failed: {e}")

return reviews[:limit]

-----------------------------

5️⃣ Enhanced Summarizer with JSON safety

-----------------------------

@st.cache_data(ttl=43200, show_spinner="🤖 Summarizing with Gemini...") def summarize_reviews(product_name, specs, reviews): try: specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found" reviews_context = "\n".join([f"- {r[:200]}..." if len(r) > 200 else f"- {r}" for r in reviews[:20]]) if reviews else "No reviews found"

prompt = f"""
    You are an AI Review Summarizer analyzing the {product_name}.
    Combine **GSMArena official specs** with **real user reviews** to create a comprehensive analysis.
    Return ONLY valid JSON.
    ... (same JSON schema as before) ...
    """

    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    clean_output = response.text.strip()
    return clean_output
except Exception as e:
    st.error(f"⚠️ Gemini API error: {e}")
    return None

-----------------------------

6️⃣ Streamlit UI with Custom Cards & Charts

-----------------------------

st.set_page_config(page_title="AI Review Engine", page_icon="📱", layout="wide")

Custom CSS for metric cards

st.markdown("""

<style>
    .metric-card { background-color: black; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; color: white; }
    .metric-title-box { color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; display: inline-block; margin-bottom: 0.5rem; }
    .metric-verdict .metric-title-box { background-color: #ff6b35; }
    .metric-best-for .metric-title-box { background-color: #1f77b4; }
    .metric-data-found .metric-title-box { background-color: #28a745; }
    .metric-card p { word-wrap: break-word; overflow-wrap: break-word; margin: 0; color: white; }
</style>""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Settings") review_limit = st.sidebar.slider("Max reviews to analyze", 10, 50, 25, step=5)

st.title("📱 AI-Powered Phone Review Engine") phone = st.text_input("Enter phone name", value="Samsung Galaxy S24") generate_button = st.button("🔍 Analyze Phone")

if generate_button and phone: product_url, review_url = resolve_gsmarena_url(phone) if not product_url: st.error(f"❌ Could not find '{phone}' on GSMArena.") st.stop()

specs = fetch_gsmarena_specs(product_url)
reviews = fetch_gsmarena_reviews(review_url, limit=review_limit) if review_url else []
summary_result = summarize_reviews(phone, specs, reviews)

if not summary_result:
    st.error("⚠️ Failed to generate AI summary.")
    st.stop()

try:
    summary_data = json.loads(summary_result)
except Exception as e:
    st.error(f"⚠️ Error parsing AI response: {e}")
    st.text_area("Raw Output", summary_result, height=200)
    st.stop()

# Metric cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card metric-verdict">
        <span class="metric-title-box">⭐ Verdict</span>
        <p style="font-size: 1.1rem; font-weight: 600;">{summary_data.get('verdict', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card metric-best-for">
        <span class="metric-title-box">🎯 Best For</span>
        <p style="font-size: 1.1rem; font-weight: 600;">{summary_data.get('recommendation', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card metric-data-found">
        <span class="metric-title-box">📊 Data Found</span>
        <p style="font-size: 1.1rem; font-weight: 600;">{len(specs)} specs, {len(reviews)} reviews</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 🎯 Bottom Line")
st.info(summary_data.get("bottom_line", "No summary available."))

col1, col2 = st.columns([0.6, 0.4])
with col1:
    if "phone_specs" in summary_data and summary_data["phone_specs"]:
        st.markdown("### 🔧 Technical Specifications")
        specs_df = pd.DataFrame(summary_data["phone_specs"].items(), columns=["Component", "Details"])
        st.table(specs_df)
    if "aspect_sentiments" in summary_data and summary_data["aspect_sentiments"]:
        st.markdown("### 📊 User Sentiment Analysis")
        df = pd.DataFrame(summary_data["aspect_sentiments"])
        if not df.empty and "Aspect" in df.columns:
            chart_data = df.set_index("Aspect")["Positive"].to_frame().join(df.set_index("Aspect")["Negative"])
            st.bar_chart(chart_data, height=300)
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

st.markdown("---")
st.markdown("**Data Sources:** GSMArena specifications and user reviews | **AI Analysis:** Google Gemini")
st.download_button("📥 Download JSON", json.dumps(summary_data, indent=2), "summary.json")

