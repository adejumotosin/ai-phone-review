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
# 2️⃣ Enhanced GSMArena URL Resolver
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="🔗 Finding GSMArena page...")
def resolve_gsmarena_url(product_name):
    """Search GSMArena for a product and return its base + reviews URL"""
    try:
        query = product_name.replace(" ", "+")
        search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={query}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        r = requests.get(search_url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Try multiple selectors for robustness
        selectors = [
            ".makers a",
            ".makers li a", 
            ".section-body .makers a",
            "a[href*='.php']"
        ]
        
        link = None
        for selector in selectors:
            link = soup.select_one(selector)
            if link:
                break
                
        if not link:
            return None, None

        product_url = "https://www.gsmarena.com/" + link["href"]
        
        # Extract the phone ID from the URL for building review URL
        # URL format: samsung_galaxy_s24-12773.php
        href = link["href"]
        if "-" in href and ".php" in href:
            # Extract the ID (e.g., "12773" from "samsung_galaxy_s24-12773.php")
            phone_id = href.split("-")[-1].replace(".php", "")
            phone_name = href.replace(f"-{phone_id}.php", "").replace("_", " ")
            
            # Try multiple review URL patterns
            review_urls = [
                f"https://www.gsmarena.com/{href.replace('.php', '-reviews.php')}",
                f"https://m.gsmarena.com/{href.replace('_', ' ').replace('.php', '')}-reviews-{phone_id}.php",
                f"https://www.gsmarena.com/reviews.php3?idPhone={phone_id}",
                f"https://m.gsmarena.com/{phone_name.replace(' ', '_')}-reviews-{phone_id}.php"
            ]
            
            # Test which review URL actually exists
            valid_review_url = None
            for review_url in review_urls:
                try:
                    test_response = requests.head(review_url, headers=headers, timeout=5)
                    if test_response.status_code == 200:
                        valid_review_url = review_url
                        break
                except:
                    continue
            
            return product_url, valid_review_url
        else:
            # Fallback to original method
            review_url = product_url.replace(".php", "-reviews.php")
            return product_url, review_url
        
    except Exception as e:
        st.warning(f"⚠️ GSMArena search failed: {e}")
        return None, None

# -----------------------------
# 3️⃣ Enhanced Specs Scraper
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Try multiple selectors for spec tables
        spec_containers = [
            ".article-info table tr",
            "#specs-list table tr",
            "table.specs tr",
            ".specs-brief-accent tr"
        ]
        
        spec_rows = []
        for container_selector in spec_containers:
            spec_rows = soup.select(container_selector)
            if spec_rows:
                break
        
        for row in spec_rows:
            # Try different cell selectors
            th = row.find("td", class_="ttl") or row.find("th") or row.find("td", class_="spec-title")
            td = row.find("td", class_="nfo") or row.find_all("td")[-1] if row.find_all("td") else None
            
            if not th or not td:
                continue
                
            key = th.get_text(strip=True)
            val = td.get_text(" ", strip=True) if hasattr(td, 'get_text') else str(td)

            # Enhanced spec mapping
            for field, keywords in key_map.items():
                if any(k.lower() in key.lower() for k in keywords):
                    if field in ["RAM", "Storage"]:
                        # Better RAM/Storage parsing
                        if "GB" in val:
                            # Extract numbers followed by GB
                            gb_matches = re.findall(r'(\d+)\s*GB', val)
                            if gb_matches:
                                if field == "RAM":
                                    specs["RAM"] = f"{gb_matches[0]}GB RAM"
                                elif field == "Storage" and len(gb_matches) > 1:
                                    specs["Storage"] = f"{gb_matches[1]}GB Storage"
                                elif field == "Storage":
                                    specs["Storage"] = f"{gb_matches[0]}GB Storage"
                        else:
                            specs[field] = val
                    else:
                        specs[field] = val
                    break
                    
    except Exception as e:
        st.warning(f"⚠️ GSMArena specs fetch failed: {e}")
    
    return specs

# -----------------------------
# 4️⃣ Enhanced Reviews Scraper (Mobile & Desktop)
# -----------------------------
@st.cache_data(ttl=21600, show_spinner="💬 Fetching GSMArena reviews...")
def fetch_gsmarena_reviews(url, limit=20):
    reviews = []
    if not url:
        return reviews
        
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Detect if this is mobile or desktop GSMArena
        is_mobile = "m.gsmarena.com" in url
        
        if is_mobile:
            # Mobile GSMArena selectors
            review_containers = [
                ".user-thread",
                ".thread",
                ".review-item",
                "#user-reviews"
            ]
            
            review_selectors = [
                ".uopin",
                ".user-opinion", 
                ".thread-content",
                "p"
            ]
        else:
            # Desktop GSMArena selectors
            review_containers = [
                "#user-comments",
                ".user-reviews",
                ".reviews-container",
                "#comments"
            ]
            
            review_selectors = [
                ".opin",
                ".user-review",
                ".review-item",
                "p"
            ]
        
        # Find review container
        review_container = None
        for container_selector in review_containers:
            if container_selector.startswith("#"):
                review_container = soup.find(id=container_selector[1:])
            else:
                review_container = soup.select_one(container_selector)
            if review_container:
                break
        
        if not review_container:
            # Try to find any container with reviews
            review_container = soup.find("div", string=lambda text: text and ("review" in text.lower() or "opinion" in text.lower()))
            if not review_container:
                review_container = soup  # Use entire page as fallback
        
        # Find individual reviews
        review_blocks = []
        for selector in review_selectors:
            if selector.startswith("."):
                review_blocks = review_container.find_all("div", class_=selector[1:])
                if not review_blocks:
                    review_blocks = review_container.find_all("li", class_=selector[1:])
            else:
                review_blocks = review_container.find_all(selector)
            
            if review_blocks:
                break

        # Extract review text
        for block in review_blocks[:limit * 2]:  # Get more to filter better
            review_text = block.get_text(strip=True)
            
            # Filter out noise and keep quality reviews
            if (len(review_text) > 30 and 
                len(review_text) < 1000 and
                not review_text.lower().startswith(("anonymous", "user", "by", "reply", "quote")) and
                not any(word in review_text.lower() for word in ["gsmarena", "admin", "moderator", "delete", "report"]) and
                any(word in review_text.lower() for word in ["phone", "camera", "battery", "screen", "good", "bad", "love", "hate", "recommend", "buy", "device"])):
                
                reviews.append(review_text)
                
                if len(reviews) >= limit:
                    break
        
        # If no reviews found with strict filtering, try with looser criteria
        if not reviews:
            for block in review_blocks[:limit]:
                review_text = block.get_text(strip=True)
                if len(review_text) > 20 and len(review_text) < 500:
                    reviews.append(review_text)
            
    except Exception as e:
        st.warning(f"⚠️ GSMArena reviews fetch failed: {e}")
    
    return reviews[:limit]

# -----------------------------
# 5️⃣ Enhanced Summarizer with Better Error Handling
# -----------------------------
@st.cache_data(ttl=43200, show_spinner="🤖 Summarizing with Gemini...")
def summarize_reviews(product_name, specs, reviews):
    try:
        reviews_text = "\n".join(reviews[:50])
        reviews_hash = hashlib.md5((reviews_text + str(specs)).encode("utf-8")).hexdigest()

        specs_context = "\n".join([f"{k}: {v}" for k, v in specs.items()]) if specs else "No specs found"
        reviews_context = "\n".join([f"- {r[:200]}..." if len(r) > 200 else f"- {r}" for r in reviews[:20]]) if reviews else "No reviews found"

        prompt = f"""
        You are an AI Review Summarizer analyzing the {product_name}.
        Combine **GSMArena official specs** with **real user reviews** to create a comprehensive analysis.

        OFFICIAL SPECS:
        {specs_context}

        USER REVIEWS SAMPLE:
        {reviews_context}

        CRITICAL REQUIREMENTS:
        1. Always include all 7 spec fields: Display, Processor, RAM, Storage, Camera, Battery, OS
        2. If a spec is missing, research common specs for this phone model or mark as "Not specified"
        3. Keep verdict under 30 characters (e.g., "Great flagship" not "Excellent high-end smartphone with premium features")
        4. Keep recommendation under 35 characters (e.g., "Power users & photographers" not "Best for users who prioritize performance and camera quality")
        5. Extract actual user quotes (2-4 sentences max each)
        6. Return ONLY valid JSON (no markdown, no extra text)

        Output this exact JSON structure:
        {{
          "verdict": "Brief rating (max 30 chars, e.g., 'Great flagship phone')",
          "pros": ["Specific advantage 1", "Specific advantage 2", "Specific advantage 3"],
          "cons": ["Specific drawback 1", "Specific drawback 2", "Specific drawback 3"],
          "aspect_sentiments": [
            {{"Aspect": "Camera", "Positive": 75, "Negative": 25}},
            {{"Aspect": "Battery", "Positive": 80, "Negative": 20}},
            {{"Aspect": "Performance", "Positive": 70, "Negative": 30}},
            {{"Aspect": "Display", "Positive": 85, "Negative": 15}},
            {{"Aspect": "Build Quality", "Positive": 65, "Negative": 35}}
          ],
          "user_quotes": ["Real quote from reviews", "Another user opinion", "Third user comment"],
          "recommendation": "Target audience (max 35 chars, e.g., 'Power users & photographers')",
          "bottom_line": "2-3 sentence final summary combining specs and user feedback",
          "phone_specs": {{
            "Display": "Screen size and type",
            "Processor": "Chipset name and performance tier",
            "RAM": "Memory amount",
            "Storage": "Storage capacity and type",
            "Camera": "Main camera specs and features",
            "Battery": "Battery capacity and charging",
            "OS": "Operating system version"
          }}
        }}
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )

        return response.text, reviews_hash
    
    except Exception as e:
        st.error(f"⚠️ Gemini API error: {e}")
        return None, None

# -----------------------------
# 6️⃣ Enhanced Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="AI Review Engine", 
    page_icon="📱", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress .st-bp {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.title("📱 AI-Powered Phone Review Engine")
st.markdown("**Get comprehensive analysis combining GSMArena specs with real user reviews**")

# Input section
col1, col2 = st.columns([0.7, 0.3])
with col1:
    phone = st.text_input(
        "Enter phone name", 
        value="Samsung Galaxy S24", 
        placeholder="e.g., iPhone 15 Pro, Samsung Galaxy S24, OnePlus 12"
    )
with col2:
    generate_button = st.button("🔍 Analyze Phone", type="primary", use_container_width=True)

# Main processing
if generate_button and phone:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: URL Resolution
    status_text.text("🔎 Searching GSMArena database...")
    progress_bar.progress(20)
    
    product_url, review_url = resolve_gsmarena_url(phone)
    
    if not product_url:
        st.error(f"❌ Could not find '{phone}' on GSMArena. Try a different phone name.")
        st.info("💡 **Tip**: Try using the exact model name (e.g., 'Galaxy S24' instead of 'Samsung S24')")
        st.stop()
    
    st.success(f"✅ Found product page: {product_url}")
    if review_url:
        st.success(f"✅ Found reviews page: {review_url}")
    else:
        st.warning("⚠️ No review page found - will proceed with specs only")
    
    # Step 2: Fetch specs
    status_text.text("📊 Extracting technical specifications...")
    progress_bar.progress(40)
    
    specs = fetch_gsmarena_specs(product_url)
    
    # Step 3: Fetch reviews
    status_text.text("💬 Collecting user reviews...")
    progress_bar.progress(60)
    
    reviews = []
    if review_url:
        reviews = fetch_gsmarena_reviews(review_url, limit=25)
        if reviews:
            st.success(f"✅ Found {len(reviews)} user reviews")
        else:
            st.warning("⚠️ No user reviews found - analysis will be based on specs only")
    else:
        st.info("ℹ️ No review URL available - proceeding with specs analysis")
    
    # Step 4: AI Analysis
    status_text.text("🤖 Generating AI analysis...")
    progress_bar.progress(80)
    
    if not reviews and not specs:
        st.error("❌ No data found. The phone might be too new or not available on GSMArena.")
        st.stop()
    
    # Summarize with Gemini
    summary_result = summarize_reviews(phone, specs, reviews)
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    if not summary_result[0]:
        st.error("⚠️ Failed to generate AI summary. Please try again.")
        st.stop()
    
    try:
        summary_data = json.loads(summary_result[0])
    except json.JSONDecodeError as e:
        st.error(f"⚠️ Error parsing AI response: {e}")
        with st.expander("Raw AI Output"):
            st.text_area("Debug Output", summary_result[0], height=200)
        st.stop()

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    st.markdown("---")
    st.subheader(f"📝 Complete Analysis: {phone}")
    
    # Metrics row with shorter labels and better formatting
    col1, col2, col3 = st.columns(3)
    with col1:
        verdict = summary_data.get("verdict", "N/A")
        # Truncate long verdicts
        if len(verdict) > 35:
            verdict = verdict[:32] + "..."
        st.metric("⭐ Verdict", verdict)
    with col2:
        recommendation = summary_data.get("recommendation", "N/A")
        # Truncate long recommendations  
        if len(recommendation) > 35:
            recommendation = recommendation[:32] + "..."
        st.metric("🎯 Best For", recommendation)
    with col3:
        st.metric("📊 Data Found", f"{len(specs)} specs, {len(reviews)} reviews")
    
    # Show full verdict and recommendation in expandable sections if truncated
    if len(summary_data.get("verdict", "")) > 35 or len(summary_data.get("recommendation", "")) > 35:
        with st.expander("📋 Full Details"):
            if len(summary_data.get("verdict", "")) > 35:
                st.write(f"**Complete Verdict:** {summary_data.get('verdict', 'N/A')}")
            if len(summary_data.get("recommendation", "")) > 35:
                st.write(f"**Complete Recommendation:** {summary_data.get('recommendation', 'N/A')}")

    # Bottom line summary
    st.markdown("### 🎯 Bottom Line")
    st.info(summary_data.get("bottom_line", "No summary available."))

    # Two-column layout for main content
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        # Specifications table
        if "phone_specs" in summary_data and summary_data["phone_specs"]:
            st.markdown("### 🔧 Technical Specifications")
            specs_df = pd.DataFrame(
                summary_data["phone_specs"].items(), 
                columns=["Component", "Details"]
            )
            st.table(specs_df)
        
        # Aspect sentiments chart
        if "aspect_sentiments" in summary_data and summary_data["aspect_sentiments"]:
            st.markdown("### 📊 User Sentiment Analysis")
            df = pd.DataFrame(summary_data["aspect_sentiments"])
            if not df.empty and "Aspect" in df.columns:
                chart_data = df.set_index("Aspect")[["Positive", "Negative"]]
                st.bar_chart(chart_data, height=300)
    
    with col2:
        # Pros and Cons
        st.markdown("### ✅ Strengths")
        for pro in summary_data.get("pros", []):
            st.success(f"✓ {pro}")
            
        st.markdown("### ⚠️ Weaknesses")  
        for con in summary_data.get("cons", []):
            st.error(f"✗ {con}")

    # User quotes section
    if summary_data.get("user_quotes"):
        st.markdown("### 💬 What Users Are Saying")
        for i, quote in enumerate(summary_data.get("user_quotes", []), 1):
            st.info(f"**User {i}:** {quote}")

    # Footer info
    st.markdown("---")
    st.markdown("**Data Sources:** GSMArena specifications and user reviews | **AI Analysis:** Google Gemini")
    
    # Option to view raw data
    with st.expander("🔍 View Raw Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.json(specs)
        with col2:
            st.write("**Sample Reviews:**")
            for i, review in enumerate(reviews[:5], 1):
                st.write(f"{i}. {review[:200]}...")