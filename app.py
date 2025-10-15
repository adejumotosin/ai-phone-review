import streamlit as st
import json
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import hashlib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import urllib.parse

# --- 1. Enhanced Pydantic Schema with Aspect-Based Sentiment ---
class AspectSentiment(BaseModel):
    """Sentiment analysis for a specific product aspect"""
    aspect: str = Field(description="Product aspect (e.g., 'Battery Life', 'Camera Quality')")
    sentiment_score: float = Field(description="Sentiment from -1.0 (very negative) to 1.0 (very positive)")
    confidence: float = Field(description="Confidence level 0.0 to 1.0")
    key_mentions: List[str] = Field(description="Key phrases from reviews about this aspect", max_items=3)


class ProductReview(BaseModel):
    """A comprehensive product review with aspect-based sentiment analysis"""
    product_name: str = Field(description="Full product name")
    specifications_inferred: str = Field(description="Key specs summary (max 100 words)")
    predicted_rating: str = Field(description="Rating (e.g., '4.5/5.0')")
    
    # Aspect-based sentiment analysis
    aspect_sentiments: List[AspectSentiment] = Field(
        description="Sentiment analysis for 5-8 key product aspects",
        min_items=5,
        max_items=8
    )
    
    pros: List[str] = Field(description="Strengths (3-5 items)", max_items=5)
    cons: List[str] = Field(description="Weaknesses (3-5 items)", max_items=5)
    verdict: str = Field(description="2-3 sentence conclusion")
    
    # Additional metadata
    price_info: Optional[str] = Field(default="Price varies", description="Pricing")
    category: Optional[str] = Field(default="General", description="Product category")
    sources: List[str] = Field(default=[], max_items=10)
    last_updated: str = Field(default="")
    data_source_type: str = Field(default="web_search")


# --- 2. Visualization Functions ---
def create_aspect_radar_chart(aspect_sentiments):
    """Create radar chart for aspect sentiments"""
    aspects = [a['aspect'] for a in aspect_sentiments]
    # Convert -1 to 1 scale to 0 to 10 scale for better visualization
    scores = [(a['sentiment_score'] + 1) * 5 for a in aspect_sentiments]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=aspects,
        fill='toself',
        name='Sentiment Score',
        line=dict(color='#2E86DE', width=2),
        fillcolor='rgba(46, 134, 222, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                ticktext=['Very Poor', 'Poor', 'Average', 'Good', 'Excellent'],
                tickvals=[2, 4, 5, 7, 9]
            )
        ),
        showlegend=False,
        title={
            'text': "Aspect-Based Sentiment Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_sentiment_bar_chart(aspect_sentiments):
    """Create horizontal bar chart for sentiments"""
    df = pd.DataFrame(aspect_sentiments)
    df = df.sort_values('sentiment_score', ascending=True)
    
    # Color coding
    colors = ['#E74C3C' if s < -0.3 else '#F39C12' if s < 0.3 else '#27AE60' 
              for s in df['sentiment_score']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['aspect'],
        x=df['sentiment_score'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f"{s:+.2f}" for s in df['sentiment_score']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<br>Confidence: %{customdata:.0%}<extra></extra>',
        customdata=df['confidence']
    ))
    
    fig.update_layout(
        title={
            'text': "Sentiment Scores by Aspect",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        xaxis_title="Sentiment Score",
        xaxis=dict(range=[-1.2, 1.2], zeroline=True, zerolinewidth=2, zerolinecolor='#95A5A6'),
        yaxis_title="",
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=80, t=60, b=40)
    )
    
    return fig


def create_sentiment_distribution_pie(aspect_sentiments):
    """Create pie chart showing positive/neutral/negative distribution"""
    positive = sum(1 for a in aspect_sentiments if a['sentiment_score'] > 0.3)
    neutral = sum(1 for a in aspect_sentiments if -0.3 <= a['sentiment_score'] <= 0.3)
    negative = sum(1 for a in aspect_sentiments if a['sentiment_score'] < -0.3)
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive Aspects', 'Neutral Aspects', 'Negative Aspects'],
        values=[positive, neutral, negative],
        hole=0.4,
        marker=dict(colors=['#27AE60', '#F39C12', '#E74C3C']),
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title={
            'text': "Overall Sentiment Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        showlegend=False,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_aspect_details_table(aspect_sentiments):
    """Create detailed table with key mentions"""
    df = pd.DataFrame([
        {
            'Aspect': a['aspect'],
            'Score': f"{a['sentiment_score']:+.2f}",
            'Confidence': f"{a['confidence']:.0%}",
            'Key Mentions': ', '.join(a['key_mentions'][:2])
        }
        for a in sorted(aspect_sentiments, key=lambda x: x['sentiment_score'], reverse=True)
    ])
    
    return df


def create_comparison_chart(reviews):
    """Compare multiple products' aspect sentiments"""
    if len(reviews) < 2:
        return None
    
    # Get all unique aspects
    all_aspects = set()
    for review in reviews:
        for aspect in review.get('aspect_sentiments', []):
            all_aspects.add(aspect['aspect'])
    
    fig = go.Figure()
    
    for review in reviews:
        aspects = []
        scores = []
        
        for aspect_name in sorted(all_aspects):
            aspect_data = next(
                (a for a in review.get('aspect_sentiments', []) if a['aspect'] == aspect_name),
                None
            )
            if aspect_data:
                aspects.append(aspect_name)
                scores.append((aspect_data['sentiment_score'] + 1) * 5)
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=aspects,
            fill='toself',
            name=review.get('product_name', 'Unknown'),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="Product Comparison",
        height=600
    )
    
    return fig


# --- 3. Free Web Search Class ---
class CompleteFreeWebSearch:
    """
    Completely free web-connected LLM solution
    Uses: DuckDuckGo (free) + Groq (free)
    """
    
    def __init__(self, groq_api_key):
        self.groq = Groq(api_key=groq_api_key)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        # Setup cache directory
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def __del__(self):
        """Cleanup session"""
        if hasattr(self, 'session'):
            self.session.close()
    
    def _get_cache_key(self, query):
        """Generate cache key from query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_cached_search(self, query, expiry_hours=24):
        """Get cached search results"""
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if expired
                cached_time = datetime.fromisoformat(cached_data['timestamp'])
                age = datetime.now() - cached_time
                
                if age.total_seconds() / 3600 < expiry_hours:
                    return cached_data['results']
            except:
                pass
        
        return None
    
    def _cache_search(self, query, results):
        """Cache search results"""
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'results': results
                }, f)
        except:
            pass
    
    def _is_safe_url(self, url):
        """Validate URL safety"""
        try:
            parsed = urllib.parse.urlparse(url)
            # Block private IPs and localhost
            if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                return False
            # Block private IP ranges
            if parsed.hostname and parsed.hostname.startswith(('10.', '172.', '192.168.')):
                return False
            return parsed.scheme in ['http', 'https']
        except:
            return False
    
    def search_duckduckgo(self, query, num_results=5):
        """Free DuckDuckGo search with fixed URL parsing"""
        
        # Check cache first
        cached = self._get_cached_search(query)
        if cached:
            st.info("📦 Using cached search results")
            return cached
        
        try:
            url = "https://html.duckduckgo.com/html/"
            data = {'q': query}
            
            response = self.session.post(url, data=data, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='result')[:num_results]:
                link = result.find('a', class_='result__a')
                snippet = result.find('a', class_='result__snippet')
                
                if link and snippet:
                    # Fixed URL parsing for DuckDuckGo redirects
                    url = link.get('href', '')
                    
                    # Parse DDG redirect URLs
                    if '/l/?' in url:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                        url = parsed.get('uddg', [''])[0]
                    
                    if url.startswith('//'):
                        url = 'https:' + url
                    
                    # Validate URL safety
                    if not self._is_safe_url(url):
                        continue
                    
                    results.append({
                        'title': link.text.strip(),
                        'url': url,
                        'snippet': snippet.text.strip()
                    })
            
            # Cache results
            if results:
                self._cache_search(query, results)
            
            return results
            
        except requests.Timeout:
            st.error("⏱️ Search timeout. Please try again.")
            return []
        except requests.RequestException as e:
            st.error(f"❌ Network error: {type(e).__name__}")
            return []
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def get_page_content(self, url, max_chars=2000):
        """Scrape page content with better error handling"""
        if not self._is_safe_url(url):
            st.warning(f"⚠️ Skipped unsafe URL: {url}")
            return None
        
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'ads', 'iframe']):
                element.decompose()
            
            # Try to find main content
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_='content') or
                soup.find('div', id='content') or
                soup.body
            )
            
            if main_content:
                import re
                text = main_content.get_text(separator=' ', strip=True)
                # Clean up whitespace efficiently
                text = re.sub(r'\s+', ' ', text)
                return text[:max_chars]
            
            return None
            
        except requests.Timeout:
            st.warning(f"⏱️ Timeout fetching {url}")
            return None
        except requests.RequestException as e:
            st.warning(f"❌ Network error for {url}: {type(e).__name__}")
            return None
        except Exception as e:
            st.warning(f"Failed to fetch {url}: {str(e)}")
            return None
    
    def estimate_tokens(self, text):
        """Rough token estimate: ~1 token per 4 characters"""
        return len(text) // 4
    
    def generate_review(self, product_name):
        """Complete pipeline: search + scrape + analyze with aspect-based sentiment"""
        
        # Step 1: Search
        st.info("🔍 Searching DuckDuckGo for current product information...")
        search_query = f"{product_name} specifications review price features"
        search_results = self.search_duckduckgo(search_query, num_results=5)
        
        if not search_results:
            st.error("❌ Search failed. Please try again or use AI knowledge mode.")
            return None
        
        st.success(f"✅ Found {len(search_results)} sources")
        
        # Step 2: Get detailed content from top 3 results
        st.info(f"📄 Reading top {min(3, len(search_results))} sources...")
        detailed_content = []
        sources_used = []
        
        for i, result in enumerate(search_results[:3], 1):
            with st.spinner(f"Reading source {i}/3: {result['title'][:50]}..."):
                content = self.get_page_content(result['url'], max_chars=2000)
                if content:
                    detailed_content.append({
                        'url': result['url'],
                        'title': result['title'],
                        'content': content
                    })
                    sources_used.append(result['url'])
                    st.success(f"✅ Read: {result['title'][:60]}...")
                time.sleep(0.5)  # Be polite to servers
        
        if not detailed_content:
            st.warning("⚠️ Could not read detailed content. Using search snippets only.")
        
        # Step 3: Build comprehensive context with token management
        context = f"# Product Review Request: {product_name}\n\n"
        context += f"## Web Search Results ({len(search_results)} sources found):\n\n"
        
        for i, result in enumerate(search_results, 1):
            context += f"**{i}. {result['title']}**\n"
            context += f"   Summary: {result['snippet'][:200]}\n"
            context += f"   URL: {result['url']}\n\n"
        
        # Smart content truncation
        if detailed_content:
            context += f"\n## Detailed Content from Top Sources:\n\n"
            total_chars = 0
            max_total_chars = 8000  # Conservative limit
            
            for i, item in enumerate(detailed_content, 1):
                remaining = max_total_chars - total_chars
                if remaining < 500:
                    st.info(f"ℹ️ Truncated to fit token limit ({i-1} sources)")
                    break
                
                chunk = item['content'][:min(remaining, 2500)]
                context += f"### Source {i}: {item['title']}\n"
                context += f"URL: {item['url']}\n\n"
                context += f"{chunk}\n\n"
                context += "---\n\n"
                total_chars += len(chunk)
        
        # Estimate tokens
        estimated_input_tokens = self.estimate_tokens(context)
        st.info(f"📊 Input: ~{estimated_input_tokens} tokens | Output: 5000 tokens reserved")
        
        # Step 4: Generate review with aspect-based sentiment analysis
        st.info("🤖 Analyzing information and generating review with sentiment analysis...")
        
        system_prompt = """You are an expert product reviewer with specialization in aspect-based sentiment analysis.

**Your Task:**
1. Analyze the product across 5-8 key aspects (e.g., Design, Performance, Battery, Camera, Value, Build Quality, Software, Audio)
2. For each aspect, determine:
   - Sentiment score: -1.0 (very negative) to +1.0 (very positive)
   - Confidence: 0.0 to 1.0 (how sure you are based on available data)
   - Key mentions: 2-3 specific phrases from sources

**Aspect Selection Guidelines:**
- For smartphones: Design, Display, Performance, Camera, Battery, Software, Value
- For laptops: Design, Display, Performance, Keyboard, Battery, Ports, Value
- For headphones: Sound Quality, Comfort, Build Quality, ANC, Battery, Value
- For general products: Choose 5-8 most relevant aspects

**Scoring Rules:**
- +0.8 to +1.0: Exceptional, best-in-class
- +0.4 to +0.7: Good, above average
- -0.3 to +0.3: Average, mixed reviews
- -0.7 to -0.4: Below average, issues noted
- -1.0 to -0.8: Poor, major problems

**Confidence Rules:**
- 0.9-1.0: Multiple sources confirm, clear consensus
- 0.7-0.8: Most sources agree
- 0.5-0.6: Some mentions, moderate agreement
- <0.5: Limited data, uncertain

Generate valid JSON matching the schema. Keep response under 2000 tokens. Use ONLY information from provided sources."""

        user_prompt = f"""Based on the following CURRENT web information gathered on {datetime.now().strftime('%B %d, %Y')}, create a comprehensive product review with aspect-based sentiment analysis.

{context}

Generate JSON with:
1. Product info (name, specs, rating, category)
2. **aspect_sentiments**: Array of 5-8 aspects with:
   - aspect name
   - sentiment_score (-1.0 to 1.0)
   - confidence (0.0 to 1.0)
   - key_mentions (2-3 quotes from sources)
3. Traditional pros/cons (3-5 each)
4. Verdict (2-3 sentences)
5. Price info and sources

Focus on extracting specific opinions about each aspect from the sources. Be critical and honest."""

        try:
            response = self.groq.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=5000,
            )
            
            review = json.loads(response.choices[0].message.content)
            review['sources'] = sources_used
            review['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            review['data_source_type'] = 'free_web_search'
            
            st.success("✅ Review with sentiment analysis generated successfully!")
            
            return review
            
        except Exception as e:
            error_str = str(e)
            
            # Handle specific token errors with fallback
            if 'max completion tokens' in error_str or 'json_validate_failed' in error_str:
                st.error("❌ Response too large. Retrying with minimal context...")
                
                # Emergency fallback: use only search snippets
                minimal_context = f"Product: {product_name}\n\n"
                for r in search_results[:3]:
                    minimal_context += f"- {r['title']}: {r['snippet'][:150]}\n"
                
                try:
                    response = self.groq.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Create brief review with aspect analysis:\n{minimal_context}"}
                        ],
                        model="llama-3.3-70b-versatile",
                        response_format={"type": "json_object"},
                        temperature=0.3,
                        max_tokens=4000
                    )
                    
                    review = json.loads(response.choices[0].message.content)
                    review['sources'] = sources_used[:3]
                    st.warning("⚠️ Generated simplified review due to size limits")
                    return review
                    
                except:
                    st.error("❌ Still failed. Falling back to AI knowledge mode...")
                    return None
            else:
                st.error(f"❌ Generation failed: {error_str}")
                return None


# --- 4. Fallback AI Knowledge Review ---
INITIAL_REVIEW_PROMPT = f"""
You are an **Expert Product Reviewer and Critical Market Analyst** with expertise in aspect-based sentiment analysis.

**Process:**
1. Use your knowledge base to recall common specifications and opinions for the product
2. Analyze 5-8 key aspects with sentiment scores
3. Provide a critical, balanced review

**Important:**
- State clearly that this is based on AI training data (updated through January 2025)
- Recommend users verify current specifications and pricing
- Be honest about what you don't know

**Format your entire output as a single JSON object** with aspect-based sentiment analysis.
"""

def generate_ai_knowledge_review(product_name, client):
    """Fallback to AI knowledge when web search fails"""
    
    user_prompt = f"""Generate a detailed review with aspect-based sentiment analysis for: {product_name}

Include:
1. Product info and specs
2. 5-8 aspect sentiments with scores, confidence, and key points
3. Pros and cons
4. Verdict

Important: Make it clear this is based on training data. Include disclaimers."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": INITIAL_REVIEW_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=5000
        )
        
        review_data = json.loads(chat_completion.choices[0].message.content)
        review_data['data_source_type'] = 'ai_knowledge'
        review_data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        review_data['sources'] = ['AI Training Data (Updated January 2025)']
        
        return review_data, None
        
    except Exception as e:
        return None, str(e)


# --- 5. Chat System Prompt ---
CHAT_SYSTEM_PROMPT = """You are an **Expert Product Reviewer and Technical Consultant** with deep knowledge of aspect-based sentiment analysis.

**Context:** You've provided a comprehensive review including sentiment analysis across multiple product aspects (e.g., design, performance, battery, camera, etc.).

**Your Role:**
- Answer questions about specific aspects and their sentiments
- Explain why certain aspects scored high or low
- Compare aspects with competing products
- Provide detailed technical explanations
- Reference the sentiment scores when relevant

**Available Data:**
- Overall rating and sentiment
- Aspect-based sentiment scores (-1.0 to +1.0)
- Confidence levels for each aspect
- Key mentions and quotes from sources

**Conversation Guidelines:**
1. When asked about specific features, reference the aspect sentiment score
2. Explain trade-offs between different aspects
3. Be honest about weak points (negative sentiment aspects)
4. Provide context for why aspects scored the way they did
5. Suggest alternatives if major aspects score poorly

Example responses:
- "The camera scored +0.8/1.0, which is excellent. Key strengths include..."
- "Battery life has a sentiment of -0.4, indicating below-average performance compared to competitors..."

Keep responses conversational but data-driven (2-4 paragraphs unless more detail is requested)."""


# --- 6. Display Review Function ---
def display_review(review_data):
    """Display the structured review with sentiment visualizations"""
    st.markdown("---")
    
    # Verdict
    st.markdown("### ✅ Final Verdict")
    st.write(review_data.get('verdict', 'No final verdict provided.'))
    
    st.markdown("---")
    
    # Sources
    if review_data.get('sources'):
        with st.expander("📚 Sources Used"):
            for i, source in enumerate(review_data['sources'], 1):
                if source.startswith('http'):
                    st.markdown(f"{i}. [{source}]({source})")
                else:
                    st.markdown(f"{i}. {source}")


# --- 7. Helper Functions ---
def generate_initial_review(product_name, use_web_search=True):
    """Generate the initial structured review with sentiment analysis"""
    
    if use_web_search:
        try:
            searcher = CompleteFreeWebSearch(groq_api_key)
            review_data = searcher.generate_review(product_name)
            
            if review_data:
                return review_data, None
            else:
                st.warning("⚠️ Web search failed. Falling back to AI knowledge...")
        except Exception as e:
            st.warning(f"⚠️ Web search error: {str(e)}. Using AI knowledge...")
    
    # Fallback to AI knowledge
    return generate_ai_knowledge_review(product_name, client)


def chat_with_ai(user_message, conversation_history):
    """Continue the conversation about the product"""
    
    try:
        # Build messages with context
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1000
        )
        
        response = chat_completion.choices[0].message.content
        return response, None
        
    except Exception as e:
        return None, str(e)


def reset_conversation():
    """Reset the chat session"""
    st.session_state.messages = []
    st.session_state.current_product = None
    st.session_state.review_data = None
    st.session_state.chat_mode = False
    st.session_state.comparison_products = []
    st.session_state.show_comparison = False


# --- 8. Streamlit Application Interface ---

# Page Configuration
st.set_page_config(
    page_title="AI Product Review with Sentiment Analysis",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .source-badge {
        display: inline-block;
        padding: 4px 12px;
        background: #e3f2fd;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
    }
    .metric-card {
        background: #f5f5f5;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_product" not in st.session_state:
    st.session_state.current_product = None
if "review_data" not in st.session_state:
    st.session_state.review_data = None
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False
if "comparison_products" not in st.session_state:
    st.session_state.comparison_products = []
if "show_comparison" not in st.session_state:
    st.session_state.show_comparison = False

# Groq Client Initialization
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("groq_api_key")
    if not groq_api_key:
        st.error("❌ Error: GROQ_API_KEY not found in `.streamlit/secrets.toml`")
        st.info("Please add: `GROQ_API_KEY = 'your-key-here'` to `.streamlit/secrets.toml`")
        st.stop()
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"❌ Error initializing Groq client: {str(e)}")
    st.stop()


# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Product Review Chat")
    st.markdown("---")
    
    if st.session_state.current_product:
        st.success(f"**Current Product:**\n{st.session_state.current_product}")
        st.markdown("---")
        
        # Show quick stats
        if st.session_state.review_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rating", st.session_state.review_data.get('predicted_rating', 'N/A'))
            with col2:
                source_type = st.session_state.review_data.get('data_source_type', 'unknown')
                if source_type == 'free_web_search':
                    st.metric("Sources", len(st.session_state.review_data.get('sources', [])))
                else:
                    st.metric("Source", "AI KB")
            
            # Aspect sentiment summary
            if 'aspect_sentiments' in st.session_state.review_data:
                aspects = st.session_state.review_data['aspect_sentiments']
                avg_sentiment = sum(a['sentiment_score'] for a in aspects) / len(aspects)
                st.metric("Avg Sentiment", f"{avg_sentiment:+.2f}")
            
            st.metric("Pros", len(st.session_state.review_data.get('pros', [])))
            st.metric("Cons", len(st.session_state.review_data.get('cons', [])))
        
        st.markdown("---")
        
        # Comparison feature
        if st.button("➕ Add to Comparison", use_container_width=True):
            if st.session_state.review_data not in st.session_state.comparison_products:
                st.session_state.comparison_products.append(st.session_state.review_data)
                st.success(f"Added! ({len(st.session_state.comparison_products)} products)")
        
        if len(st.session_state.comparison_products) >= 2:
            if st.button("📊 View Comparison", use_container_width=True):
                st.session_state.show_comparison = True
        
        if st.session_state.comparison_products:
            if st.button("🗑️ Clear Comparison", use_container_width=True):
                st.session_state.comparison_products = []
                st.session_state.show_comparison = False
        
        st.markdown("---")
        
        if st.button("🔄 Review Different Product", use_container_width=True):
            reset_conversation()
            st.rerun()
    else:
        st.info("👈 Enter a product name to start")
    
    st.markdown("---")
    
    # Tips section
    with st.expander("💡 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. Enter a product name
        2. Choose data source (Web or AI)
        3. Get instant review with sentiment analysis
        4. Ask follow-up questions
        
        **Data Sources:**
        - 🌐 **Web Search**: Current, real-time data (recommended)
        - 🤖 **AI Knowledge**: Fast but may be outdated
        
        **Sentiment Analysis:**
        - View aspect-based sentiment scores
        - See radar charts and visualizations
        - Understand product strengths/weaknesses
        
        **Example Questions:**
        - "How does it compare to [competitor]?"
        - "Why did the battery score low?"
        - "What aspect scored highest?"
        - "Is it good for gaming?"
        """)
    
    with st.expander("📝 Suggested Questions"):
        suggestions = [
            "Compare with alternatives",
            "Explain sentiment scores",
            "Best use cases",
            "Value for money",
            "Why certain aspects scored low",
            "Which aspects are strongest"
        ]
        for suggestion in suggestions:
            st.markdown(f"• {suggestion}")
    
    st.markdown("---")
    st.caption("🆓 100% Free • No API costs")
    st.caption("🌐 Web Search: DuckDuckGo")
    st.caption("🤖 AI: Groq Llama 3.3 70B")
    st.caption("📊 Sentiment: Aspect-Based Analysis")


# --- Main Content Area ---

# Show comparison view if requested
if st.session_state.show_comparison and len(st.session_state.comparison_products) >= 2:
    st.title("📊 Product Comparison")
    
    if st.button("⬅️ Back to Review"):
        st.session_state.show_comparison = False
        st.rerun()
    
    st.markdown("---")
    
    # Comparison radar chart
    comparison_fig = create_comparison_chart(st.session_state.comparison_products)
    if comparison_fig:
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Side-by-side comparison
    st.markdown("### Detailed Comparison")
    cols = st.columns(len(st.session_state.comparison_products))
    
    for idx, (col, product) in enumerate(zip(cols, st.session_state.comparison_products)):
        with col:
            st.markdown(f"#### {product.get('product_name', 'Unknown')}")
            st.metric("Rating", product.get('predicted_rating', 'N/A'))
            st.metric("Price", product.get('price_info', 'N/A'))
            
            if 'aspect_sentiments' in product:
                avg_sentiment = sum(a['sentiment_score'] for a in product['aspect_sentiments']) / len(product['aspect_sentiments'])
                st.metric("Avg Sentiment", f"{avg_sentiment:+.2f}")
            
            with st.expander("View Details"):
                st.markdown("**Pros:**")
                for pro in product.get('pros', [])[:3]:
                    st.markdown(f"• {pro}")
                
                st.markdown("**Cons:**")
                for con in product.get('cons', [])[:3]:
                    st.markdown(f"• {con}")

elif not st.session_state.chat_mode:
    # Initial product search interface
    st.title("🤖 AI Product Review Assistant")
    st.markdown("### Get expert reviews with aspect-based sentiment analysis")
    
    # Product input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        product_input = st.text_input(
            "Enter Product Name",
            placeholder="e.g., Sony WH-1000XM5, MacBook Pro M3, Nintendo Switch OLED",
            label_visibility="collapsed"
        )
    
    # Data source selection
    data_source = st.radio(
        "Choose Data Source:",
        ["🌐 Web Search (Real-time, Accurate - Recommended)", "🤖 AI Knowledge (Fast, May be outdated)"],
        horizontal=True,
        help="Web Search scrapes current product info from the internet. AI Knowledge uses training data from January 2025."
    )
    
    use_web = data_source.startswith("🌐")
    
    with col2:
        search_button = st.button("🔍 Analyze", use_container_width=True, type="primary")
    
    # Example products
    st.markdown("**Popular Products:**")
    example_cols = st.columns(4)
    examples = [
        "iPhone 15 Pro",
        "Sony WH-1000XM5",
        "iPad Pro M4",
        "Nintendo Switch"
    ]
    
    for idx, example in enumerate(examples):
        with example_cols[idx]:
            if st.button(example, use_container_width=True):
                product_input = example
                search_button = True
    
    # Info box
    st.info("""
    **🌐 Web Search Mode**: Searches DuckDuckGo and analyzes current product information from multiple sources. Takes 10-20 seconds but provides accurate, up-to-date data with aspect-based sentiment analysis.
    
    **🤖 AI Knowledge Mode**: Fast responses using AI training data (updated January 2025). Instant results but may not reflect latest specifications or pricing.
    
    **📊 Sentiment Analysis**: Get detailed sentiment scores for 5-8 key product aspects (e.g., Design, Performance, Battery, Camera) with confidence levels and visualizations.
    """)
    
    if search_button and product_input:
        with st.spinner(f"{'🔍 Searching the web and analyzing' if use_web else '🤖 Analyzing'} '{product_input}'..."):
            review_data, error = generate_initial_review(product_input, use_web_search=use_web)
            
            if error:
                st.error(f"❌ Error: {error}")
            elif review_data:
                st.session_state.current_product = product_input
                st.session_state.review_data = review_data
                st.session_state.chat_mode = True
                
                # Build review summary with sentiment info
                review_summary = f"""I've analyzed the {review_data.get('product_name')}. Here's my comprehensive review:

**Rating:** {review_data.get('predicted_rating')}

**Price:** {review_data.get('price_info', 'Price varies')}

**Key Specs:** {review_data.get('specifications_inferred')}

**Aspect-Based Sentiment Analysis:**"""
                
                if 'aspect_sentiments' in review_data and review_data['aspect_sentiments']:
                    for aspect in sorted(review_data['aspect_sentiments'], key=lambda x: x['sentiment_score'], reverse=True)[:5]:
                        score = aspect['sentiment_score']
                        emoji = "🟢" if score > 0.3 else "🟡" if score > -0.3 else "🔴"
                        review_summary += f"\n- {emoji} {aspect['aspect']}: {score:+.2f} (Confidence: {aspect['confidence']:.0%})"
                
                review_summary += f"""

**Strengths:** {', '.join(review_data.get('pros', [])[:3])}

**Weaknesses:** {', '.join(review_data.get('cons', [])[:3])}

**Verdict:** {review_data.get('verdict')}

**Data Source:** {review_data.get('data_source_type', 'unknown').replace('_', ' ').title()}

Feel free to ask me any questions about this product, specific aspects, or comparisons with competitors!"""
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": review_summary,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                
                st.rerun()

else:
    # Chat interface
    st.title(f"💬 Chat about: {st.session_state.current_product}")
    
    # Display the structured review at the top
    with st.expander("📊 View Full Review with Sentiment Analysis", expanded=False):
        if st.session_state.review_data:
            display_review(st.session_state.review_data)
    
    st.markdown("---")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(message.get("timestamp", ""))
    
    # Suggested questions (only show if few messages)
    if len(st.session_state.messages) <= 1:
        st.markdown("**💡 Try asking:**")
        suggestion_cols = st.columns(3)
        quick_questions = [
            f"How does {st.session_state.current_product} compare to competitors?",
            f"Why did certain aspects score low?",
            f"Which aspect is the strongest?"
        ]
        
        for idx, question in enumerate(quick_questions):
            with suggestion_cols[idx]:
                if st.button(question, key=f"quick_{idx}"):
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    
                    # Get AI response
                    with st.spinner("🤔 Thinking..."):
                        conversation_history = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state.messages[:-1]
                        ]
                        
                        response, error = chat_with_ai(question, conversation_history)
                        
                        if error:
                            st.error(f"Error: {error}")
                            st.session_state.messages.pop()
                        elif response:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "timestamp": datetime.now().strftime("%I:%M %p")
                            })
                            st.rerun()
    
    # Chat input
    user_input = st.chat_input("Ask anything about this product...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            # Build conversation context (exclude timestamps)
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]
            
            response, error = chat_with_ai(user_input, conversation_history)
            
            if error:
                st.error(f"Error: {error}")
                # Remove the failed user message
                st.session_state.messages.pop()
            elif response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                st.rerun()
    
    # Header with rating and data source
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.header(f"📱 {review_data.get('product_name', 'Unknown Product')}")
    with col2:
        rating = review_data.get('predicted_rating', 'N/A')
        st.markdown(f"### ⭐ {rating}")
    with col3:
        source_type = review_data.get('data_source_type', 'unknown')
        if source_type == 'free_web_search':
            st.success("🌐 Live Web Data")
        else:
            st.info("🤖 AI Knowledge")
    
    # Data source info
    if review_data.get('data_source_type') == 'free_web_search':
        st.success(f"✅ Information verified from {len(review_data.get('sources', []))} web sources on {review_data.get('last_updated', 'today')}")
    else:
        st.warning(f"⚠️ Based on AI training data (updated January 2025). Please verify current specifications and pricing with official sources.")
    
    # Price and specs
    col_price, col_specs = st.columns([1, 2])
    
    with col_price:
        st.markdown("### 💰 Pricing")
        st.info(review_data.get('price_info', 'Price not available'))
    
    with col_specs:
        st.markdown("### 🔧 Key Specifications")
        st.info(review_data.get('specifications_inferred', 'Specifications could not be determined.'))
    
    st.markdown("---")
    
    # Aspect-Based Sentiment Analysis Section
    if 'aspect_sentiments' in review_data and review_data['aspect_sentiments']:
        st.markdown("## 📊 Aspect-Based Sentiment Analysis")
        
        # Visualizations in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Radar Chart", "📊 Bar Chart", "🥧 Distribution", "📋 Details"])
        
        with tab1:
            radar_fig = create_aspect_radar_chart(review_data['aspect_sentiments'])
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with tab2:
            bar_fig = create_sentiment_bar_chart(review_data['aspect_sentiments'])
            st.plotly_chart(bar_fig, use_container_width=True)
        
        with tab3:
            pie_fig = create_sentiment_distribution_pie(review_data['aspect_sentiments'])
            st.plotly_chart(pie_fig, use_container_width=True)
        
        with tab4:
            details_df = create_aspect_details_table(review_data['aspect_sentiments'])
            st.dataframe(details_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
    
    # Pros and Cons
    col_pros, col_cons = st.columns(2)
    
    with col_pros:
        st.markdown("### 🟢 Strengths")
        for i, pro in enumerate(review_data.get('pros', []), 1):
            st.markdown(f"**{i}.** {pro}")
    
    with col_cons:
        st.markdown("### 🔴 Weaknesses")
        for i, con in enumerate(review_data.get('cons', []), 1):
            st.markdown(f"**{i}.** {con}")
    
    st.markdown("---")
    
    # Verdict
    st.markdown("### ✅ Final Verdict")
    st.write(review_data.get('verdict', 'No final verdict provided.'))
    
    st.markdown("---")
    
    # Sources
    if review_data.get('sources'):
        with st.expander("📚 Sources Used"):
            for i, source in enumerate(review_data['sources'], 1):
                if source.startswith('http'):
                    st.markdown(f"{i}. [{source}]({source})")
                else:
                    st.markdown(f"{i}. {source}")


# --- 7. Helper Functions ---
def generate_initial_review(product_name, use_web_search=True):
    """Generate the initial structured review with sentiment analysis"""
    
    if use_web_search:
        try:
            searcher = CompleteFreeWebSearch(groq_api_key)
            review_data = searcher.generate_review(product_name)
            
            if review_data:
                return review_data, None
            else:
                st.warning("⚠️ Web search failed. Falling back to AI knowledge...")
        except Exception as e:
            st.warning(f"⚠️ Web search error: {str(e)}. Using AI knowledge...")
    
    # Fallback to AI knowledge
    return generate_ai_knowledge_review(product_name, client)


def chat_with_ai(user_message, conversation_history):
    """Continue the conversation about the product"""
    
    try:
        # Build messages with context
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1000
        )
        
        response = chat_completion.choices[0].message.content
        return response, None
        
    except Exception as e:
        return None, str(e)


def reset_conversation():
    """Reset the chat session"""
    st.session_state.messages = []
    st.session_state.current_product = None
    st.session_state.review_data = None
    st.session_state.chat_mode = False
    st.session_state.comparison_products = []
    st.session_state.show_comparison = False


# --- 8. Streamlit Application Interface ---

# Page Configuration
st.set_page_config(
    page_title="AI Product Review with Sentiment Analysis",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .source-badge {
        display: inline-block;
        padding: 4px 12px;
        background: #e3f2fd;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
    }
    .metric-card {
        background: #f5f5f5;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_product" not in st.session_state:
    st.session_state.current_product = None
if "review_data" not in st.session_state:
    st.session_state.review_data = None
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False
if "comparison_products" not in st.session_state:
    st.session_state.comparison_products = []
if "show_comparison" not in st.session_state:
    st.session_state.show_comparison = False

# Groq Client Initialization
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("groq_api_key")
    if not groq_api_key:
        st.error("❌ Error: GROQ_API_KEY not found in `.streamlit/secrets.toml`")
        st.info("Please add: `GROQ_API_KEY = 'your-key-here'` to `.streamlit/secrets.toml`")
        st.stop()
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"❌ Error initializing Groq client: {str(e)}")
    st.stop()


# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Product Review Chat")
    st.markdown("---")
    
    if st.session_state.current_product:
        st.success(f"**Current Product:**\n{st.session_state.current_product}")
        st.markdown("---")
        
        # Show quick stats
        if st.session_state.review_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rating", st.session_state.review_data.get('predicted_rating', 'N/A'))
            with col2:
                source_type = st.session_state.review_data.get('data_source_type', 'unknown')
                if source_type == 'free_web_search':
                    st.metric("Sources", len(st.session_state.review_data.get('sources', [])))
                else:
                    st.metric("Source", "AI KB")
            
            # Aspect sentiment summary
            if 'aspect_sentiments' in st.session_state.review_data:
                aspects = st.session_state.review_data['aspect_sentiments']
                avg_sentiment = sum(a['sentiment_score'] for a in aspects) / len(aspects)
                st.metric("Avg Sentiment", f"{avg_sentiment:+.2f}")
            
            st.metric("Pros", len(st.session_state.review_data.get('pros', [])))
            st.metric("Cons", len(st.session_state.review_data.get('cons', [])))
        
        st.markdown("---")
        
        # Comparison feature
        if st.button("➕ Add to Comparison", use_container_width=True):
            if st.session_state.review_data not in st.session_state.comparison_products:
                st.session_state.comparison_products.append(st.session_state.review_data)
                st.success(f"Added! ({len(st.session_state.comparison_products)} products)")
        
        if len(st.session_state.comparison_products) >= 2:
            if st.button("📊 View Comparison", use_container_width=True):
                st.session_state.show_comparison = True
                st.rerun()
        
        if st.session_state.comparison_products:
            if st.button("🗑️ Clear Comparison", use_container_width=True):
                st.session_state.comparison_products = []
                st.session_state.show_comparison = False
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🔄 Review Different Product", use_container_width=True):
            reset_conversation()
            st.rerun()
    else:
        st.info("👈 Enter a product name to start")
    
    st.markdown("---")
    
    # Tips section
    with st.expander("💡 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. Enter a product name
        2. Choose data source (Web or AI)
        3. Get instant review with sentiment analysis
        4. Ask follow-up questions
        
        **Data Sources:**
        - 🌐 **Web Search**: Current, real-time data (recommended)
        - 🤖 **AI Knowledge**: Fast but may be outdated
        
        **Sentiment Analysis:**
        - View aspect-based sentiment scores
        - See radar charts and visualizations
        - Understand product strengths/weaknesses
        
        **Example Questions:**
        - "How does it compare to [competitor]?"
        - "Why did the battery score low?"
        - "What aspect scored highest?"
        - "Is it good for gaming?"
        """)
    
    with st.expander("📝 Suggested Questions"):
        suggestions = [
            "Compare with alternatives",
            "Explain sentiment scores",
            "Best use cases",
            "Value for money",
            "Why certain aspects scored low",
            "Which aspects are strongest"
        ]
        for suggestion in suggestions:
            st.markdown(f"• {suggestion}")
    
    st.markdown("---")
    st.caption("🆓 100% Free • No API costs")
    st.caption("🌐 Web Search: DuckDuckGo")
    st.caption("🤖 AI: Groq Llama 3.3 70B")
    st.caption("📊 Sentiment: Aspect-Based Analysis")


# --- Main Content Area ---

# Show comparison view if requested
if st.session_state.show_comparison and len(st.session_state.comparison_products) >= 2:
    st.title("📊 Product Comparison")
    
    if st.button("⬅️ Back to Review"):
        st.session_state.show_comparison = False
        st.rerun()
    
    st.markdown("---")
    
    # Comparison radar chart
    comparison_fig = create_comparison_chart(st.session_state.comparison_products)
    if comparison_fig:
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Side-by-side comparison
    st.markdown("### Detailed Comparison")
    cols = st.columns(len(st.session_state.comparison_products))
    
    for idx, (col, product) in enumerate(zip(cols, st.session_state.comparison_products)):
        with col:
            st.markdown(f"#### {product.get('product_name', 'Unknown')}")
            st.metric("Rating", product.get('predicted_rating', 'N/A'))
            st.metric("Price", product.get('price_info', 'N/A'))
            
            if 'aspect_sentiments' in product:
                avg_sentiment = sum(a['sentiment_score'] for a in product['aspect_sentiments']) / len(product['aspect_sentiments'])
                st.metric("Avg Sentiment", f"{avg_sentiment:+.2f}")
            
            with st.expander("View Details"):
                st.markdown("**Pros:**")
                for pro in product.get('pros', [])[:3]:
                    st.markdown(f"• {pro}")
                
                st.markdown("**Cons:**")
                for con in product.get('cons', [])[:3]:
                    st.markdown(f"• {con}")

elif not st.session_state.chat_mode:
    # Initial product search interface
    st.title("🤖 AI Product Review Assistant")
    st.markdown("### Get expert reviews with aspect-based sentiment analysis")
    
    # Product input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        product_input = st.text_input(
            "Enter Product Name",
            placeholder="e.g., Sony WH-1000XM5, MacBook Pro M3, Nintendo Switch OLED",
            label_visibility="collapsed"
        )
    
    # Data source selection
    data_source = st.radio(
        "Choose Data Source:",
        ["🌐 Web Search (Real-time, Accurate - Recommended)", "🤖 AI Knowledge (Fast, May be outdated)"],
        horizontal=True,
        help="Web Search scrapes current product info from the internet. AI Knowledge uses training data from January 2025."
    )
    
    use_web = data_source.startswith("🌐")
    
    with col2:
        search_button = st.button("🔍 Analyze", use_container_width=True, type="primary")
    
    # Example products
    st.markdown("**Popular Products:**")
    example_cols = st.columns(4)
    examples = [
        "iPhone 15 Pro",
        "Sony WH-1000XM5",
        "iPad Pro M4",
        "Nintendo Switch"
    ]
    
    for idx, example in enumerate(examples):
        with example_cols[idx]:
            if st.button(example, use_container_width=True):
                product_input = example
                search_button = True
    
    # Info box
    st.info("""
    **🌐 Web Search Mode**: Searches DuckDuckGo and analyzes current product information from multiple sources. Takes 10-20 seconds but provides accurate, up-to-date data with aspect-based sentiment analysis.
    
    **🤖 AI Knowledge Mode**: Fast responses using AI training data (updated January 2025). Instant results but may not reflect latest specifications or pricing.
    
    **📊 Sentiment Analysis**: Get detailed sentiment scores for 5-8 key product aspects (e.g., Design, Performance, Battery, Camera) with confidence levels and visualizations.
    """)
    
    if search_button and product_input:
        with st.spinner(f"{'🔍 Searching the web and analyzing' if use_web else '🤖 Analyzing'} '{product_input}'..."):
            review_data, error = generate_initial_review(product_input, use_web_search=use_web)
            
            if error:
                st.error(f"❌ Error: {error}")
            elif review_data:
                st.session_state.current_product = product_input
                st.session_state.review_data = review_data
                st.session_state.chat_mode = True
                
                # Build review summary with sentiment info
                review_summary = f"""I've analyzed the {review_data.get('product_name')}. Here's my comprehensive review:

**Rating:** {review_data.get('predicted_rating')}

**Price:** {review_data.get('price_info', 'Price varies')}

**Key Specs:** {review_data.get('specifications_inferred')}

**Aspect-Based Sentiment Analysis:**"""
                
                if 'aspect_sentiments' in review_data and review_data['aspect_sentiments']:
                    for aspect in sorted(review_data['aspect_sentiments'], key=lambda x: x['sentiment_score'], reverse=True)[:5]:
                        score = aspect['sentiment_score']
                        emoji = "🟢" if score > 0.3 else "🟡" if score > -0.3 else "🔴"
                        review_summary += f"\n- {emoji} {aspect['aspect']}: {score:+.2f} (Confidence: {aspect['confidence']:.0%})"
                
                review_summary += f"""

**Strengths:** {', '.join(review_data.get('pros', [])[:3])}

**Weaknesses:** {', '.join(review_data.get('cons', [])[:3])}

**Verdict:** {review_data.get('verdict')}

**Data Source:** {review_data.get('data_source_type', 'unknown').replace('_', ' ').title()}

Feel free to ask me any questions about this product, specific aspects, or comparisons with competitors!"""
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": review_summary,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                
                st.rerun()

else:
    # Chat interface
    st.title(f"💬 Chat about: {st.session_state.current_product}")
    
    # Display the structured review at the top
    with st.expander("📊 View Full Review with Sentiment Analysis", expanded=False):
        if st.session_state.review_data:
            display_review(st.session_state.review_data)
    
    st.markdown("---")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(message.get("timestamp", ""))
    
    # Suggested questions (only show if few messages)
    if len(st.session_state.messages) <= 1:
        st.markdown("**💡 Try asking:**")
        suggestion_cols = st.columns(3)
        quick_questions = [
            f"How does {st.session_state.current_product} compare to competitors?",
            f"Why did certain aspects score low?",
            f"Which aspect is the strongest?"
        ]
        
        for idx, question in enumerate(quick_questions):
            with suggestion_cols[idx]:
                if st.button(question, key=f"quick_{idx}"):
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    
                    # Get AI response
                    with st.spinner("🤔 Thinking..."):
                        conversation_history = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state.messages[:-1]
                        ]
                        
                        response, error = chat_with_ai(question, conversation_history)
                        
                        if error:
                            st.error(f"Error: {error}")
                            st.session_state.messages.pop()
                        elif response:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "timestamp": datetime.now().strftime("%I:%M %p")
                            })
                            st.rerun()
    
    # Chat input
    user_input = st.chat_input("Ask anything about this product...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            # Build conversation context (exclude timestamps)
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]
            
            response, error = chat_with_ai(user_input, conversation_history)
            
            if error:
                st.error(f"Error: {error}")
                # Remove the failed user message
                st.session_state.messages.pop()
            elif response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                st.rerun()
