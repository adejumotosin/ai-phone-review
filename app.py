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

# --- 1. Pydantic Schema for Structured Output ---
class ProductReview(BaseModel):
    """A comprehensive product review based on real web data."""
    product_name: str = Field(description="The full name of the product being reviewed.")
    specifications_inferred: str = Field(description="A concise summary of the key technical specs (e.g., '6.1-inch OLED, A16 Bionic, 48MP main camera, $999 USD').")
    predicted_rating: str = Field(description="A critical rating out of 5.0 (e.g., '4.6 / 5.0').")
    pros: List[str] = Field(description="A list of strengths and advantages.")
    cons: List[str] = Field(description="A list of weaknesses, trade-offs, or user pain points.")
    verdict: str = Field(description="A concluding summary of the product's overall value proposition.")
    price_info: Optional[str] = Field(default="Price not available", description="Current pricing information if found.")
    sources: List[str] = Field(default=[], description="List of source URLs used.")
    last_updated: str = Field(default="", description="Date when information was gathered.")
    data_source_type: str = Field(default="web_search", description="Type of data source used.")


# --- 2. Free Web Search Class ---
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
    
    def search_duckduckgo(self, query, num_results=5):
        """Free DuckDuckGo search"""
        
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
                    # Clean the URL
                    url = link.get('href', '')
                    if url.startswith('//'):
                        url = 'https:' + url
                    
                    results.append({
                        'title': link.text.strip(),
                        'url': url,
                        'snippet': snippet.text.strip()
                    })
            
            # Cache results
            if results:
                self._cache_search(query, results)
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def get_page_content(self, url, max_chars=5000):
        """Scrape page content"""
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
                text = main_content.get_text(separator=' ', strip=True)
                # Clean up whitespace
                text = ' '.join(text.split())
                return text[:max_chars]
            
            return None
            
        except Exception as e:
            st.warning(f"Failed to fetch {url}: {str(e)}")
            return None
    
    def generate_review(self, product_name):
        """Complete pipeline: search + scrape + analyze"""
        
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
                content = self.get_page_content(result['url'])
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
        
        # Step 3: Build comprehensive context
        context = f"# Product Review Request: {product_name}\n\n"
        context += f"## Web Search Results ({len(search_results)} sources found):\n\n"
        
        for i, result in enumerate(search_results, 1):
            context += f"**{i}. {result['title']}**\n"
            context += f"   Summary: {result['snippet']}\n"
            context += f"   URL: {result['url']}\n\n"
        
        if detailed_content:
            context += f"\n## Detailed Content from Top Sources:\n\n"
            for i, item in enumerate(detailed_content, 1):
                context += f"### Source {i}: {item['title']}\n"
                context += f"URL: {item['url']}\n\n"
                context += f"{item['content'][:3000]}\n\n"
                context += "---\n\n"
        
        # Step 4: Generate review with Groq
        st.info("🤖 Analyzing information and generating review...")
        
        system_prompt = """You are an expert product reviewer and analyst. Your task is to create a comprehensive, critical product review based STRICTLY on the web search results provided.

**Critical Instructions:**
1. Use ONLY information from the provided sources
2. Be specific and reference actual features/specs found in the sources
3. If pricing is mentioned, include it
4. Be balanced - mention both strengths and weaknesses
5. If information is conflicting, note it
6. Do NOT fabricate information not present in the sources
7. Rate the product fairly based on the information available

Generate your review in valid JSON format matching the exact schema provided."""

        user_prompt = f"""Based on the following CURRENT web information gathered on {datetime.now().strftime('%B %d, %Y')}, create a comprehensive product review.

{context}

Generate a JSON response with this EXACT structure:
{{
  "product_name": "Full product name as found in sources",
  "specifications_inferred": "Concise summary of key specs found (e.g., 'Display: X, Processor: Y, Camera: Z, Storage: W')",
  "predicted_rating": "X.X / 5.0 (based on analysis)",
  "pros": ["Specific advantage 1 from sources", "Specific advantage 2", "etc"],
  "cons": ["Specific disadvantage 1 from sources", "Specific disadvantage 2", "etc"],
  "verdict": "Comprehensive concluding paragraph about value proposition and target audience",
  "price_info": "Current pricing if found in sources, otherwise 'Price varies by retailer'",
  "sources": {json.dumps(sources_used)},
  "last_updated": "{datetime.now().strftime('%Y-%m-%d')}",
  "data_source_type": "free_web_search"
}}

Important: Be critical and honest. If the product has issues mentioned in sources, include them."""

        try:
            response = self.groq.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2500
            )
            
            review = json.loads(response.choices[0].message.content)
            st.success("✅ Review generated successfully from web sources!")
            
            return review
            
        except Exception as e:
            st.error(f"Failed to generate review: {str(e)}")
            return None


# --- 3. Fallback AI Knowledge Review ---
INITIAL_REVIEW_PROMPT = f"""
You are an **Expert Product Reviewer and Critical Market Analyst**. Your task is to generate a comprehensive, structured product review.

**Process:**
1. Use your knowledge base to recall common specifications for the product.
2. Provide a critical, balanced review noting trade-offs and competitive positioning.

**Important:**
- State clearly that this is based on AI training data (updated through January 2025)
- Recommend users verify current specifications and pricing
- Be honest about what you don't know

**Format your entire output as a single JSON object** conforming to this schema:
{ProductReview.schema_json(indent=2)}
"""

def generate_ai_knowledge_review(product_name, client):
    """Fallback to AI knowledge when web search fails"""
    
    user_prompt = f"""Generate a detailed review for the product: {product_name}

Important: Make it clear this is based on training data. Include disclaimers about verifying current info."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": INITIAL_REVIEW_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        review_data = json.loads(chat_completion.choices[0].message.content)
        review_data['data_source_type'] = 'ai_knowledge'
        review_data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        review_data['sources'] = ['AI Training Data (Updated January 2025)']
        
        return review_data, None
        
    except Exception as e:
        return None, str(e)


# --- 4. Chat System Prompt ---
CHAT_SYSTEM_PROMPT = """You are an **Expert Product Reviewer and Technical Consultant**. You have just provided a comprehensive review of a product, and now the user has follow-up questions.

**Your Role:**
- Answer questions about the product with expert knowledge
- Provide comparisons with similar products when asked
- Explain technical specifications in detail
- Give purchasing advice and recommendations
- Discuss use cases and real-world performance
- Be conversational but maintain your expertise

**Guidelines:**
1. Draw from your extensive knowledge about the product and its market
2. Be honest about limitations and trade-offs
3. Provide specific examples and scenarios
4. If asked to compare, give balanced pros/cons for each option
5. Keep responses concise but informative (2-4 paragraphs unless more detail is requested)
6. Reference the initial review context when relevant

**Conversation Style:**
- Professional but friendly
- Use analogies to explain complex features
- Ask clarifying questions if the user's query is ambiguous
- Proactively suggest related information that might be helpful
"""


# --- 5. Streamlit Application Interface ---

# Page Configuration
st.set_page_config(
    page_title="AI Product Review Chat",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
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

# Groq Client Initialization
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=groq_api_key)
except Exception:
    st.error("❌ Error: Groq API key not found in `.streamlit/secrets.toml`. Please set it up.")
    st.stop()


# --- Helper Functions ---

def generate_initial_review(product_name, use_web_search=True):
    """Generate the initial structured review"""
    
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


def display_review(review_data):
    """Display the structured review"""
    st.markdown("---")
    
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
                st.markdown(f"{i}. [{source}]({source})")


def reset_conversation():
    """Reset the chat session"""
    st.session_state.messages = []
    st.session_state.current_product = None
    st.session_state.review_data = None
    st.session_state.chat_mode = False


# --- UI Layout ---

# Sidebar
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
            
            st.metric("Pros", len(st.session_state.review_data.get('pros', [])))
            st.metric("Cons", len(st.session_state.review_data.get('cons', [])))
        
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
        3. Get instant review
        4. Ask follow-up questions
        
        **Data Sources:**
        - 🌐 **Web Search**: Current, real-time data (recommended)
        - 🤖 **AI Knowledge**: Fast but may be outdated
        
        **Example Questions:**
        - "How does it compare to [competitor]?"
        - "Is it good for gaming?"
        - "What about battery life?"
        - "Should I wait for the next version?"
        """)
    
    with st.expander("📝 Suggested Questions"):
        suggestions = [
            "Compare with alternatives",
            "Best use cases",
            "Value for money",
            "Long-term reliability",
            "Setup and learning curve",
            "Compatibility issues"
        ]
        for suggestion in suggestions:
            st.markdown(f"• {suggestion}")
    
    st.markdown("---")
    st.caption("🆓 100% Free • No API costs")
    st.caption("🌐 Web Search: DuckDuckGo")
    st.caption("🤖 AI: Groq Llama 3.3 70B")


# Main Content Area
if not st.session_state.chat_mode:
    # Initial product search interface
    st.title("🤖 AI Product Review Assistant")
    st.markdown("### Get expert reviews with real-time web data or AI knowledge")
    
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
    **🌐 Web Search Mode**: Searches DuckDuckGo and analyzes current product information from multiple sources. Takes 10-20 seconds but provides accurate, up-to-date data.
    
    **🤖 AI Knowledge Mode**: Fast responses using AI training data (updated January 2025). Instant results but may not reflect latest specifications or pricing.
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
                
                # Add initial review to conversation history
                review_summary = f"""I've analyzed the {review_data.get('product_name')}. Here's my review:

**Rating:** {review_data.get('predicted_rating')}

**Price:** {review_data.get('price_info', 'Price varies')}

**Key Specs:** {review_data.get('specifications_inferred')}

**Strengths:** {', '.join(review_data.get('pros', [])[:3])}

**Weaknesses:** {', '.join(review_data.get('cons', [])[:3])}

**Verdict:** {review_data.get('verdict')}

**Data Source:** {review_data.get('data_source_type', 'unknown').replace('_', ' ').title()}

Feel free to ask me any questions about this product!"""
                
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
    with st.expander("📊 View Full Review", expanded=False):
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
            f"What are the best use cases for this product?",
            f"Is {st.session_state.current_product} worth the price?"
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
