import streamlit as st
import json
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import hashlib
from pathlib import Path
import logging
from dataclasses import dataclass
from functools import wraps
from contextlib import contextmanager
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

@dataclass
class AppConfig:
    """Centralized configuration management"""
    # API Settings
    model_name: str = "llama-3.3-70b-versatile"
    max_tokens_review: int = 2500
    max_tokens_chat: int = 1000
    temperature_review: float = 0.3
    temperature_chat: float = 0.7
    
    # Web Settings
    max_search_results: int = 5
    max_scrape_results: int = 3
    request_timeout: int = 10
    request_delay: float = 0.5
    max_content_length: int = 5000
    
    # Cache Settings
    cache_ttl_hours: int = 24
    cache_max_size: int = 100
    
    # UI Settings
    max_pros_cons_display: int = 10

class Constants:
    """Application constants"""
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ACCEPT_HEADER = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    ACCEPT_LANGUAGE = "en-US,en;q=0.5"

# =============================================================================
# DATA MODELS
# =============================================================================

class SearchResult(BaseModel):
    """Model for search results"""
    title: str
    url: str
    snippet: str
    domain: str = Field(default="", description="Domain of the URL")

class ScrapedContent(BaseModel):
    """Model for scraped web content"""
    url: str
    title: str
    content: str
    content_length: int
    scrape_timestamp: datetime

class ProductReview(BaseModel):
    """A comprehensive product review based on real web data."""
    product_name: str = Field(description="The full name of the product being reviewed.")
    specifications_inferred: str = Field(description="A concise summary of key technical specs.")
    predicted_rating: str = Field(description="A critical rating out of 5.0 (e.g., '4.6 / 5.0').")
    pros: List[str] = Field(description="A list of strengths and advantages.")
    cons: List[str] = Field(description="A list of weaknesses, trade-offs, or user pain points.")
    verdict: str = Field(description="A concluding summary of the product's overall value proposition.")
    price_info: str = Field(default="Price not available", description="Current pricing information if found.")
    sources: List[str] = Field(default=[], description="List of source URLs used.")
    last_updated: str = Field(default="", description="Date when information was gathered.")
    data_source_type: str = Field(default="web_search", description="Type of data source used.")
    
    @classmethod
    def from_ai_knowledge(cls, product_name: str) -> 'ProductReview':
        """Create a placeholder for AI knowledge-based reviews"""
        return cls(
            product_name=product_name,
            specifications_inferred="Based on AI training data (updated January 2025)",
            predicted_rating="N/A (AI Knowledge)",
            pros=["Information from AI training data"],
            cons=["May not reflect current specifications or pricing"],
            verdict="This review is based on AI training data. Please verify current information with official sources.",
            price_info="Price varies - check current retailers",
            sources=["AI Training Data (Updated January 2025)"],
            last_updated=datetime.now().strftime('%Y-%m-%d'),
            data_source_type="ai_knowledge"
        )

# =============================================================================
# EXCEPTIONS
# =============================================================================

class ProductReviewError(Exception):
    """Base exception for product review errors"""
    pass

class SearchError(ProductReviewError):
    """Search-related errors"""
    pass

class ScrapingError(ProductReviewError):
    """Web scraping errors"""
    pass

class AIGenerationError(ProductReviewError):
    """AI generation errors"""
    pass

class ValidationError(ProductReviewError):
    """Data validation errors"""
    pass

# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

class CacheManager:
    """Managed cache with TTL and size limits"""
    
    def __init__(self, cache_dir: Path = Path(".cache"), ttl_hours: int = 24, max_size: int = 100):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_hours = ttl_hours
        self.max_size = max_size
        
    def _get_cache_key(self, key_data: str) -> str:
        """Generate cache key from data"""
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _clean_old_cache(self):
        """Remove old cache files if over size limit"""
        cache_files = list(self.cache_dir.glob("*.json"))
        if len(cache_files) > self.max_size:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            for old_file in cache_files[:len(cache_files) - self.max_size]:
                try:
                    old_file.unlink()
                    logger.info(f"Cleaned old cache file: {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean cache file {old_file}: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check TTL
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                cache_file.unlink()  # Remove expired cache
                return None
                
            return cached_data['data']
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            try:
                cache_file.unlink()  # Remove corrupted cache
            except:
                pass
            return None
    
    def set(self, key: str, data: Any):
        """Cache data with timestamp"""
        try:
            self._clean_old_cache()
            
            cache_file = self.cache_dir / f"{key}.json"
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")

# =============================================================================
# WEB SEARCH COMPONENTS
# =============================================================================

class WebSearchClient:
    """Handles web search operations"""
    
    def __init__(self, cache_manager: CacheManager, config: AppConfig):
        self.cache = cache_manager
        self.config = config
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create configured HTTP session"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': Constants.USER_AGENT,
            'Accept': Constants.ACCEPT_HEADER,
            'Accept-Language': Constants.ACCEPT_LANGUAGE,
        })
        return session
    
    def search_products(self, product_name: str) -> List[SearchResult]:
        """Search for product information"""
        cache_key = self.cache._get_cache_key(f"search_{product_name}")
        cached_results = self.cache.get(cache_key)
        
        if cached_results:
            logger.info(f"Using cached search results for: {product_name}")
            return [SearchResult(**result) for result in cached_results]
        
        try:
            search_query = f"{product_name} specifications review price features"
            results = self._duckduckgo_search(search_query)
            
            if results:
                # Cache the raw dict data
                self.cache.set(cache_key, [result.dict() for result in results])
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for {product_name}: {e}")
            raise SearchError(f"Search failed: {str(e)}")
    
    def _duckduckgo_search(self, query: str) -> List[SearchResult]:
        """Perform DuckDuckGo search"""
        try:
            url = "https://html.duckduckgo.com/html/"
            data = {'q': query}
            
            response = self.session.post(url, data=data, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.find_all('div', class_='result')[:self.config.max_search_results]:
                link = result.find('a', class_='result__a')
                snippet = result.find('a', class_='result__snippet')
                
                if link and snippet:
                    url = self._clean_url(link.get('href', ''))
                    domain = self._extract_domain(url)
                    
                    results.append(SearchResult(
                        title=link.text.strip(),
                        url=url,
                        snippet=snippet.text.strip(),
                        domain=domain
                    ))
            
            return results
            
        except requests.RequestException as e:
            raise SearchError(f"Search request failed: {str(e)}")
        except Exception as e:
            raise SearchError(f"Search parsing failed: {str(e)}")
    
    def _clean_url(self, url: str) -> str:
        """Clean and format URL"""
        if url.startswith('//'):
            return 'https:' + url
        return url
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ""

class ContentScraper:
    """Handles web content scraping"""
    
    def __init__(self, cache_manager: CacheManager, config: AppConfig):
        self.cache = cache_manager
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': Constants.USER_AGENT})
    
    def scrape_content(self, search_results: List[SearchResult]) -> List[ScrapedContent]:
        """Scrape content from search results"""
        scraped_data = []
        
        for i, result in enumerate(search_results[:self.config.max_scrape_results]):
            try:
                content = self._scrape_single_page(result.url, result.title)
                if content:
                    scraped_data.append(content)
                    
                # Be polite to servers
                if i < len(search_results) - 1:
                    time.sleep(self.config.request_delay)
                    
            except Exception as e:
                logger.warning(f"Failed to scrape {result.url}: {e}")
                continue
                
        return scraped_data
    
    def _scrape_single_page(self, url: str, title: str) -> Optional[ScrapedContent]:
        """Scrape a single web page"""
        cache_key = self.cache._get_cache_key(f"content_{url}")
        cached_content = self.cache.get(cache_key)
        
        if cached_content:
            return ScrapedContent(**cached_content)
        
        try:
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'ads', 'iframe']):
                element.decompose()
            
            # Extract main content
            content = self._extract_main_content(soup)
            if not content:
                return None
            
            # Clean and truncate content
            cleaned_content = self._clean_content(content)
            truncated_content = cleaned_content[:self.config.max_content_length]
            
            scraped_content = ScrapedContent(
                url=url,
                title=title,
                content=truncated_content,
                content_length=len(truncated_content),
                scrape_timestamp=datetime.now()
            )
            
            # Cache the content
            self.cache.set(cache_key, scraped_content.dict())
            
            return scraped_content
            
        except Exception as e:
            logger.warning(f"Scraping failed for {url}: {e}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main content from page"""
        content_selectors = [
            'main', 'article', 
            'div.content', 'div#content',
            'div.main-content', 'div.article-content',
            'div.post-content', 'div.entry-content'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(separator=' ', strip=True)
        
        # Fallback to body
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)
            
        return None
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content"""
        # Remove extra whitespace
        content = ' '.join(content.split())
        # Remove excessive line breaks
        content = re.sub(r'\n\s*\n', '\n\n', content)
        return content

# =============================================================================
# AI INTEGRATION
# =============================================================================

class ReviewGenerator:
    """Handles AI review generation"""
    
    def __init__(self, groq_client: Groq, config: AppConfig):
        self.client = groq_client
        self.config = config
    
    def generate_web_review(self, product_name: str, search_results: List[SearchResult], 
                          scraped_content: List[ScrapedContent]) -> ProductReview:
        """Generate review from web data"""
        context = self._build_web_context(product_name, search_results, scraped_content)
        
        system_prompt = self._get_web_review_system_prompt()
        user_prompt = self._get_web_review_user_prompt(product_name, context, scraped_content)
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.config.model_name,
                response_format={"type": "json_object"},
                temperature=self.config.temperature_review,
                max_tokens=self.config.max_tokens_review
            )
            
            review_data = json.loads(response.choices[0].message.content)
            validated_review = self._validate_review_data(review_data, scraped_content)
            
            return validated_review
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"AI review generation failed: {e}")
            raise AIGenerationError(f"Failed to generate review: {str(e)}")
    
    def generate_ai_knowledge_review(self, product_name: str) -> ProductReview:
        """Generate review from AI knowledge"""
        return ProductReview.from_ai_knowledge(product_name)
    
    def _build_web_context(self, product_name: str, search_results: List[SearchResult],
                          scraped_content: List[ScrapedContent]) -> str:
        """Build context from web data"""
        context_parts = [f"# Product Review Request: {product_name}\n"]
        
        # Add search results
        context_parts.append("## Search Results:\n")
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"{i}. **{result.title}**")
            context_parts.append(f"   Summary: {result.snippet}")
            context_parts.append(f"   URL: {result.url}\n")
        
        # Add detailed content
        if scraped_content:
            context_parts.append("\n## Detailed Content:\n")
            for i, content in enumerate(scraped_content, 1):
                context_parts.append(f"### Source {i}: {content.title}")
                context_parts.append(f"Content: {content.content[:2000]}...\n")
        
        return "\n".join(context_parts)
    
    def _get_web_review_system_prompt(self) -> str:
        return """You are an expert product reviewer. Create a comprehensive review STRICTLY from provided sources.

Critical Rules:
1. Use ONLY information from provided sources
2. Be specific - reference actual features/specs found
3. Include pricing if mentioned
4. Be balanced - mention both strengths and weaknesses
5. Note conflicting information if present
6. NEVER fabricate information
7. Rate fairly based on available information

Output must be valid JSON matching the exact schema."""
    
    def _get_web_review_user_prompt(self, product_name: str, context: str, 
                                   scraped_content: List[ScrapedContent]) -> str:
        sources = [content.url for content in scraped_content]
        
        return f"""Based on this current web information (gathered on {datetime.now().strftime('%B %d, %Y')}), create a product review:

{context}

Generate JSON with this exact structure:
{{
"product_name": "Full product name from sources",
"specifications_inferred": "Concise summary of key specs found",
"predicted_rating": "X.X / 5.0 (based on analysis)",
"pros": ["Specific advantage 1", "Specific advantage 2", "..."],
"cons": ["Specific disadvantage 1", "Specific disadvantage 2", "..."],
"verdict": "Comprehensive concluding paragraph",
"price_info": "Current pricing if found, else 'Price varies by retailer'",
"sources": {json.dumps(sources)},
"last_updated": "{datetime.now().strftime('%Y-%m-%d')}",
"data_source_type": "free_web_search"
}}

Be critical and honest. Include issues mentioned in sources."""
    
    def _validate_review_data(self, review_data: Dict, scraped_content: List[ScrapedContent]) -> ProductReview:
        """Validate and clean review data"""
        try:
            # Ensure sources are properly set
            if not review_data.get('sources') and scraped_content:
                review_data['sources'] = [content.url for content in scraped_content]
            
            # Ensure data source type is set
            review_data['data_source_type'] = 'free_web_search'
            review_data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            
            return ProductReview(**review_data)
            
        except ValidationError as e:
            logger.error(f"Review validation failed: {e}")
            raise ValidationError(f"Invalid review data: {e}")

class ChatService:
    """Handles product chat conversations"""
    
    def __init__(self, groq_client: Groq, config: AppConfig):
        self.client = groq_client
        self.config = config
    
    def get_chat_response(self, user_message: str, conversation_history: List[Dict], 
                         product_review: ProductReview) -> str:
        """Get chat response about the product"""
        system_prompt = self._get_chat_system_prompt(product_review)
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.config.model_name,
                temperature=self.config.temperature_chat,
                max_tokens=self.config.max_tokens_chat
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Chat response failed: {e}")
            raise AIGenerationError(f"Chat failed: {str(e)}")
    
    def _get_chat_system_prompt(self, product_review: ProductReview) -> str:
        base_prompt = """You are an Expert Product Reviewer and Technical Consultant.

Your Role:
- Answer questions about the product with expert knowledge
- Provide comparisons with similar products
- Explain technical specifications in detail
- Give purchasing advice and recommendations
- Discuss use cases and real-world performance
- Be conversational but maintain expertise

Guidelines:
1. Draw from your knowledge about the product and market
2. Be honest about limitations and trade-offs
3. Provide specific examples and scenarios
4. Give balanced pros/cons for comparisons
5. Keep responses concise but informative (2-4 paragraphs)
6. Reference the initial review context when relevant

Conversation Style: Professional but friendly, use analogies for complex features."""
        
        # Add product context
        if product_review.data_source_type == 'free_web_search':
            base_prompt += f"""

Current Product Context:
- Product: {product_review.product_name}
- Key Specs: {product_review.specifications_inferred}
- Rating: {product_review.predicted_rating}
- Price: {product_review.price_info}
- Data Source: Real-time web search (current information)
"""
        else:
            base_prompt += f"""

Current Product Context:
- Product: {product_review.product_name}
- Data Source: AI Knowledge (may not reflect current information)
- Note: Recommend verifying current specs and pricing
"""
        
        return base_prompt

# =============================================================================
# MAIN SERVICE
# =============================================================================

class ProductReviewService:
    """Orchestrates product review generation"""
    
    def __init__(self, groq_api_key: str, config: AppConfig = None):
        self.config = config or AppConfig()
        self.groq_client = Groq(api_key=groq_api_key)
        self.cache_manager = CacheManager(ttl_hours=self.config.cache_ttl_hours)
        
        # Initialize components
        self.search_client = WebSearchClient(self.cache_manager, self.config)
        self.scraper = ContentScraper(self.cache_manager, self.config)
        self.review_generator = ReviewGenerator(self.groq_client, self.config)
        self.chat_service = ChatService(self.groq_client, self.config)
    
    def generate_review(self, product_name: str, use_web_search: bool = True) -> ProductReview:
        """Generate product review"""
        try:
            if use_web_search:
                return self._generate_web_review(product_name)
            else:
                return self._generate_ai_knowledge_review(product_name)
                
        except ProductReviewError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating review: {e}")
            raise ProductReviewError(f"Failed to generate review: {str(e)}")
    
    def _generate_web_review(self, product_name: str) -> ProductReview:
        """Generate review using web search"""
        # Step 1: Search
        search_results = self.search_client.search_products(product_name)
        if not search_results:
            raise SearchError("No search results found")
        
        # Step 2: Scrape content
        scraped_content = self.scraper.scrape_content(search_results)
        
        # Step 3: Generate review
        return self.review_generator.generate_web_review(
            product_name, search_results, scraped_content
        )
    
    def _generate_ai_knowledge_review(self, product_name: str) -> ProductReview:
        """Generate review using AI knowledge"""
        return self.review_generator.generate_ai_knowledge_review(product_name)

# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================

class StreamlitUI:
    """Handles Streamlit user interface"""
    
    def __init__(self, review_service: ProductReviewService):
        self.service = review_service
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_product" not in st.session_state:
            st.session_state.current_product = None
        if "review_data" not in st.session_state:
            st.session_state.review_data = None
        if "chat_mode" not in st.session_state:
            st.session_state.chat_mode = False
    
    def render_sidebar(self):
        """Render sidebar content"""
        with st.sidebar:
            st.title("🤖 Product Review Chat")
            st.markdown("---")
            
            if st.session_state.current_product:
                self._render_current_product_sidebar()
            else:
                st.info("👈 Enter a product name to start")
            
            self._render_help_section()
            self._render_footer()
    
    def _render_current_product_sidebar(self):
        """Render current product info in sidebar"""
        st.success(f"**Current Product:**\n{st.session_state.current_product}")
        st.markdown("---")
        
        if st.session_state.review_data:
            review = st.session_state.review_data
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rating", review.predicted_rating)
            with col2:
                source_type = review.data_source_type
                if source_type == 'free_web_search':
                    st.metric("Sources", len(review.sources))
                else:
                    st.metric("Source", "AI KB")
            
            st.metric("Pros", len(review.pros))
            st.metric("Cons", len(review.cons))
        
        st.markdown("---")
        
        if st.button("🔄 Review Different Product", use_container_width=True):
            self._reset_conversation()
            st.rerun()
    
    def _render_help_section(self):
        """Render help and tips section"""
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
    
    def _render_footer(self):
        """Render sidebar footer"""
        st.markdown("---")
        st.caption("🆓 100% Free • No API costs")
        st.caption("🌐 Web Search: DuckDuckGo")
        st.caption("🤖 AI: Groq Llama 3.3 70B")
    
    def render_search_interface(self):
        """Render initial search interface"""
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
            help="Web Search scrapes current product info. AI Knowledge uses training data from January 2025."
        )
        
        use_web = data_source.startswith("🌐")
        
        with col2:
            search_button = st.button("🔍 Analyze", use_container_width=True, type="primary")
        
        # Example products
        self._render_example_products()
        
        # Info box
        st.info("""
        **🌐 Web Search Mode**: Searches DuckDuckGo and analyzes current product information from multiple sources. 
        Takes 10-20 seconds but provides accurate, up-to-date data.
        
        **🤖 AI Knowledge Mode**: Fast responses using AI training data. 
        Instant results but may not reflect latest specifications or pricing.
        """)
        
        return product_input, search_button, use_web
    
    def _render_example_products(self):
        """Render example product buttons"""
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
                if st.button(example, use_container_width=True, key=f"example_{idx}"):
                    st.session_state.example_product = example
                    st.rerun()
    
    def render_review_display(self, review: ProductReview):
        """Display the structured review"""
        st.markdown("---")
        
        # Header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.header(f"📱 {review.product_name}")
        with col2:
            st.markdown(f"### ⭐ {review.predicted_rating}")
        with col3:
            if review.data_source_type == 'free_web_search':
                st.success("🌐 Live Web Data")
            else:
                st.info("🤖 AI Knowledge")
        
        # Data source info
        if review.data_source_type == 'free_web_search':
            st.success(f"✅ Information verified from {len(review.sources)} web sources on {review.last_updated}")
        else:
            st.warning("⚠️ Based on AI training data (updated January 2025). Please verify current specifications and pricing.")
        
        # Price and specs
        col_price, col_specs = st.columns([1, 2])
        
        with col_price:
            st.markdown("### 💰 Pricing")
            st.info(review.price_info)
        
        with col_specs:
            st.markdown("### 🔧 Key Specifications")
            st.info(review.specifications_inferred)
        
        st.markdown("---")
        
        # Pros and Cons
        col_pros, col_cons = st.columns(2)
        
        with col_pros:
            st.markdown("### 🟢 Strengths")
            for i, pro in enumerate(review.pros[:10], 1):
                st.markdown(f"**{i}.** {pro}")
        
        with col_cons:
            st.markdown("### 🔴 Weaknesses")
            for i, con in enumerate(review.cons[:10], 1):
                st.markdown(f"**{i}.** {con}")
        
        st.markdown("---")
        
        # Verdict
        st.markdown("### ✅ Final Verdict")
        st.write(review.verdict)
        
        # Sources
        if review.sources and review.data_source_type == 'free_web_search':
            with st.expander("📚 Sources Used"):
                for i, source in enumerate(review.sources, 1):
                    st.markdown(f"{i}. [{source}]({source})")
    
    def render_chat_interface(self):
        """Render chat interface"""
        st.title(f"💬 Chat about: {st.session_state.current_product}")
        
        # Display review
        with st.expander("📊 View Full Review", expanded=False):
            if st.session_state.review_data:
                self.render_review_display(st.session_state.review_data)
        
        st.markdown("---")
        
        # Chat messages
        self._render_chat_messages()
        
        # Quick questions for new conversations
        if len(st.session_state.messages) <= 1:
            self._render_quick_questions()
        
        # Chat input
        self._handle_chat_input()
    
    def _render_chat_messages(self):
        """Render chat message history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(message.get("timestamp", ""))
    
    def _render_quick_questions(self):
        """Render quick question suggestions"""
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
                    self._process_user_message(question)
    
    def _handle_chat_input(self):
        """Handle chat input from user"""
        user_input = st.chat_input("Ask anything about this product...")
        
        if user_input:
            self._process_user_message(user_input)
    
    def _process_user_message(self, user_input: str):
        """Process user message and get AI response"""
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            try:
                conversation_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                ]
                
                response = self.service.chat_service.get_chat_response(
                    user_input, 
                    conversation_history,
                    st.session_state.review_data
                )
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                
                st.rerun()
                
            except AIGenerationError as e:
                st.error(f"Chat error: {e}")
                st.session_state.messages.pop()  # Remove failed user message
    
    def _reset_conversation(self):
        """Reset conversation state"""
        st.session_state.messages = []
        st.session_state.current_product = None
        st.session_state.review_data = None
        st.session_state.chat_mode = False

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
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
    
    # Initialize services
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        review_service = ProductReviewService(groq_api_key)
        ui = StreamlitUI(review_service)
        
    except KeyError:
        st.error("❌ Groq API key not found in secrets.toml. Please add it to continue.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {e}")
        st.stop()
    
    # Render sidebar
    ui.render_sidebar()
    
    # Main content area
    if not st.session_state.chat_mode:
        # Search interface
        product_input, search_button, use_web = ui.render_search_interface()
        
        # Handle example product
        if hasattr(st.session_state, 'example_product'):
            product_input = st.session_state.example_product
            search_button = True
            del st.session_state.example_product
        
        # Handle search
        if search_button and product_input:
            with st.spinner(f"{'🔍 Searching the web and analyzing' if use_web else '🤖 Analyzing'} '{product_input}'..."):
                try:
                    review_data = review_service.generate_review(product_input, use_web_search=use_web)
                    
                    st.session_state.current_product = product_input
                    st.session_state.review_data = review_data
                    st.session_state.chat_mode = True
                    
                    # Add initial review to conversation
                    review_summary = f"""I've analyzed the {review_data.product_name}. Here's my review:

Rating: {review_data.predicted_rating}

Price: {review_data.price_info}

Key Specs: {review_data.specifications_inferred}

Strengths: {', '.join(review_data.pros[:3])}

Weaknesses: {', '.join(review_data.cons[:3])}

Verdict: {review_data.verdict}

Data Source: {review_data.data_source_type.replace('_', ' ').title()}

Feel free to ask me any questions about this product!"""

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": review_summary,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    
                    st.rerun()
                    
                except ProductReviewError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
    
    else:
        # Chat interface
        ui.render_chat_interface()

if __name__ == "__main__":
    main()