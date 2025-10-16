import streamlit as st
import json
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import hashlib
from pathlib import Path
import logging
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# UNIVERSAL PRODUCT CONFIGURATION - FIXED DATACLASSES
# =============================================================================

class ProductCategories:
    """Define product categories and their specific search strategies"""
    ELECTRONICS = "electronics"
    HOME_APPLIANCES = "home_appliances"
    KITCHEN = "kitchen"
    TOOLS = "tools"
    FURNITURE = "furniture"
    SPORTS = "sports"
    BEAUTY = "beauty"
    AUTOMOTIVE = "automotive"
    BOOKS_MEDIA = "books_media"
    CLOTHING = "clothing"
    TOYS_GAMES = "toys_games"
    OFFICE = "office"
    HEALTH = "health"
    GARDEN = "garden"
    PET_SUPPLIES = "pet_supplies"

@dataclass
class SafetyConfig:
    """Safety configuration to prevent API overuse"""
    max_daily_llm_requests: int = 50
    rate_limit_check_interval: int = 10
    fallback_to_non_llm: bool = True
    cache_non_llm_responses: bool = True

@dataclass
class AppConfig:
    """Universal configuration for all product types"""
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
    
    # Safety Settings - FIXED: Use default_factory for mutable objects
    safety: SafetyConfig = field(default_factory=SafetyConfig)

class Constants:
    """Application constants"""
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ACCEPT_HEADER = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    ACCEPT_LANGUAGE = "en-US,en;q=0.5"

# =============================================================================
# UNIVERSAL PRODUCT REVIEW DATA MODEL - FIXED
# =============================================================================

class ProductReview(BaseModel):
    """A comprehensive product review for ANY product type"""
    product_name: str = Field(description="The full name of the product being reviewed.")
    category: str = Field(description="Detected product category")
    key_features: str = Field(description="A concise summary of key features and specifications.")
    predicted_rating: str = Field(description="A critical rating out of 5.0 (e.g., '4.6 / 5.0').")
    pros: List[str] = Field(default_factory=list, description="A list of strengths and advantages.")
    cons: List[str] = Field(default_factory=list, description="A list of weaknesses, trade-offs, or user pain points.")
    verdict: str = Field(description="A concluding summary of the product's overall value proposition.")
    price_info: str = Field(default="Price not available", description="Current pricing information if found.")
    best_uses: List[str] = Field(default_factory=list, description="Recommended use cases for this product.")
    target_audience: str = Field(default="", description="Who this product is best suited for.")
    sources: List[str] = Field(default_factory=list, description="List of source URLs used.")
    last_updated: str = Field(default="", description="Date when information was gathered.")
    data_source_type: str = Field(default="web_search", description="Type of data source used.")
    
    @classmethod
    def from_ai_knowledge(cls, product_name: str, category: str) -> 'ProductReview':
        """Create a placeholder for AI knowledge-based reviews"""
        return cls(
            product_name=product_name,
            category=category,
            key_features="Based on AI training data (updated January 2025)",
            predicted_rating="N/A (AI Knowledge)",
            pros=["Information from AI training data"],
            cons=["May not reflect current specifications or pricing"],
            verdict="This review is based on AI training data. Please verify current information with official sources.",
            price_info="Price varies - check current retailers",
            best_uses=["General purpose use"],
            target_audience="General consumers",
            sources=["AI Training Data (Updated January 2025)"],
            last_updated=datetime.now().strftime('%Y-%m-%d'),
            data_source_type="ai_knowledge"
        )

# =============================================================================
# DATA MODELS FOR SEARCH AND CONTENT - FIXED
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

# =============================================================================
# CACHE MANAGEMENT - FIXED
# =============================================================================

class CacheManager:
    """Managed cache with TTL and size limits"""
    
    def __init__(self, cache_dir: Path = None, ttl_hours: int = 24, max_size: int = 100):
        self.cache_dir = cache_dir or Path(".cache")
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
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            for old_file in cache_files[:len(cache_files) - self.max_size]:
                try:
                    old_file.unlink()
                except Exception:
                    pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                cache_file.unlink()
                return None
                
            return cached_data['data']
        except Exception:
            try:
                cache_file.unlink()
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
                json.dump(cache_data, f)
                
        except Exception:
            pass

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
            return [SearchResult(**result) for result in cached_results]
        
        try:
            search_query = f"{product_name} specifications review price features"
            results = self._duckduckgo_search(search_query)
            
            if results:
                self.cache.set(cache_key, [result.dict() for result in results])
            
            return results
            
        except Exception as e:
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
                    
                if i < len(search_results) - 1:
                    time.sleep(self.config.request_delay)
                    
            except Exception:
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
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'ads', 'iframe']):
                element.decompose()
            
            content = self._extract_main_content(soup)
            if not content:
                return None
            
            cleaned_content = self._clean_content(content)
            truncated_content = cleaned_content[:self.config.max_content_length]
            
            scraped_content = ScrapedContent(
                url=url,
                title=title,
                content=truncated_content,
                content_length=len(truncated_content),
                scrape_timestamp=datetime.now()
            )
            
            self.cache.set(cache_key, scraped_content.dict())
            
            return scraped_content
            
        except Exception:
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
        
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)
            
        return None
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content"""
        content = ' '.join(content.split())
        content = re.sub(r'\n\s*\n', '\n\n', content)
        return content

# =============================================================================
# PRODUCT CATEGORY DETECTION
# =============================================================================

class ProductCategoryDetector:
    """Detects product categories to optimize search strategies"""
    
    def __init__(self):
        self.category_keywords = self._initialize_category_keywords()
    
    def _initialize_category_keywords(self) -> Dict[str, List[str]]:
        """Initialize category keywords"""
        return {
            ProductCategories.ELECTRONICS: [
                'phone', 'smartphone', 'tablet', 'laptop', 'computer', 'tv', 'television',
                'camera', 'headphone', 'earbud', 'speaker', 'smartwatch', 'console'
            ],
            ProductCategories.HOME_APPLIANCES: [
                'refrigerator', 'washer', 'dryer', 'dishwasher', 'oven', 'microwave',
                'vacuum', 'air conditioner', 'heater', 'fan', 'dehumidifier'
            ],
            ProductCategories.KITCHEN: [
                'blender', 'mixer', 'coffee maker', 'toaster', 'air fryer', 'instant pot',
                'knife', 'cookware', 'bakeware', 'food processor'
            ],
            ProductCategories.TOOLS: [
                'drill', 'saw', 'wrench', 'screwdriver', 'toolkit', 'hammer',
                'power tool', 'hand tool', 'measurement'
            ],
            ProductCategories.FURNITURE: [
                'sofa', 'couch', 'chair', 'table', 'desk', 'bed', 'mattress',
                'cabinet', 'shelf', 'wardrobe'
            ]
        }
    
    def detect_category(self, product_name: str) -> str:
        """Detect the most likely product category"""
        product_lower = product_name.lower()
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in product_lower)
            if score > 0:
                category_scores[category] = score
        
        return max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else ProductCategories.ELECTRONICS

# =============================================================================
# API KEY MANAGEMENT & SAFETY
# =============================================================================

class APIKeyManager:
    """Manages API key validation and safety"""
    
    def __init__(self):
        self.llm_available = False
        self.llm_client = None
        self.usage_tracker = {
            'daily_requests': 0,
            'last_reset_date': datetime.now().date(),
            'total_requests': 0,
            'errors_count': 0
        }
    
    def initialize_llm_client(self, groq_api_key: str = "") -> bool:
        """Initialize LLM client with safety checks"""
        try:
            if not groq_api_key:
                groq_api_key = st.secrets.get("GROQ_API_KEY", "")
            
            if not groq_api_key:
                st.warning("🔒 No Groq API key found. Using fast non-LLM mode only.")
                self.llm_available = False
                return False
            
            test_client = Groq(api_key=groq_api_key)
            test_client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model="llama-3.3-70b-versatile",
                max_tokens=5
            )
            
            self.llm_client = test_client
            self.llm_available = True
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if "invalid api key" in error_msg or "authentication" in error_msg:
                st.error("❌ Invalid Groq API key. Using fast non-LLM mode.")
            elif "rate limit" in error_msg:
                st.warning("⚠️ Rate limit reached. Using fast non-LLM mode.")
            else:
                st.warning(f"🔧 API issue. Using fast non-LLM mode.")
            
            self.llm_available = False
            return False
    
    def can_use_llm(self) -> bool:
        """Check if we can use LLM based on safety rules"""
        if not self.llm_available:
            return False
        
        current_date = datetime.now().date()
        if current_date != self.usage_tracker['last_reset_date']:
            self.usage_tracker['daily_requests'] = 0
            self.usage_tracker['last_reset_date'] = current_date
        
        if self.usage_tracker['daily_requests'] >= 50:
            st.warning("📊 Daily LLM limit reached. Using fast non-LLM mode.")
            return False
        
        if (self.usage_tracker['errors_count'] > 5 and 
            self.usage_tracker['total_requests'] > 0 and
            (self.usage_tracker['errors_count'] / self.usage_tracker['total_requests']) > 0.2):
            st.warning("🔄 High error rate. Using fast non-LLM mode temporarily.")
            return False
        
        return True
    
    def track_llm_usage(self, success: bool = True):
        """Track LLM usage for safety"""
        self.usage_tracker['total_requests'] += 1
        self.usage_tracker['daily_requests'] += 1
        
        if not success:
            self.usage_tracker['errors_count'] += 1

# =============================================================================
# UNIVERSAL REVIEW GENERATOR
# =============================================================================

class UniversalReviewGenerator:
    """Generates reviews for any product type"""
    
    def __init__(self, groq_client: Groq, config: AppConfig):
        self.client = groq_client
        self.config = config
    
    def generate_web_review(self, product_name: str, category: str, 
                          search_results: List[SearchResult], 
                          scraped_content: List[ScrapedContent]) -> ProductReview:
        """Generate review from web data for any product"""
        context = self._build_universal_context(product_name, category, search_results, scraped_content)
        
        system_prompt = self._get_universal_system_prompt(category)
        user_prompt = self._get_universal_user_prompt(product_name, category, context, scraped_content)
        
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
            return ProductReview(**review_data)
            
        except Exception as e:
            raise AIGenerationError(f"Failed to generate review: {str(e)}")
    
    def generate_ai_knowledge_review(self, product_name: str, category: str) -> ProductReview:
        """Generate review from AI knowledge"""
        return ProductReview.from_ai_knowledge(product_name, category)
    
    def _build_universal_context(self, product_name: str, category: str, 
                               search_results: List[SearchResult],
                               scraped_content: List[ScrapedContent]) -> str:
        """Build context for any product type"""
        context_parts = [f"# Product Review Request: {product_name}"]
        context_parts.append(f"# Product Category: {category.replace('_', ' ').title()}\n")
        
        context_parts.append("## Search Results:\n")
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"{i}. **{result.title}**")
            context_parts.append(f"   Summary: {result.snippet}")
            context_parts.append(f"   URL: {result.url}\n")
        
        if scraped_content:
            context_parts.append("\n## Detailed Content:\n")
            for i, content in enumerate(scraped_content, 1):
                context_parts.append(f"### Source {i}: {content.title}")
                context_parts.append(f"Content: {content.content[:2000]}...\n")
        
        return "\n".join(context_parts)
    
    def _get_universal_system_prompt(self, category: str) -> str:
        return f"""You are an expert product reviewer for {category.replace('_', ' ')} products. 
Create a comprehensive review STRICTLY from provided sources.

Critical Rules:
1. Use ONLY information from provided sources
2. Focus on aspects relevant to {category.replace('_', ' ')} products
3. Include pricing if mentioned
4. Be balanced - mention both strengths and weaknesses
5. Note conflicting information if present
6. NEVER fabricate information
7. Rate fairly based on available information

Output must be valid JSON matching the exact schema."""
    
    def _get_universal_user_prompt(self, product_name: str, category: str, context: str, 
                                 scraped_content: List[ScrapedContent]) -> str:
        sources = [content.url for content in scraped_content]
        
        return f"""Based on this current web information for {category.replace('_', ' ')} product (gathered on {datetime.now().strftime('%B %d, %Y')}), create a product review:

{context}

Generate JSON with this exact structure:
{{
"product_name": "Full product name from sources",
"category": "{category}",
"key_features": "Concise summary of key features and specs",
"predicted_rating": "X.X / 5.0 (based on analysis)",
"pros": ["Specific advantage 1", "Specific advantage 2", "..."],
"cons": ["Specific disadvantage 1", "Specific disadvantage 2", "..."],
"verdict": "Comprehensive concluding paragraph",
"price_info": "Current pricing if found, else 'Price varies by retailer'",
"best_uses": ["Primary use case 1", "Use case 2", "..."],
"target_audience": "Description of who this product is best for",
"sources": {json.dumps(sources)},
"last_updated": "{datetime.now().strftime('%Y-%m-%d')}",
"data_source_type": "free_web_search"
}}

Focus on aspects relevant to {category.replace('_', ' ')} products. Be critical and honest."""

# =============================================================================
# SIMPLIFIED STREAMLIT UI (Working Version)
# =============================================================================

def main():
    """Main application with simplified UI"""
    
    st.set_page_config(
        page_title="Universal Product Review",
        page_icon="🌐", 
        layout="wide"
    )
    
    st.title("🌐 Universal Product Review Assistant")
    st.markdown("### Analyze any product with zero API costs")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_product" not in st.session_state:
        st.session_state.current_product = None
    if "review_data" not in st.session_state:
        st.session_state.review_data = None
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = False
    
    # Sidebar
    with st.sidebar:
        st.title("Product Review")
        
        if st.session_state.current_product:
            st.success(f"Current: {st.session_state.current_product}")
            if st.button("New Product"):
                st.session_state.messages = []
                st.session_state.current_product = None
                st.session_state.review_data = None
                st.session_state.chat_mode = False
                st.rerun()
        else:
            st.info("Enter a product name to start")
    
    # Main interface
    if not st.session_state.chat_mode:
        # Product input
        product_input = st.text_input(
            "Enter any product name:",
            placeholder="e.g., Instant Pot, DeWalt Drill, Sony Headphones...",
            key="product_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            data_source = st.radio(
                "Data source:",
                ["🌐 Web Search", "🤖 AI Knowledge"],
                horizontal=True
            )
        with col2:
            analyze_btn = st.button("🔍 Analyze", type="primary")
        
        # Example products
        st.markdown("**Try these examples:**")
        examples = ["Instant Pot", "DeWalt Drill", "Sony WH-1000XM5", "Dyson Vacuum"]
        cols = st.columns(4)
        for idx, example in enumerate(examples):
            with cols[idx]:
                if st.button(example, key=f"ex_{idx}"):
                    st.session_state.example_product = example
                    st.rerun()
        
        # Handle analysis
        if analyze_btn and product_input or hasattr(st.session_state, 'example_product'):
            product_to_analyze = st.session_state.example_product if hasattr(st.session_state, 'example_product') else product_input
            if hasattr(st.session_state, 'example_product'):
                del st.session_state.example_product
            
            with st.spinner(f"Analyzing {product_to_analyze}..."):
                try:
                    # Initialize services
                    config = AppConfig()
                    cache_manager = CacheManager()
                    search_client = WebSearchClient(cache_manager, config)
                    scraper = ContentScraper(cache_manager, config)
                    
                    # Try to initialize LLM
                    api_manager = APIKeyManager()
                    api_manager.initialize_llm_client()
                    
                    if api_manager.llm_available and data_source == "🌐 Web Search":
                        # Web search with LLM
                        search_results = search_client.search_products(product_to_analyze)
                        if search_results:
                            scraped_content = scraper.scrape_content(search_results)
                            category_detector = ProductCategoryDetector()
                            category = category_detector.detect_category(product_to_analyze)
                            
                            review_generator = UniversalReviewGenerator(api_manager.llm_client, config)
                            review_data = review_generator.generate_web_review(
                                product_to_analyze, category, search_results, scraped_content
                            )
                        else:
                            st.error("No search results found")
                            return
                    else:
                        # AI knowledge or fallback
                        category_detector = ProductCategoryDetector()
                        category = category_detector.detect_category(product_to_analyze)
                        if api_manager.llm_available:
                            review_generator = UniversalReviewGenerator(api_manager.llm_client, config)
                            review_data = review_generator.generate_ai_knowledge_review(product_to_analyze, category)
                        else:
                            # Simple fallback review
                            review_data = ProductReview(
                                product_name=product_to_analyze,
                                category=category,
                                key_features="Check manufacturer website for details",
                                predicted_rating="N/A (Basic Info)",
                                pros=["Product information available online", "Multiple retailer options"],
                                cons=["Limited details available", "Verify with official sources"],
                                verdict=f"Basic information for {product_to_analyze}. Check official sources for current specifications.",
                                price_info="Check retailers for current pricing",
                                best_uses=[f"General {category.replace('_', ' ')} use"],
                                target_audience="General consumers",
                                sources=[],
                                last_updated=datetime.now().strftime('%Y-%m-%d'),
                                data_source_type="basic_info"
                            )
                    
                    st.session_state.current_product = product_to_analyze
                    st.session_state.review_data = review_data
                    st.session_state.chat_mode = True
                    
                    # Add welcome message
                    welcome_msg = f"""I've analyzed **{review_data.product_name}** ({review_data.category.replace('_', ' ').title()}):

**Rating**: {review_data.predicted_rating}
**Price**: {review_data.price_info}

**Key Features**: {review_data.key_features}

**Best For**: {review_data.target_audience or 'General use'}

Ask me anything about this product!"""
                    
                    st.session_state.messages = [{
                        "role": "assistant", 
                        "content": welcome_msg,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    }]
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    else:
        # Chat interface
        st.header(f"💬 Chat about {st.session_state.current_product}")
        
        # Display review summary
        if st.session_state.review_data:
            with st.expander("📊 Review Summary", expanded=False):
                review = st.session_state.review_data
                st.subheader(review.product_name)
                st.caption(f"Category: {review.category.replace('_', ' ').title()}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rating", review.predicted_rating)
                with col2:
                    st.metric("Price", review.price_info)
                with col3:
                    st.metric("Data Source", review.data_source_type.replace('_', ' ').title())
                
                st.markdown("**Key Features:**")
                st.info(review.key_features)
                
                col_pros, col_cons = st.columns(2)
                with col_pros:
                    st.markdown("**Strengths**")
                    for pro in review.pros[:3]:
                        st.write(f"• {pro}")
                with col_cons:
                    st.markdown("**Considerations**")
                    for con in review.cons[:3]:
                        st.write(f"• {con}")
        
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(message["timestamp"])
        
        # Chat input
        if prompt := st.chat_input("Ask about this product..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            
            # Simple response (non-LLM for now)
            response = f"I understand you're asking about '{prompt}'. For detailed answers about {st.session_state.current_product}, please check the product's official website or current retailer listings for the most up-to-date information."
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            
            st.rerun()

if __name__ == "__main__":
    main()