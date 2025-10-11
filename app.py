
import os
import re
import json
import time
import hashlib
import pickle
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging
import traceback
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rapidfuzz import fuzz
from urllib import robotparser
from difflib import SequenceMatcher


PLAYWRIGHT_AVAILABLE = False
SELENIUM_AVAILABLE = False
WEBDRIVER_MANAGER_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except Exception:
    WEBDRIVER_MANAGER_AVAILABLE = False

# ========== LOGGING CONFIG ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== CACHE DIRECTORY ==========
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# ========== CONFIG ==========
class Config:
    DEFAULT_TIMEOUT = 15
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2.0

    MAX_REVIEW_LENGTH = 2000
    MIN_REVIEW_LENGTH = 30
    MAX_SPEC_VALUE_LENGTH = 500

    MIN_REVIEW_QUALITY_SCORE = 0.6
    DUPLICATE_SIMILARITY_THRESHOLD = 0.85  # used with rapidfuzz (0-1 scaled)

    CACHE_TTL_HOURS = {
        'specs': 24,
        'reviews': 6,
        'urls': 24,
    }
    CACHE_MAX_AGE_DAYS = 7

    DEFAULT_RATE_LIMIT_DELAY = 1.2
    RATE_LIMIT_BACKOFF = 2.0

    MAX_CONSECUTIVE_EMPTY_PAGES = 3
    DEFAULT_MAX_PAGES = 50

    DEFAULT_CHUNK_SIZE = 200
    MIN_REVIEWS_FOR_ANALYSIS = 5

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

# ========== GROQ CLIENT ==========
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("❌ Missing Groq API key. Please set GROQ_API_KEY in st.secrets or environment variables.")
    st.stop()
client = Groq(api_key=api_key)

# ========== REQUESTS SESSION WITH RETRIES ==========
def make_session(max_retries: int = Config.MAX_RETRIES, backoff: float = Config.BACKOFF_FACTOR):
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(['GET', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(Config.DEFAULT_HEADERS)
    return session

SESSION = make_session()

# ========== CACHING UTILITIES ==========
def get_cache_key(url: str, params: dict = None) -> str:
    cache_str = url or ""
    if params:
        try:
            cache_str += json.dumps(params, sort_keys=True, ensure_ascii=False)
        except Exception:
            cache_str += str(sorted(params.items()))
    return hashlib.md5(cache_str.encode()).hexdigest()

def load_from_cache(cache_key: str, max_age_hours: int = 24) -> Any:
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if not cache_file.exists():
        return None
    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
    age = datetime.now() - file_time
    if age > timedelta(hours=max_age_hours):
        try:
            cache_file.unlink()
        except Exception:
            pass
        return None
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Cache load failed: {e}")
        return None

def save_to_cache(cache_key: str, data: Any) -> None:
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")

def clean_old_cache(max_age_days: int = 7) -> None:
    cutoff = datetime.now() - timedelta(days=max_age_days)
    for cache_file in CACHE_DIR.glob("*.pkl"):
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_time < cutoff:
                cache_file.unlink()
                logger.info(f"Deleted old cache file: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to delete cache file {cache_file}: {e}")

# ========== URL UTILITIES ==========
def normalize_url(url: str) -> str:
    if not url:
        return url
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    tracking_params = {
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'fbclid', 'gclid', 'msclkid', '_ga', 'mc_cid', 'mc_eid',
        'ref', 'referrer', 'source'
    }
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    filtered_params = {k: v for k, v in query_params.items() if k.lower() not in tracking_params}
    new_query = urlencode(filtered_params, doseq=True)
    normalized = f"{parsed.scheme}://{netloc}{parsed.path}"
    if new_query:
        normalized += f"?{new_query}"
    return normalized

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    if not url:
        return False, "URL is empty"
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format"
    except Exception as e:
        return False, f"URL parsing error: {e}"
    suspicious_patterns = [r'javascript:', r'data:', r'file:', r'localhost', r'127\.0\.0\.1']
    for pattern in suspicious_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return False, f"Suspicious URL pattern detected"
    if len(parsed.netloc) > 253:
        return False, "Domain name too long"
    return True, None

def build_gsmarena_review_url(product_url: str) -> Optional[str]:
    if not product_url or not product_url.endswith(".php"):
        return None
    try:
        base, phone_id_part = product_url.rsplit("-", 1)
        phone_id = phone_id_part.replace(".php", "")
        return f"{base}-reviews-{phone_id}.php"
    except Exception:
        return None

def extract_product_id(url: str, site: str = None) -> Optional[str]:
    patterns = {
        'gsmarena': r'(\w+)-(\d+)\.php',
        'amazon': r'/dp/([A-Z0-9]{10})',
        'bestbuy': r'skuId=(\d+)',
        'notebookcheck': r'/review/([^/]+)\.html'
    }
    if not site:
        if 'gsmarena.com' in url:
            site = 'gsmarena'
        elif 'amazon.' in url:
            site = 'amazon'
        elif 'bestbuy.com' in url:
            site = 'bestbuy'
        elif 'notebookcheck' in url:
            site = 'notebookcheck'
    if site and site in patterns:
        match = re.search(patterns[site], url)
        if match:
            if site == 'gsmarena' and match.lastindex and match.lastindex >= 2:
                return match.group(2)
            return match.group(1)
    return None

# ========== JSON UTIL ==========
def safe_load_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
            try:
                return json.loads(text)
            except:
                pass
        match = re.search(r"\{(?:.|\n)*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

# ========== ERROR TRACKER ==========
class ErrorTracker:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info_messages = []
    def add_error(self, operation: str, error: Exception, details: str = ""):
        error_entry = {
            'timestamp': datetime.now(),
            'operation': operation,
            'error_type': type(error).__name__,
            'message': str(error),
            'details': details,
            'traceback': traceback.format_exc()
        }
        self.errors.append(error_entry)
        logger.error(f"{operation}: {error}")
    def add_warning(self, operation: str, message: str):
        warning_entry = {'timestamp': datetime.now(), 'operation': operation, 'message': message}
        self.warnings.append(warning_entry)
        logger.warning(f"{operation}: {message}")
    def add_info(self, operation: str, message: str):
        info_entry = {'timestamp': datetime.now(), 'operation': operation, 'message': message}
        self.info_messages.append(info_entry)
        logger.info(f"{operation}: {message}")
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    def get_summary(self) -> dict:
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'error_types': list(set(e['error_type'] for e in self.errors))
        }
    def display_report(self):
        summary = self.get_summary()
        if summary['total_errors'] == 0 and summary['total_warnings'] == 0:
            return
        st.markdown("### 📋 Error Report")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Errors", summary['total_errors'])
        with col2:
            st.metric("Warnings", summary['total_warnings'])
        if self.errors:
            with st.expander(f"❌ Errors ({len(self.errors)})", expanded=False):
                for i, error in enumerate(self.errors, 1):
                    st.error(f"**{i}. {error['operation']}** ({error['error_type']})")
                    st.text(f"Message: {error['message']}")
                    st.text(f"Time: {error['timestamp'].strftime('%H:%M:%S')}")
                    st.markdown("---")
        if self.warnings:
            with st.expander(f"⚠️ Warnings ({len(self.warnings)})"):
                for i, warning in enumerate(self.warnings, 1):
                    st.warning(f"**{i}. {warning['operation']}**")
                    st.text(warning['message'])
                    st.markdown("---")

error_tracker = ErrorTracker()
clean_old_cache(Config.CACHE_MAX_AGE_DAYS)

# ========== ROBOTS.TXT CHECK (with session timeout) ==========
def allowed_by_robots(url: str, user_agent: str = None) -> bool:
    """Check if URL is allowed by robots.txt with timeout protection."""
    if user_agent is None:
        user_agent = Config.DEFAULT_HEADERS.get("User-Agent", "*")
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            response = SESSION.get(robots_url, timeout=5)
            if response.status_code == 200:
                rp.parse(response.text.splitlines())
            else:
                # no robots file or inaccessible -> allow
                return True
        except Exception:
            error_tracker.add_warning("robots.txt", f"Could not fetch robots.txt for {parsed.netloc}, allowing by default")
            return True
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        error_tracker.add_warning("robots.txt", f"Robots check failed: {e}, allowing by default")
        return True

# ========== RENDERING FALLBACKS (Playwright, Selenium) ==========
def render_with_playwright(url: str, timeout: int = 30) -> Optional[str]:
    if not PLAYWRIGHT_AVAILABLE:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
            page = browser.new_page()
            page.set_default_navigation_timeout(timeout * 1000)
            page.goto(url, wait_until='networkidle')
            # allow small additional time for JS
            time.sleep(1.0)
            content = page.content()
            browser.close()
            return content
    except Exception as e:
        error_tracker.add_warning("Playwright", f"Playwright render failed: {e}")
        return None

def render_with_selenium(url: str, timeout: int = 30) -> Optional[str]:
    if not SELENIUM_AVAILABLE:
        return None
    try:
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver_path = None
        if WEBDRIVER_MANAGER_AVAILABLE:
            try:
                driver_path = ChromeDriverManager().install()
            except Exception:
                driver_path = None
        if driver_path:
            driver = webdriver.Chrome(executable_path=driver_path, options=options)
        else:
            driver = webdriver.Chrome(options=options)  # may fail if chromedriver not in PATH
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(1.0)
        content = driver.page_source
        driver.quit()
        return content
    except Exception as e:
        error_tracker.add_warning("Selenium", f"Selenium render failed: {e}")
        try:
            driver.quit()
        except Exception:
            pass
        return None

def fetch_rendered_html(url: str, prefer_playwright: bool = True, prefer_selenium: bool = True) -> Optional[str]:
    """Try to fetch fully rendered HTML using Playwright then Selenium, fallback to requests."""
    # Try cache quickly
    cache_key = get_cache_key(f"render_{url}", {"pw": prefer_playwright, "sel": prefer_selenium})
    cached = load_from_cache(cache_key, max_age_hours=1)
    if cached:
        return cached
    # Playwright
    if prefer_playwright and PLAYWRIGHT_AVAILABLE:
        html = render_with_playwright(url)
        if html:
            save_to_cache(cache_key, html)
            return html
    # Selenium
    if prefer_selenium and SELENIUM_AVAILABLE:
        html = render_with_selenium(url)
        if html:
            save_to_cache(cache_key, html)
            return html
    # Fallback to requests
    try:
        r = SESSION.get(url, timeout=Config.DEFAULT_TIMEOUT)
        r.raise_for_status()
        html = r.text
        save_to_cache(cache_key, html)
        return html
    except Exception as e:
        error_tracker.add_warning("Render Fallback", f"Requests fallback failed: {e}")
        return None

# ========== SITE-SPECIFIC PARSERS ==========
def parse_gsmarena_specs(soup: BeautifulSoup) -> Dict[str, str]:
    specs = {}
    for table in soup.select("table"):
        for row in table.select("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2:
                key = cells[0].get_text(" ", strip=True)
                val = cells[-1].get_text(" ", strip=True)
                if key and val:
                    specs.setdefault(key, val)
    for dl in soup.select("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            key = dt.get_text(" ", strip=True)
            val = dd.get_text(" ", strip=True)
            if key and val:
                specs.setdefault(key, val)
    return specs

def parse_amazon_specs_and_reviews(url: str, soup: BeautifulSoup) -> Tuple[Dict[str,str], List[str]]:
    specs = {}
    reviews = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") == "Product":
                if "name" in data:
                    specs.setdefault("Product Name", data["name"])
                if "brand" in data:
                    b = data["brand"]
                    specs.setdefault("Brand", b.get("name") if isinstance(b, dict) else str(b))
                if "offers" in data:
                    offers = data["offers"]
                    if isinstance(offers, dict) and "price" in offers:
                        specs.setdefault("Price", f"{offers.get('priceCurrency','') or ''} {offers.get('price','')}".strip())
        except Exception:
            continue
    title = soup.select_one("#productTitle")
    if title:
        specs.setdefault("Product Name", title.get_text(" ", strip=True))
    price = soup.select_one("#priceblock_ourprice, #priceblock_dealprice")
    if price:
        specs.setdefault("Price", price.get_text(" ", strip=True))
    for review_block in soup.select(".review-text, .review-text-content, .a-size-base.review-text"):
        txt = review_block.get_text(" ", strip=True)
        if txt:
            reviews.append(txt)
    for rb in soup.select("div[data-hook='review']"):
        txt = rb.select_one("span[data-hook='review-body']")
        if txt:
            reviews.append(txt.get_text(" ", strip=True))
    return specs, reviews

def parse_notebookcheck_specs_and_reviews(url: str, soup: BeautifulSoup) -> Tuple[Dict[str,str], List[str]]:
    specs = {}
    reviews = []
    title = soup.select_one("h1[itemprop='name'], h1")
    if title:
        specs.setdefault("Product Name", title.get_text(" ", strip=True))
    for pros in soup.select(".test-summary li, .pros-cons li, .pro-con li"):
        t = pros.get_text(" ", strip=True)
        if t:
            reviews.append(t)
    for p in soup.select("article p, .article p, .content p"):
        text = p.get_text(" ", strip=True)
        if text and len(text) > 80:
            reviews.append(text)
    return specs, reviews

# ========== GENERIC SPECS SCRAPER ==========
def extract_schema_org_data(soup) -> Dict[str, str]:
    specs = {}
    json_ld_scripts = soup.find_all("script", type="application/ld+json")
    for script in json_ld_scripts:
        try:
            data = json.loads(script.string)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") in ["Product", "MobileApplication"]:
                    if "name" in item:
                        specs["Product Name"] = str(item["name"])
                    if "brand" in item:
                        brand = item["brand"]
                        if isinstance(brand, dict):
                            specs["Brand"] = brand.get("name", str(brand))
                        else:
                            specs["Brand"] = str(brand)
                    if "model" in item:
                        specs["Model"] = str(item["model"])
                    if "description" in item:
                        desc = str(item["description"])
                        if len(desc) < Config.MAX_SPEC_VALUE_LENGTH:
                            specs["Description"] = desc
                    if "offers" in item:
                        offer = item["offers"]
                        if isinstance(offer, list):
                            offer = offer[0]
                        if "price" in offer:
                            currency = offer.get("priceCurrency", "")
                            price = offer["price"]
                            specs["Price"] = f"{currency} {price}".strip()
                        if "availability" in offer:
                            specs["Availability"] = offer["availability"].split("/")[-1]
                    if "aggregateRating" in item:
                        rating = item["aggregateRating"]
                        if "ratingValue" in rating:
                            specs["Rating"] = f"{rating['ratingValue']}/5"
                        if "reviewCount" in rating:
                            specs["Review Count"] = str(rating["reviewCount"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return specs

def fetch_product_specs(url: str, use_js_render: bool = False, prefer_playwright: bool = True, prefer_selenium: bool = True) -> Dict[str, str]:
    specs = {}
    if not url:
        return specs
    try:
        # Optionally render page
        html = None
        if use_js_render:
            html = fetch_rendered_html(url, prefer_playwright=prefer_playwright, prefer_selenium=prefer_selenium)
        else:
            try:
                r = SESSION.get(url, timeout=Config.DEFAULT_TIMEOUT)
                r.raise_for_status()
                html = r.text
            except Exception:
                html = fetch_rendered_html(url, prefer_playwright=prefer_playwright, prefer_selenium=prefer_selenium)
        if not html:
            error_tracker.add_warning("Specs Extraction", f"No HTML content for URL: {url}")
            return specs
        soup = BeautifulSoup(html, "html.parser")
        hostname = urlparse(url).netloc.lower()
        # Site-specific
        if "amazon." in hostname:
            s, _ = parse_amazon_specs_and_reviews(url, soup)
            specs.update(s)
        if "notebookcheck" in hostname:
            s, _ = parse_notebookcheck_specs_and_reviews(url, soup)
            specs.update(s)
        if "gsmarena.com" in hostname:
            specs.update(parse_gsmarena_specs(soup))
        # Generic strategies
        specs.update(extract_schema_org_data(soup))
        for table in soup.select("table"):
            for row in table.select("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    key = cells[0].get_text(" ", strip=True)
                    val = cells[-1].get_text(" ", strip=True)
                    if key and val and len(val) < Config.MAX_SPEC_VALUE_LENGTH:
                        specs.setdefault(key, val)
        for dl in soup.select("dl"):
            dts = dl.find_all("dt")
            dds = dl.find_all("dd")
            for dt, dd in zip(dts, dds):
                key = dt.get_text(" ", strip=True)
                val = dd.get_text(" ", strip=True)
                if key and val and len(val) < Config.MAX_SPEC_VALUE_LENGTH:
                    specs.setdefault(key, val)
        for elem in soup.select("[itemprop]"):
            prop = elem.get("itemprop")
            content = elem.get("content") or elem.get_text(" ", strip=True)
            if prop and content and len(content) < Config.MAX_SPEC_VALUE_LENGTH:
                specs.setdefault(prop.title(), content)
        spec_containers = soup.select(
            ".specs, .specifications, .product-details, .tech-specs, "
            "#specifications, .spec-list"
        )
        for container in spec_containers:
            for item in container.select("li, .spec-item"):
                text = item.get_text(" ", strip=True)
                if ":" in text and len(text.split(":")[0]) < 80:
                    parts = text.split(":", 1)
                    if len(parts) == 2:
                        key, val = parts
                        key = key.strip()
                        val = val.strip()
                        if key and val and len(val) < Config.MAX_SPEC_VALUE_LENGTH:
                            specs.setdefault(key, val)
        error_tracker.add_info("Specs Extraction", f"Extracted {len(specs)} specifications")
    except Exception as e:
        error_tracker.add_error("Specs Extraction", e, f"URL: {url}")
    cleaned_specs = {}
    for k, v in specs.items():
        k = k.strip()
        v = v.strip()
        if (k and v and len(v) > 0 and len(v) < Config.MAX_SPEC_VALUE_LENGTH
            and k.lower() not in ["share", "email", "print", "tweet", "save"]):
            cleaned_specs[k] = v
    return cleaned_specs

# ========== REVIEW QUALITY FILTER ==========
def is_quality_review(text: str) -> bool:
    text_lower = text.lower()
    if len(text) < Config.MIN_REVIEW_LENGTH or len(text) > Config.MAX_REVIEW_LENGTH:
        return False
    spam_patterns = [
        'advertisement', 'sponsored', 'click here', 'buy now',
        'terms of service', 'privacy policy', 'cookie',
        'sign up', 'subscribe', 'newsletter',
        'http://', 'https://', 'www.',
        'copyright ©', 'all rights reserved'
    ]
    if any(pattern in text_lower for pattern in spam_patterns):
        return False
    if sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.3:
        return False
    review_indicators = [
        'good', 'great', 'excellent', 'bad', 'poor', 'terrible',
        'battery', 'camera', 'screen', 'display', 'performance',
        'recommend', 'buy', 'bought', 'purchased', 'price',
        'quality', 'worth', 'fast', 'slow', 'love', 'hate'
    ]
    indicator_count = sum(1 for word in review_indicators if word in text_lower)
    if indicator_count < 2:
        return False
    punctuation_ratio = sum(1 for c in text if c in '!?.,;:') / max(len(text), 1)
    if punctuation_ratio > 0.15:
        return False
    if re.search(r'(.)\1{4,}', text):
        return False
    return True

# ========== PAGINATION HELPERS ==========
def find_next_page_link(soup, current_url: str, current_page: int) -> Optional[str]:
    next_candidates = [
        soup.find("a", string=re.compile(r"next", re.I)),
        soup.find("a", {"rel": "next"}),
        soup.select_one("a[aria-label*='Next' i]"),
        soup.select_one(".pagination a[rel='next']"),
        soup.select_one(".next a"),
        soup.select_one("a.next"),
    ]
    for candidate in next_candidates:
        if candidate and candidate.has_attr("href"):
            href = candidate["href"]
            return urljoin(current_url, href)
    page_links = soup.find_all("a", href=True)
    for link in page_links:
        href = link["href"]
        match = re.search(r"[?&]page=(\d+)", href)
        if match and int(match.group(1)) == current_page + 1:
            return urljoin(current_url, href)
        match = re.search(r"/page/(\d+)", href)
        if match and int(match.group(1)) == current_page + 1:
            return urljoin(current_url, href)
    parsed = urlparse(current_url)
    params = parse_qs(parsed.query)
    if "page" in params:
        try:
            current_page_num = int(params["page"][0])
            params["page"] = [str(current_page_num + 1)]
            new_query = urlencode(params, doseq=True)
            return parsed._replace(query=new_query).geturl()
        except (ValueError, IndexError):
            pass
    elif current_page == 1:
        separator = "&" if "?" in current_url else "?"
        return f"{current_url}{separator}page=2"
    return None

# ========== DEDUPE HELPERS (rapidfuzz) ==========
def is_similar_rapidfuzz(a: str, b: str, threshold: float = Config.DUPLICATE_SIMILARITY_THRESHOLD) -> bool:
    # rapidfuzz's ratio yields 0-100; convert threshold
    try:
        score = fuzz.token_sort_ratio(a, b)
        return score >= int(threshold * 100)
    except Exception:
        # fallback to difflib
        return SequenceMatcher(None, a, b).ratio() >= threshold

def dedupe_reviews(reviews: List[str]) -> List[str]:
    unique = []
    for r in reviews:
        if not any(is_similar_rapidfuzz(r, u) for u in unique):
            unique.append(r)
    return unique

# ========== REVIEW SCRAPER (improved dedupe, optional JS render) ==========
def fetch_reviews_generic(url: str, limit: int = 1000, polite: bool = True, respect_robots: bool = True, use_js_render: bool = False, prefer_playwright: bool = True, prefer_selenium: bool = True, site_parsers_enabled: bool = True) -> List[str]:
    """Enhanced review scraper with better deduplication."""
    reviews = []
    if not url:
        return reviews
    if respect_robots and not allowed_by_robots(url):
        error_tracker.add_warning("Review Scraping", f"Blocked by robots.txt: {url}")
        st.warning(f"⚠️ Scraping blocked by robots.txt for {urlparse(url).netloc}")
        return reviews
    page_url = url
    visited_urls = set()
    page_count = 0
    max_pages = Config.DEFAULT_MAX_PAGES
    consecutive_empty_pages = 0
    hostname = urlparse(url).netloc.lower()
    seen_signatures = set()
    try:
        while page_url and len(reviews) < limit and page_count < max_pages:
            if page_url in visited_urls:
                error_tracker.add_warning("Review Scraping", f"Already visited: {page_url}")
                break
            visited_urls.add(page_url)
            page_count += 1
            st.info(f"📖 Scraping page {page_count}... ({len(reviews)} reviews collected)")
            delay = Config.DEFAULT_RATE_LIMIT_DELAY * (2.0 if not polite else 1.0)
            time.sleep(delay)
            # Fetch page (optionally rendered)
            html = None
            if use_js_render:
                html = fetch_rendered_html(page_url, prefer_playwright=prefer_playwright, prefer_selenium=prefer_selenium)
                if not html:
                    # fallback to requests
                    try:
                        r = SESSION.get(page_url, timeout=Config.DEFAULT_TIMEOUT)
                        r.raise_for_status()
                        html = r.text
                    except Exception as e:
                        error_tracker.add_error("Review Scraping", e, f"Failed to fetch page {page_count}")
                        break
            else:
                try:
                    r = SESSION.get(page_url, timeout=Config.DEFAULT_TIMEOUT)
                    r.raise_for_status()
                    html = r.text
                except Exception:
                    # Try rendered fallback for JS heavy pages
                    html = fetch_rendered_html(page_url, prefer_playwright=prefer_playwright, prefer_selenium=prefer_selenium)
                    if not html:
                        error_tracker.add_error("Review Scraping", Exception("Failed to fetch page"), f"Page: {page_url}")
                        break
            soup = BeautifulSoup(html, "html.parser")
            page_review_blocks = []
            # Site-specific parsing
            if site_parsers_enabled:
                if "amazon." in hostname:
                    _, amazon_reviews = parse_amazon_specs_and_reviews(page_url, soup)
                    page_review_blocks.extend([BeautifulSoup(f"<div>{rv}</div>", "html.parser").div for rv in amazon_reviews if rv])
                elif "notebookcheck" in hostname:
                    _, nb_reviews = parse_notebookcheck_specs_and_reviews(page_url, soup)
                    page_review_blocks.extend([BeautifulSoup(f"<div>{rv}</div>", "html.parser").div for rv in nb_reviews if rv])
            # Generic selectors fallback
            if not page_review_blocks:
                selectors_to_try = [
                    ".review", ".user-review", ".review-item", ".review-body",
                    ".opinion", ".user-opinion", ".comment", ".testimonial",
                    "[itemprop='review']", ".customer-review", ".review-text", ".post"
                ]
                for sel in selectors_to_try:
                    found = soup.select(sel)
                    if found:
                        page_review_blocks.extend(found)
            # Fallback paragraphs (limit)
            if not page_review_blocks:
                for elem in soup.find_all(["p", "div"], limit=100):
                    text = elem.get_text(" ", strip=True)
                    if (50 < len(text) < 2000 and any(k in text.lower() for k in ["battery", "camera", "screen", "performance", "recommend"])):
                        page_review_blocks.append(elem)
            new_reviews_count = 0
            for block in page_review_blocks:
                if not block:
                    continue
                text = block.get_text(" ", strip=True)
                if not text:
                    continue
                if not is_quality_review(text):
                    continue
                # signature first 200 chars
                signature = text[:200]
                if signature in seen_signatures:
                    continue
                # Add
                reviews.append(text)
                seen_signatures.add(signature)
                new_reviews_count += 1
                if len(reviews) >= limit:
                    break
            error_tracker.add_info("Review Scraping", f"Page {page_count}: +{new_reviews_count} new (total: {len(reviews)})")
            if new_reviews_count == 0:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= Config.MAX_CONSECUTIVE_EMPTY_PAGES:
                    error_tracker.add_warning("Review Scraping", f"No new reviews for {Config.MAX_CONSECUTIVE_EMPTY_PAGES} consecutive pages")
                    break
            else:
                consecutive_empty_pages = 0
            next_link = find_next_page_link(soup, page_url, page_count)
            if next_link:
                page_url = next_link
            else:
                break
    except Exception as e:
        error_tracker.add_error("Review Scraping", e, f"URL: {url}, Page: {page_count}")
    st.success(f"✅ Scraping complete: {len(reviews)} reviews from {page_count} pages")
    error_tracker.add_info("Review Scraping", f"Final: {len(reviews)} reviews from {page_count} pages")
    return reviews[:limit]

# ========== GROQ API WITH RETRY ==========
def call_groq_with_retry(prompt: str, max_retries: int = 3) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
            error_tracker.add_info("Groq API", f"Successful call on attempt {attempt + 1}")
            return text
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                delay = Config.BACKOFF_FACTOR * (2 ** attempt)
                error_tracker.add_warning("Groq API", f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s")
                st.warning(f"⏳ API retry {attempt + 1}/{max_retries} (waiting {delay:.1f}s)")
                time.sleep(delay)
            else:
                error_tracker.add_error("Groq API", e, f"Failed after {max_retries} attempts")
                st.error(f"❌ API failed after {max_retries} attempts: {error_msg}")
                return None
    return None

# ========== PROMPT BUILDERS (category aware) ==========
def build_aspect_list_for_category(category: str) -> List[str]:
    cat = (category or "").lower()
    if "phone" in cat:
        return ["Battery","Camera","Display","Performance","Software","Value"]
    if "laptop" in cat:
        return ["CPU","GPU","RAM","Storage","Battery","Display","Build","Value"]
    if "accessory" in cat:
        return ["Build","Comfort","Compatibility","Value","Durability"]
    return ["Build","Performance","Value","Reliability","Usability"]

def chunk_prompt(product_name: str, specs: dict, reviews_subset: list, category: str = "Generic") -> str:
    specs_json = json.dumps(specs, ensure_ascii=False, indent=2) if specs else "{}"
    reviews_context = "\n".join([f"- {r}" for r in reviews_subset]) if reviews_subset else "No reviews found"
    aspects = build_aspect_list_for_category(category)
    aspects_text = ", ".join(aspects)
    prompt = f"""You are an AI Review Summarizer analyzing the product: {product_name}.
Category: {category}
Focus on these aspects when present: {aspects_text}

Combine scraped product specs with real user reviews to create a concise structured JSON summary.

SPECS (JSON):
{specs_json}

REVIEWS SAMPLE:
{reviews_context}

OUTPUT RULES:
- Return ONLY valid JSON (no markdown, no commentary).
- Do not split or spell words character-by-character.
- Arrays (pros, cons, user_quotes) must be JSON arrays of full strings.
- Keep 'verdict' under 30 characters and 'recommendation' under 35 characters.
- Provide 2-6 user quotes (1-2 sentences each).
- Include all scraped specs inside the product_specs object exactly as key/value pairs.
- For aspect_sentiments, include the aspects above when possible and rate Positive/Negative from 0-100 integers.

Return a JSON object matching this example schema:
{{
  "verdict": "Short verdict",
  "pros": ["Pro 1", "Pro 2"],
  "cons": ["Con 1", "Con 2"],
  "aspect_sentiments": [
    {{"Aspect":"{aspects[0]}","Positive":75,"Negative":25}}
  ],
  "user_quotes": ["Quote 1", "Quote 2"],
  "recommendation": "Short target audience",
  "bottom_line": "2-3 sentence final summary combining specs & reviews",
  "product_specs": {specs_json}
}}"""
    return prompt

def final_merge_prompt(product_name: str, specs: dict, partial_texts: list, category: str = "Generic") -> str:
    joined = "\n\n---- PARTIAL SUMMARY ----\n\n".join(partial_texts)
    specs_json = json.dumps(specs, ensure_ascii=False, indent=2) if specs else "{}"
    aspects = build_aspect_list_for_category(category)
    aspects_text = ", ".join(aspects)
    prompt = f"""You are an AI assistant. You are given multiple JSON partial analyses for the same product ({product_name}).
Category: {category}
Focus aspects: {aspects_text}

Each partial analysis is valid JSON (or near-JSON). Your job: MERGE them into ONE VALID JSON object that exactly matches the schema below.

Requirements:
- Return ONLY valid JSON (no explanation, no markdown).
- Ensure 'pros' and 'cons' are arrays of short phrases.
- Deduplicate pros/cons while keeping meaningful phrases.
- Aggregate user quotes (2-6 unique quotes total).
- For aspect_sentiments, combine by taking the arithmetic mean of Positive and Negative across partials (round to nearest integer).
- Ensure product_specs contains the scraped specs (use the scraped specs preferentially).
- Keep verdict under 30 chars, recommendation under 35 chars.

Partial analyses:
{joined}

Now output a single JSON object with this schema:
{{
  "verdict": "Short verdict under 30 chars",
  "pros": ["Pro 1", "Pro 2"],
  "cons": ["Con 1", "Con 2"],
  "aspect_sentiments": [
    {{"Aspect":"{aspects[0]}","Positive":75,"Negative":25}}
  ],
  "user_quotes": ["quote1","quote2"],
  "recommendation": "short target audience under 35 chars",
  "bottom_line": "2-3 sentence final summary combining specs & reviews",
  "product_specs": {specs_json}
}}"""
    return prompt

# ========== SUMMARIZATION ==========
def summarize_reviews_chunked(product_name: str, specs: dict, reviews: list, chunk_size: int = None, category: str = "Generic", prog: Optional[st.delta_generator] = None) -> Optional[str]:
    if chunk_size is None:
        chunk_size = Config.DEFAULT_CHUNK_SIZE
    if not reviews:
        st.info("No reviews found, generating summary from specs only...")
        prompt = chunk_prompt(product_name, specs, [], category)
        return call_groq_with_retry(prompt)
    chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
    partial_texts = []
    total = len(chunks)
    failed_chunks = 0
    st.info(f"🔄 Processing {total} chunks ({chunk_size} reviews per chunk)...")
    for idx, chunk in enumerate(chunks, start=1):
        # progress update
        if prog:
            progress_pct = int(60 + (idx / total) * 30)  # 60% -> 90%
            try:
                prog.progress(progress_pct)
            except Exception:
                pass
        st.info(f"🤖 Processing chunk {idx}/{total} ({len(chunk)} reviews)")
        prompt = chunk_prompt(product_name, specs, chunk, category)
        result = call_groq_with_retry(prompt, max_retries=3)
        if result:
            partial_texts.append(result)
            st.success(f"✅ Chunk {idx}/{total} completed")
        else:
            failed_chunks += 1
            partial_texts.append("{}")
            st.warning(f"⚠️ Chunk {idx}/{total} failed")
        time.sleep(0.5)
    st.info(f"📊 Processed {total} chunks ({failed_chunks} failed)")
    if failed_chunks > total * 0.5:
        st.error(f"❌ Too many chunks failed ({failed_chunks}/{total}). Cannot proceed.")
        error_tracker.add_error("Summarization", Exception(f"Too many failed chunks: {failed_chunks}/{total}"))
        return None
    st.info("🔀 Merging partial summaries into final analysis...")
    merge_prompt_text = final_merge_prompt(product_name, specs, partial_texts, category)
    final_text = call_groq_with_retry(merge_prompt_text, max_retries=5)
    if final_text:
        st.success("✅ Final merge completed")
    return final_text

# ========== UI: Streamlit ==========
st.set_page_config(page_title="AI Product Review Engine", page_icon="📦", layout="wide")
st.title("📦 AI-Powered Product Review Engine — Full Improved")
st.markdown("""
**Features:** polite scraping, robots.txt respect, site-specific parsers (Amazon, Notebookcheck, GSMArena),
rapidfuzz deduplication, Playwright/Selenium JS rendering fallback, category-aware prompts, caching, and exports.
""")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
review_limit = st.sidebar.slider("Max reviews to analyze", 50, 1000, 400, step=50)
chunk_size = st.sidebar.slider("Reviews per chunk", 100, 300, Config.DEFAULT_CHUNK_SIZE, step=50)
show_raw = st.sidebar.checkbox("Show raw AI JSON", value=False)
show_debug = st.sidebar.checkbox("Show debug info", value=False)
clear_cache = st.sidebar.button("🗑️ Clear Cache")
respect_robots_toggle = st.sidebar.checkbox("Respect robots.txt", value=True)
polite_mode = st.sidebar.checkbox("Polite scraping mode (slower)", value=True)
dedupe_toggle = st.sidebar.checkbox("Deduplicate similar reviews (rapidfuzz)", value=True)
site_parsers_toggle = st.sidebar.checkbox("Use site-specific parsers (Amazon/Notebookcheck/GSMArena)", value=True)
use_js_render_toggle = st.sidebar.checkbox("Use JS-rendering fallback (Playwright/Selenium) when needed", value=False)
prefer_playwright_toggle = st.sidebar.checkbox("Prefer Playwright over Selenium", value=True)

if clear_cache:
    for cache_file in CACHE_DIR.glob("*.pkl"):
        try:
            cache_file.unlink()
        except Exception:
            pass
    st.sidebar.success("✅ Cache cleared!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Instructions")
st.sidebar.markdown("""
1. Enter a product name OR paste a direct URL  
2. Choose category to help the model (Auto-detect recommended)  
3. Toggle JS rendering only if Playwright/Selenium are installed and you need dynamic pages  
4. Click **Analyze Product**
""")

# Session state init
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

product = st.text_input("Enter product name (e.g., Samsung Galaxy S24 FE)", value="Samsung Galaxy S24")
product_url = st.text_input("Or paste direct product URL (optional)", value="")
category = st.selectbox("Product category", ["Auto-detect / Generic", "Phone", "Laptop", "Accessory", "Generic"])
analyze = st.button("🔍 Analyze Product", type="primary")

# Main flow
if analyze and (product or product_url):
    # BUG FIX: reset module-level error_tracker instance
    error_tracker = ErrorTracker()

    prog = st.progress(0)
    start_time = time.time()

    # Resolve product URL (with caching)
    st.info("🔎 Resolving product page...")
    cache_key = get_cache_key(f"resolve_{product}_{product_url}")
    cached_urls = load_from_cache(cache_key, Config.CACHE_TTL_HOURS['urls'])
    if cached_urls:
        st.success("✅ Using cached URL")
        product_url_resolved, review_url = cached_urls
    else:
        def resolve_product_url(product_name: str = None, product_url_in: str = None) -> Tuple[Optional[str], Optional[str]]:
            if product_url_in:
                u = normalize_url(product_url_in)
                valid, err = validate_url(u)
                if not valid:
                    error_tracker.add_error("URL Validation", Exception(err))
                    return None, None
                review = None
                if site_parsers_toggle and "gsmarena.com" in u:
                    review = build_gsmarena_review_url(u)
                if site_parsers_toggle and "amazon." in u:
                    pid = extract_product_id(u, 'amazon')
                    if pid:
                        review = f"https://www.amazon.com/product-reviews/{pid}"
                return u, review
            if product_name:
                try:
                    query = product_name.strip()
                    search_url = f"https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={requests.utils.quote(query)}"
                    r = SESSION.get(search_url, timeout=Config.DEFAULT_TIMEOUT)
                    r.raise_for_status()
                    soup = BeautifulSoup(r.text, "html.parser")
                    link = soup.select_one(".makers a") or soup.select_one("a[href*='.php']")
                    if link and link.has_attr("href"):
                        purl = "https://www.gsmarena.com/" + link["href"]
                        review_u = build_gsmarena_review_url(purl)
                        error_tracker.add_info("URL Resolution", f"Found GSMArena product: {purl}")
                        return purl, review_u
                except Exception as e:
                    error_tracker.add_error("GSMArena Search", e)
                    return None, None
            return None, None

        product_url_resolved, review_url = resolve_product_url(product, product_url)
        if product_url_resolved:
            save_to_cache(cache_key, (product_url_resolved, review_url))
    if not product_url_resolved:
        st.error(f"❌ Could not find product page for '{product}'. Try pasting a direct URL.")
        error_tracker.display_report()
        st.stop()

    st.success(f"✅ Product page: {product_url_resolved}")
    if review_url:
        st.success(f"✅ Reviews page (derived): {review_url}")
    else:
        st.info("ℹ️ Will attempt to scrape reviews from product page")

    prog.progress(15)

    # Fetch specs
    st.info("📊 Fetching product specifications...")
    cache_key = get_cache_key(f"specs_{product_url_resolved}")
    specs = load_from_cache(cache_key, Config.CACHE_TTL_HOURS['specs'])
    if specs:
        st.success(f"✅ Using cached specs ({len(specs)} specifications)")
    else:
        specs = fetch_product_specs(product_url_resolved, use_js_render=use_js_render_toggle, prefer_playwright=prefer_playwright_toggle, prefer_selenium=not prefer_playwright_toggle)
        if specs:
            save_to_cache(cache_key, specs)
        st.success(f"✅ Extracted {len(specs)} specifications")
    prog.progress(30)

    # Fetch reviews
    st.info("💬 Collecting user reviews...")
    reviews_source = review_url if review_url else product_url_resolved
    cache_key = get_cache_key(f"reviews_{reviews_source}", {"limit": review_limit})
    reviews = load_from_cache(cache_key, Config.CACHE_TTL_HOURS['reviews'])
    if reviews:
        st.success(f"✅ Using cached reviews ({len(reviews)} reviews)")
    else:
        reviews = fetch_reviews_generic(
            reviews_source,
            limit=review_limit,
            polite=polite_mode,
            respect_robots=respect_robots_toggle,
            use_js_render=use_js_render_toggle,
            prefer_playwright=prefer_playwright_toggle,
            prefer_selenium=not prefer_playwright_toggle,
            site_parsers_enabled=site_parsers_toggle
        )
        if reviews:
            save_to_cache(cache_key, reviews)
    st.info(f"✅ Collected {len(reviews)} reviews (limit: {review_limit})")
    prog.progress(50)

    # Optional dedupe
    if dedupe_toggle and reviews:
        before = len(reviews)
        reviews = dedupe_reviews(reviews)
        after = len(reviews)
        st.info(f"🧹 Deduplicated reviews: {before} -> {after}")

    # Category auto-detection (improved)
    def detect_product_category(product_name: str, url: str, specs: dict) -> str:
        product_lower = (product_name or "").lower()
        url_lower = (url or "").lower()
        specs_str = " ".join(str(v) for v in specs.values()).lower() if specs else ""
        phone_keywords = ["phone", "galaxy", "iphone", "pixel", "oppo", "xiaomi", "oneplus", "redmi", "realme", "vivo", "huawei", "smartphone"]
        phone_sites = ["gsmarena.com", "phonearena.com"]
        if any(k in product_lower for k in phone_keywords[:5]):
            return "Phone"
        if any(site in url_lower for site in phone_sites):
            return "Phone"
        if any(k in specs_str for k in ["android", "ios", "snapdragon", "mediatek", "exynos"]):
            return "Phone"
        laptop_keywords = ["laptop", "notebook", "macbook", "thinkpad", "ideapad", "chromebook", "ultrabook", "gaming laptop", "zenbook", "vivobook"]
        laptop_sites = ["notebookcheck.net", "notebookreview.com", "laptopmag.com"]
        if any(k in product_lower for k in laptop_keywords):
            return "Laptop"
        if any(site in url_lower for site in laptop_sites):
            return "Laptop"
        if any(k in specs_str for k in ["intel core", "ryzen", "nvidia", "radeon", "ram", "ssd"]):
            return "Laptop"
        accessory_keywords = ["case", "cover", "charger", "cable", "adapter", "headphone", "earbuds", "airpods", "mouse", "keyboard", "stand", "screen protector"]
        if any(k in product_lower for k in accessory_keywords):
            return "Accessory"
        return "Generic"

    chosen_category = category
    if chosen_category == "Auto-detect / Generic":
        chosen_category = detect_product_category(product, product_url_resolved, specs)
        st.info(f"🔍 Detected category: {chosen_category}")
    prog.progress(60)

    # Summarize with AI
    st.info("🤖 Analyzing with Groq AI (this may take a minute)...")
    final_json_text = summarize_reviews_chunked(product, specs, reviews, chunk_size, chosen_category, prog=prog)
    prog.progress(100)

    if not final_json_text:
        st.error("❌ Failed to generate summary")
        error_tracker.display_report()
        st.stop()

    final_obj = safe_load_json(final_json_text)
    if not final_obj:
        st.error("⚠️ Could not parse AI output as valid JSON")
        with st.expander("🔍 Raw AI output for debugging"):
            st.text_area("AI output", final_json_text, height=400)
        error_tracker.display_report()
        st.stop()

    product_specs = final_obj.get("product_specs", {})
    for k, v in specs.items():
        product_specs.setdefault(k, v)
    final_obj["product_specs"] = product_specs

    elapsed_time = time.time() - start_time

    # Save to session history
    st.session_state.last_analysis = {
        'product': product,
        'timestamp': datetime.now(),
        'result': final_obj,
        'specs_count': len(product_specs),
        'reviews_count': len(reviews)
    }
    st.session_state.analysis_history.append(st.session_state.last_analysis)
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history = st.session_state.analysis_history[-10:]

    # Display results
    st.markdown("---")
    st.subheader(f"📝 Analysis Results — {product}")
    st.caption(f"⏱️ Analysis completed in {elapsed_time:.1f} seconds")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div style='background:#0b0b0b;color:white;padding:15px;border-radius:8px;text-align:center;'>"
            f"<b>⭐ Verdict</b><div style='font-size:20px;margin-top:8px'>{final_obj.get('verdict','N/A')}</div></div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div style='background:#0b0b0b;color:white;padding:15px;border-radius:8px;text-align:center;'>"
            f"<b>🎯 Best For</b><div style='font-size:20px;margin-top:8px'>{final_obj.get('recommendation','N/A')}</div></div>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"<div style='background:#0b0b0b;color:white;padding:15px;border-radius:8px;text-align:center;'>"
            f"<b>📊 Data</b><div style='font-size:20px;margin-top:8px'>{len(product_specs)} specs, {len(reviews)} reviews</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("### 🎯 Bottom Line")
    st.info(final_obj.get("bottom_line", "No summary available"))

    left_col, right_col = st.columns([0.6, 0.4])
    with left_col:
        st.markdown("### 🔧 Product Specifications")
        if product_specs:
            df_specs = pd.DataFrame(list(product_specs.items()), columns=["Component", "Details"])
            st.dataframe(df_specs, use_container_width=True, hide_index=True)
        else:
            st.info("No specifications found")
        if final_obj.get("aspect_sentiments"):
            st.markdown("### 📊 User Sentiment by Aspect")
            df_aspects = pd.DataFrame(final_obj["aspect_sentiments"])
            if not df_aspects.empty and "Aspect" in df_aspects.columns:
                df_chart = df_aspects.set_index("Aspect")
                if "Positive" in df_chart.columns and "Negative" in df_chart.columns:
                    st.bar_chart(df_chart[["Positive", "Negative"]], height=320)

    with right_col:
        st.markdown("### ✅ Strengths")
        pros = final_obj.get("pros", [])
        if pros:
            for pro in pros:
                st.success(f"✓ {pro}")
        else:
            st.info("No strengths identified")
        st.markdown("### ⚠️ Weaknesses")
        cons = final_obj.get("cons", [])
        if cons:
            for con in cons:
                st.error(f"✗ {con}")
        else:
            st.info("No weaknesses identified")

    if final_obj.get("user_quotes"):
        st.markdown("### 💬 What Users Are Saying")
        for i, quote in enumerate(final_obj.get("user_quotes", []), 1):
            st.info(f"**User {i}:** _{quote}_")

    # Export options
    st.markdown("### 📤 Export Options")
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        filename_safe = (product or "product").replace(" ", "_").replace("/", "-")[:80]
        st.download_button("📥 Download JSON", json.dumps(final_obj, indent=2, ensure_ascii=False), f"{filename_safe}_analysis.json", "application/json", help="Download complete analysis as JSON")
    with export_col2:
        if product_specs:
            csv_data = pd.DataFrame(list(product_specs.items()), columns=["Specification", "Value"]).to_csv(index=False)
            st.download_button("📊 Download Specs CSV", csv_data, f"{filename_safe}_specs.csv", "text/csv", help="Download specifications as CSV")
    with export_col3:
        text_report = f"""PRODUCT ANALYSIS REPORT
{'=' * 50}
Product: {product}
Category: {chosen_category}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

VERDICT: {final_obj.get('verdict', 'N/A')}
RECOMMENDATION: {final_obj.get('recommendation', 'N/A')}

BOTTOM LINE:
{final_obj.get('bottom_line', 'N/A')}

STRENGTHS:
{chr(10).join(f'  ✓ {p}' for p in final_obj.get('pros', []))}

WEAKNESSES:
{chr(10).join(f'  ✗ {c}' for c in final_obj.get('cons', []))}

USER QUOTES:
{chr(10).join(f'  \"{q}\"' for q in final_obj.get('user_quotes', []))}

DATA SOURCES:
  • Specifications: {len(product_specs)} extracted
  • Reviews analyzed: {len(reviews)}
  • Product URL: {product_url_resolved}
"""
        st.download_button("📄 Download Report", text_report, f"{filename_safe}_report.txt", "text/plain", help="Download human-readable text report")

    # Debug & history
    if show_raw:
        with st.expander("📄 Raw AI JSON"):
            st.json(final_obj)
    if show_debug:
        error_tracker.display_report()

# Sidebar: history
if st.session_state.analysis_history:
    st.sidebar.markdown("### 📜 Recent Analyses")
    for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:]), 1):
        st.sidebar.caption(f"{i}. {analysis['product']} ({analysis['timestamp'].strftime('%H:%M')})")

st.markdown("---")
st.caption("🤖 Powered by Groq AI | Made with Streamlit")