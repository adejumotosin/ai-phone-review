import streamlit as st
import json
from groq import Groq
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

# --- 1. Pydantic Schema for Structured Output ---
class ProductReview(BaseModel):
    """A comprehensive product review based on inferred specifications."""
    product_name: str = Field(description="The full name of the product being reviewed.")
    specifications_inferred: str = Field(description="A concise summary of the key technical specs the review is based on (e.g., '6.1-inch OLED, A16 Bionic, 48MP main camera, $999 USD').")
    predicted_rating: str = Field(description="A critical rating out of 5.0 (e.g., '4.6 / 5.0').")
    pros: List[str] = Field(description="A list of predicted strengths and advantages based on the specs and market context.")
    cons: List[str] = Field(description="A list of predicted weaknesses, trade-offs, or user pain points (e.g., high price, outdated charging port).")
    verdict: str = Field(description="A concluding summary of the product's overall value proposition and who it is best suited for.")


# --- 2. System Prompts ---
INITIAL_REVIEW_PROMPT = f"""
You are an **Expert Product Reviewer and Critical Market Analyst**. Your task is to generate a comprehensive, structured product review for the user.

**Process:**
1.  **Data Retrieval (Simulated):** First, search your knowledge base for the most common, key specifications for the product name provided by the user. If the product name is too vague, use your best judgment for a well-known, high-end, or representative model.
2.  **Critical Review:** Use the specifications you retrieved to perform a critical, unbiased review. Point out the trade-offs, likely user experience issues, and whether the product is competitive in the current market.

**Crucial Constraints:**
1.  **DO NOT mention the source of the specifications.** State them as fact.
2.  **NEVER use the word "simulated" or "inferred".** Present the review as a standard professional analysis.
3.  **Format your entire output STRICTLY as a single JSON object** that conforms to the following schema. Ensure all fields are populated correctly.
{ProductReview.schema_json(indent=2)}
"""

CHAT_SYSTEM_PROMPT = """
You are an **Expert Product Reviewer and Technical Consultant**. You have just provided a comprehensive review of a product, and now the user has follow-up questions.

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

# --- 3. Streamlit Application Interface ---

# Page Configuration
st.set_page_config(
    page_title="AI Product Review Chat",
    page_icon="🤖",
    layout="wide"
)

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
    st.error("Error: Groq API key not found in `.streamlit/secrets.toml`. Please set it up.")
    st.stop()

# --- Helper Functions ---

def generate_initial_review(product_name):
    """Generate the initial structured review"""
    user_prompt = f"Generate a detailed, critical review for the product: {product_name}"
    json_schema = {"type": "json_object"}
    model_choice = "llama-3.3-70b-versatile"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": INITIAL_REVIEW_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model=model_choice,
            response_format=json_schema
        )
        
        json_string = chat_completion.choices[0].message.content
        review_data = json.loads(json_string)
        return review_data, None
    except Exception as e:
        return None, str(e)

def chat_with_ai(user_message, conversation_history):
    """Continue the conversation about the product"""
    model_choice = "llama-3.3-70b-versatile"
    
    try:
        # Build messages with context
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_choice,
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
    
    # Header with rating
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"📱 {review_data.get('product_name', 'Unknown Product')}")
    with col2:
        rating = review_data.get('predicted_rating', 'N/A')
        st.markdown(f"### ⭐ {rating}")
    
    # Specifications
    st.subheader("🔧 Key Specifications")
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
            st.metric("Rating", st.session_state.review_data.get('predicted_rating', 'N/A'))
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
        2. Get instant AI review
        3. Ask follow-up questions
        
        **Example Questions:**
        - "How does it compare to [competitor]?"
        - "Is it good for gaming?"
        - "What about battery life?"
        - "Should I wait for the next version?"
        - "Is it worth the price?"
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

# Main Content Area
if not st.session_state.chat_mode:
    # Initial product search interface
    st.title("🤖 AI Product Review Assistant")
    st.markdown("### Get expert reviews and ask questions about any product")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        product_input = st.text_input(
            "Enter Product Name",
            placeholder="e.g., Sony WH-1000XM5, MacBook Pro M3, Nintendo Switch OLED",
            label_visibility="collapsed"
        )
    
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
    
    if search_button and product_input:
        with st.spinner(f"🔍 Analyzing '{product_input}'..."):
            review_data, error = generate_initial_review(product_input)
            
            if error:
                st.error(f"❌ Error: {error}")
            elif review_data:
                st.session_state.current_product = product_input
                st.session_state.review_data = review_data
                st.session_state.chat_mode = True
                
                # Add initial review to conversation history
                review_summary = f"""I've analyzed the {review_data.get('product_name')}. Here's my review:

**Rating:** {review_data.get('predicted_rating')}

**Key Specs:** {review_data.get('specifications_inferred')}

**Strengths:** {', '.join(review_data.get('pros', [])[:3])}

**Weaknesses:** {', '.join(review_data.get('cons', [])[:3])}

**Verdict:** {review_data.get('verdict')}

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
    
    # Suggested questions (only show if no messages yet)
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
                    # Trigger the question
                    user_message = question
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    
                    with st.spinner("🤔 Thinking..."):
                        response, error = chat_with_ai(user_message, st.session_state.messages[:-1])
                        
                        if error:
                            st.error(f"Error: {error}")
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

# Footer
st.markdown("---")
st.caption("🤖 Powered by Llama 3.3 70B via Groq • Built with Streamlit")