import streamlit as st
import json
from groq import Groq
from pydantic import BaseModel, Field
from typing import List

# --- 1. Pydantic Schema for Structured Output ---
# This defines the exact structure we want the LLM to return.
class SimulatedReview(BaseModel):
    """A critical, simulated product review based on specifications."""
    product_name: str = Field(description="The name of the product being reviewed.")
    predicted_rating: str = Field(description="A predicted rating out of 5.0 (e.g., '4.2 / 5.0').")
    pros: List[str] = Field(description="A list of predicted strengths and advantages based on the specs and market context.")
    cons: List[str] = Field(description="A list of predicted weaknesses, trade-offs, and likely user pain points (e.g., poor battery for a high-res screen).")
    market_position: str = Field(description="A critical analysis of how this product's specs position it against the current market (e.g., 'strong budget competitor' or 'overpriced for features').")

# --- 2. System Prompt Definition ---
# This is the core instruction that makes the AI act as an expert critic without web scraping.
SYSTEM_PROMPT = """
You are an **Expert Product Reviewer and Critical Market Analyst**. Your sole function is to provide a comprehensive, critical, and simulated review of a product based *only* on the specifications the user provides.

**Crucial Constraints:**
1.  **NEVER use language that suggests you have read customer reviews.** You must rely on first principles, industry knowledge, and logical deductions.
2.  **Avoid web search or external data retrieval.** Your analysis is grounded in your massive general training data's knowledge of industry standards, typical user behavior, component trade-offs, and historical market trends (e.g., a fast processor plus small battery equals poor endurance).
3.  **Be highly critical and unbiased.** Point out likely bottlenecks and trade-offs clearly.
4.  **Format your entire output STRICTLY as a single JSON object** that conforms to the provided Pydantic schema.
"""

# --- 3. Streamlit Application Interface ---

# Page Configuration
st.set_page_config(
    page_title="Pure AI Product Review Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Pure AI Product Review Chatbot")
st.caption("Enter product specifications below. The AI will act as a critical market analyst and generate a simulated, structured review **without using any web-scraped data.**")

# Groq Client Initialization (using Streamlit secrets)
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error("Error: Groq API key not found in `.streamlit/secrets.toml`. Please set it up.")
    st.stop()

# --- User Input Form ---
with st.form("product_review_form"):
    product_name = st.text_input("Product Name (e.g., 'ChronoWatch Pro 2')", "NovaPhone X1")
    specifications = st.text_area(
        "Product Specifications (Required)",
        "6-inch 60Hz LCD screen, 4000mAh battery, 64MP camera, plastic build, launches at $600."
    )
    competitors = st.text_input(
        "Target Competitor (Optional for context)",
        "Compare against the current market leader, the 'Aura S10' (120Hz OLED, $650)."
    )

    submit_button = st.form_submit_button("Generate Simulated Review")

# --- 4. Logic to Call the Groq API ---

if submit_button:
    if not specifications:
        st.error("Please provide the product specifications to generate a review.")
    else:
        # Construct the detailed user prompt
        user_prompt = f"""
        Analyze the following product.

        **Product Name:** {product_name}
        **Specifications:** {specifications}
        **Context/Comparison:** {competitors if competitors else 'No specific competitor provided.'}

        Based on these facts, generate a critical, simulated product review.
        """

        # Display waiting message
        with st.spinner("Analyzing specifications and generating critical review..."):
            try:
                # Call the Groq API with structured output enforcement
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama3-70b-8192",  # A powerful model for complex reasoning
                    response_model=SimulatedReview
                )

                # The output is a Pydantic object, which can be easily used
                review_data = chat_completion.model_dump()
                
                # --- 5. Display the Structured Output ---
                st.subheader(f"Simulated Review: {review_data.get('product_name', 'N/A')}")
                st.markdown(f"### Predicted Rating: **{review_data.get('predicted_rating', 'N/A')}**", unsafe_allow_html=True)
                
                st.divider()

                # Pros Section
                st.markdown("### 🟢 Predicted Strengths (Pros)")
                for pro in review_data.get('pros', []):
                    st.markdown(f"- **{pro}**")

                # Cons Section
                st.markdown("### 🔴 Predicted Weaknesses & Trade-offs (Cons)")
                for con in review_data.get('cons', []):
                    st.markdown(f"- **{con}**")

                # Market Position
                st.markdown("### 📊 Market Position Analysis")
                st.info(review_data.get('market_position', 'No analysis available.'))
                
                st.markdown("---")
                st.caption("💡 This review is a critical market deduction, not based on real user feedback.")

            except Exception as e:
                st.error(f"An error occurred during API call: {e}")
                st.warning("This error often means the model couldn't format the output correctly. Try simplifying the specifications or re-running.")

# --- How to Run the App ---
# 1. Save the code as `app.py`.
# 2. Run in your terminal: `streamlit run app.py`
