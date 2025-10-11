import streamlit as st
import json
from groq import Groq
from pydantic import BaseModel, Field
from typing import List

# --- 1. Pydantic Schema for Structured Output ---
class ProductReview(BaseModel):
    """A comprehensive product review based on inferred specifications."""
    product_name: str = Field(description="The full name of the product being reviewed.")
    specifications_inferred: str = Field(description="A concise summary of the key technical specs the review is based on (e.g., '6.1-inch OLED, A16 Bionic, 48MP main camera, $999 USD').")
    predicted_rating: str = Field(description="A critical rating out of 5.0 (e.g., '4.6 / 5.0').")
    pros: List[str] = Field(description="A list of predicted strengths and advantages based on the specs and market context.")
    cons: List[str] = Field(description="A list of predicted weaknesses, trade-offs, or user pain points (e.g., high price, outdated charging port).")
    verdict: str = Field(description="A concluding summary of the product's overall value proposition and who it is best suited for.")


# --- 2. System Prompt Definition ---
SYSTEM_PROMPT = f"""
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

# --- 3. Streamlit Application Interface ---

# Page Configuration
st.set_page_config(
    page_title="AI Product Review Generator",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Product Review Generator")
st.caption("Simply enter the product name. Our AI Market Analyst will retrieve the specs and generate a critical review instantly.")

# Groq Client Initialization (using Streamlit secrets)
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=groq_api_key)
except Exception:
    st.error("Error: Groq API key not found in `.streamlit/secrets.toml`. Please set it up.")
    st.stop()

# --- User Input Form (Updated with CURRENT supported models) ---
with st.form("product_review_form"):
    product_name = st.text_input(
        "Enter Product Name",
        "Samsung Galaxy S24 Ultra", # Example for the user
        placeholder="e.g., Sony WH-1000XM5, Nintendo Switch OLED, MacBook Pro M3"
    )
    
    # Model selection - UPDATED with currently active models as of late 2024/early 2025
    model_choice = st.selectbox(
        "Select AI Model",
        (
            "llama-3.3-70b-versatile",      # Latest 70B Llama model
            "llama-3.1-8b-instant",         # Fast 8B model
            "llama3-groq-70b-8192-tool-use-preview",  # Tool use optimized
            "llama3-groq-8b-8192-tool-use-preview",   # Smaller tool use model
            "mixtral-8x7b-32768",           # Mixtral model
            "gemma2-9b-it",                 # Google Gemma
            "llama3-8b-8192",               # Standard Llama 3 8B
        ),
        index=0,
        help="llama-3.3-70b-versatile is recommended for best quality and complex reasoning"
    )

    submit_button = st.form_submit_button("Generate Review")

# --- 4. Logic to Call the Groq API ---

if submit_button:
    if not product_name:
        st.error("Please enter a product name.")
    else:
        user_prompt = f"Generate a detailed, critical review for the product: {product_name}"
        json_schema = {"type": "json_object"}

        # Display waiting message
        with st.spinner(f"Searching for specs and analyzing '{product_name}'..."):
            try:
                # Call the Groq API
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=model_choice, 
                    response_format=json_schema
                )
                
                # Parse the JSON string output
                json_string = chat_completion.choices[0].message.content
                review_data = json.loads(json_string)
                
                # --- 5. Display the Structured Output ---
                
                final_product_name = review_data.get('product_name', product_name)
                
                st.header(f"Review: {final_product_name}")
                st.markdown(f"### Score: **{review_data.get('predicted_rating', 'N/A')}**", unsafe_allow_html=True)
                
                st.subheader("Key Specifications Analyzed")
                st.info(review_data.get('specifications_inferred', 'Specifications could not be determined.'))

                st.divider()

                # Pros Section
                col_pros, col_cons = st.columns(2)
                
                with col_pros:
                    st.markdown("### 🟢 Strengths (Pros)")
                    for pro in review_data.get('pros', []):
                        st.markdown(f"- **{pro}**")

                # Cons Section
                with col_cons:
                    st.markdown("### 🔴 Weaknesses (Cons)")
                    for con in review_data.get('cons', []):
                        st.markdown(f"- **{con}**")

                # Verdict
                st.markdown("### ✅ Verdict")
                st.write(review_data.get('verdict', 'No final verdict provided.'))

                st.markdown("---")
                st.caption(f"Analysis generated by {model_choice} via Groq.")

            except json.JSONDecodeError:
                st.error("The AI failed to return a valid, structured JSON review. Try using a more specific product name or a different model.")
                st.code(json_string, language='json')
            except Exception as e:
                st.error(f"An unexpected API error occurred: {e}")

# --- How to Run the App ---
# 1. Save the code as `app.py`.
# 2. Ensure your `requirements.txt` is installed and `.streamlit/secrets.toml` is set up.
# 3. Run in your terminal: `streamlit run app.py`