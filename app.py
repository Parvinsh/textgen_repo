import streamlit as st
from fetcher import fetch_text_from_url
from retriever import initialize_retrieval_chain, generate_long_answer
from pdf_generator import save_to_pdf
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")

# Load environment variables (for API key)
load_dotenv()
GroqClod_API_KEY = os.getenv('GroqClod_API_KEY_8b')

# Initialize Streamlit session state
if "llm" not in st.session_state:
    from langchain_groq import ChatGroq
    st.session_state["llm"] = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=GroqClod_API_KEY, 
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

# App layout
st.title("Web Link Information Synthesis")
st.write("Enter at least 3 links, and then ask questions based on the combined information.")

# Reset button
if st.button("Reset App"):
    st.session_state.clear()
    st.success("App reset successfully. Please re-enter your links.")

# Input links
link1 = st.text_input("Enter Link 1")
link2 = st.text_input("Enter Link 2")
link3 = st.text_input("Enter Link 3")
additional_links = st.text_area("Enter additional links, one per line (optional)")
links = [link for link in [link1, link2, link3] if link] + additional_links.splitlines()

# Process links
if len(links) >= 3:
    if "retrieval_chain" not in st.session_state:
        st.write("Fetching and processing content from links...")
        try:
            st.session_state["retrieval_chain"] = initialize_retrieval_chain(links, st.session_state["llm"])
            st.success("Content processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error initializing retrieval chain: {e}")
else:
    st.info("Please enter at least 3 valid links to proceed.")

# Question answering
if "retrieval_chain" in st.session_state:
    user_query = st.text_area("Enter your query about the information from the links:", height=150)
    if user_query:
        try:
            answer = generate_long_answer(st.session_state["retrieval_chain"], user_query, max_length=5000)
            st.write("Answer:", answer)
            
            # Save answer as PDF
            if st.button("Save as PDF"):
                pdf_file = save_to_pdf(answer)
                with open(pdf_file, "rb") as f:
                    st.download_button("Download PDF", f, file_name="answer_output.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"An error occurred: {e}")

