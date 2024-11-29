import requests
from bs4 import BeautifulSoup
import streamlit as st

def fetch_text_from_url(url):
    """Fetch and clean text from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(paragraph.text for paragraph in paragraphs)
        return text
    except Exception as e:
        st.error(f"Error fetching content from {url}: {e}")
        return None

