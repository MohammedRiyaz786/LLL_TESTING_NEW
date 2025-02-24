import os
import groq
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def initialize_apis():
    """Initialize Groq and Google Gemini APIs with environment variables."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    
    if groq_api_key:
        groq_client = groq.Groq(api_key=groq_api_key)
    else:
        st.error("Groq API key not found in environment variables!")
        groq_client = None
        
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    else:
        st.error("Google API key not found in environment variables!")
    
    return groq_client
