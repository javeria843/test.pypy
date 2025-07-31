import os
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer, util

# ✅ Setup
nltk.download('punkt')
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Gemini API setup (if used)
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Safe spaCy loading
import spacy
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

# ✅ Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ✅ Clean and summarize text
def clean_text(text):
    doc = nlp_spacy(text)
    return " ".join([sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10])

# ✅ Gemini AI suggestion
def generate_feedback(text):
    try:
        model_g = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Give short improvement suggestions and career roles for this resume:\n\n{text}"
        res = model_g.generate_content(prompt)
        return res.text
    except Exception as e:
        return "Gemini API error or not configured."

# ✅ Main App
st.set_page_config(page_title="Resume Matcher AI", layout="centered")
st.title("📄 Resume Matcher AI with Suggestions")

resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF or Text)", type=["pdf", "txt"])

if st.button("Compare and Suggest") and resume_file and jd_file:
    # Read text
    resume_text = extract_text_from_pdf(resume_file)
    if jd_file.type == "application/pdf":
        jd_text = extract_text_from_pdf(jd_file)
    else:
        jd_text = jd_file.read().decode("utf-8")

    # Clean text
    resume_clean = clean_text(resume_text)
    jd_clean = clea_
