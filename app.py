import os
import streamlit as st
import fitz  # PyMuPDF
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import google.generativeai as genai

# ✅ Download NLTK punkt tokenizer
nltk.download('punkt')

# ✅ Load spaCy model safely
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

# ✅ Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Utility functions
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def preprocess_text(text):
    doc = nlp_spacy(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha])

def get_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

def calculate_similarity(resume_text, jd_text):
    resume_sentences = sent_tokenize(resume_text)
    jd_sentences = sent_tokenize(jd_text)
    resume_embeddings = get_embeddings(resume_sentences)
    jd_embeddings = get_embeddings(jd_sentences)
    similarity_matrix = util.pytorch_cos_sim(resume_embeddings, jd_embeddings)
    max_score = similarity_matrix.max().item()
    return round(max_score * 100, 2)

def gemini_response(prompt):
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Error from Gemini: " + str(e)

def display_history():
    if os.path.exists("history.csv"):
        try:
            df = pd.read_csv("history.csv", header=None)
            df.columns = ["Resume Snippet", "JD Snippet", "Similarity", "Skills", "Roles"]
            st.dataframe(df)
        except:
            st.warning("History file found but couldn't read.")

# ✅ Streamlit UI
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("📄 AI Resume & JD Matcher with Suggestions")

with st.sidebar:
    st.subheader("Upload Files")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    if st.button("Show History"):
        display_history()

if resume_file and jd_file:
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    st.subheader("📌 Resume Preview")
    st.text_area("Resume Text", resume_text[:1500], height=200)

    st.subheader("📌 Job Description Preview")
    st.text_area("JD Text", jd_text[:1500], height=200)

    if st.button("⚡ Match and Suggest"):
        with st.spinner("Calculating similarity and generating suggestions..."):
            clean_resume = preprocess_text(resume_text)
            clean_jd = preprocess_text(jd_text)
            similarity = calculate_similarity(clean_resume, clean_jd)

            prompt_skills = f"Given this resume: {resume_text} and this job description: {jd_text}, suggest top 5 missing or weak skills in the resume that should be improved."
            prompt_roles = f"Based on this resume: {resume_text}, suggest 3 ideal job roles or career paths that best match the candidate profile."

            skills = gemini_response(prompt_skills)
            roles = gemini_response(prompt_roles)

        st.success(f"✅ Similarity Score: {similarity}%")

        st.markdown("### 🔧 Skills to Improve")
        st.write(skills)

        st.markdown("### 🧭 Recommended Roles")
        st.write(roles)

        # ✅ Save to history
        data = [[resume_text[:300], jd_text[:300], similarity, skills, roles]]
        df = pd.DataFrame(data)
        df.to_csv("history.csv", mode='a', index=False, header=False)

