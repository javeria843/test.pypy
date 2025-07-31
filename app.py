import os
import csv
import re
import fitz  # PyMuPDF
import spacy
import nltk
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# ========== Initial Setup ==========

nltk.download("punkt")
nltk.download("stopwords")

# ✅ Load spaCy model with fallback install
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm", "--user"])
    nlp_spacy = spacy.load("en_core_web_sm")

# ✅ SentenceTransformer model
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-8b")

# ========== Helper Functions ==========

def extract_text(file, filetype):
    try:
        if filetype == "pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return " ".join([page.get_text() for page in doc])
        else:
            return file.read().decode("utf-8")
    except Exception as e:
        return f"❌ Error extracting text: {e}"

def clean(text):
    return re.sub(r"\s+", " ", text.strip())

def preprocess(text):
    doc = nlp_spacy(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

def embed(text):
    return model_embed.encode(text, convert_to_tensor=True)

def get_score(resume_emb, jd_emb):
    return float(util.pytorch_cos_sim(resume_emb, jd_emb)[0][0])

def ask_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") else "🤖 No response received"
    except Exception as e:
        return f"❌ Gemini Error: {e}"

def save_history(resume, jd, score, skills, roles):
    try:
        row = [resume[:50], jd[:50], round(score, 2), skills, roles]
        with open("history.csv", "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)
    except:
        pass

# ========== Streamlit UI ==========

st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("💼 AI Resume Matcher & Job Advisor")

col1, col2 = st.columns(2)
with col1:
    res_file = st.file_uploader("📄 Upload Resume", type=["pdf", "txt"])
with col2:
    jd_file = st.file_uploader("📝 Upload Job Description", type=["pdf", "txt"])

if st.button("🔍 Match & Advise") and res_file and jd_file:
    r_txt = clean(extract_text(res_file, res_file.name.split(".")[-1]))
    j_txt = clean(extract_text(jd_file, jd_file.name.split(".")[-1]))

    r_proc = preprocess(r_txt)
    j_proc = preprocess(j_txt)

    r_emb = embed(r_proc)
    j_emb = embed(j_proc)
    score = get_score(r_emb, j_emb)

    st.success(f"Similarity Score: **{score:.2f}**")

    skill_prompt = f"Given this job:\n{j_txt}\nand resume:\n{r_txt}\nSuggest 5 missing skills."
    role_prompt = f"Based on resume:\n{r_txt}\nSuggest 2-3 ideal job roles."

    skills = ask_gemini(skill_prompt)
    roles = ask_gemini(role_prompt)

    st.subheader("🔧 Skills to Improve")
    st.write(skills)

    st.subheader("🏷️ Recommended Roles")
    st.write(roles)

    save_history(r_txt, j_txt, score, skills, roles)

# ========== History Section ==========

st.markdown("---")
st.subheader("📊 Upload History")

if os.path.exists("history.csv"):
    df = pd.read_csv("history.csv", header=None)
    df.columns = ["Resume Snippet", "JD Snippet", "Similarity", "Skills", "Roles"]
    st.dataframe(df)
else:
    st.info("No history yet. Start analyzing to build one!")
