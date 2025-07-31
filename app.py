import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import nltk
import os
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import spacy

# ✅ Handle en_core_web_sm loading safely
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

# ✅ Setup Gemini API (make sure to set GEMINI_API_KEY in Secrets or env)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Download punkt tokenizer if not present
nltk.download('punkt')

# ✅ Sentence Transformer Model
model_embed = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Text extraction from PDF
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# ✅ Preprocessing
def preprocess_text(text):
    doc = nlp_spacy(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# ✅ Gemini Prompt for suggestions
def get_suggestions_from_gemini(resume_text, jd_text):
    prompt = f"""
    You are an expert HR system. Compare the resume and job description below. 
    1. List 3 missing important skills from the resume.
    2. Suggest 3 potential job roles to target.
    Resume: {resume_text}
    Job Description: {jd_text}
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# ✅ Similarity Calculation
def calculate_similarity(text1, text2):
    embedding1 = model_embed.encode(text1, convert_to_tensor=True)
    embedding2 = model_embed.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

# ✅ Streamlit UI
st.title("📄 AI Resume Matcher with Suggestions")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if st.button("Match and Analyze"):
    if resume_file and jd_file:
        with st.spinner("Extracting and analyzing..."):
            resume_text = extract_text_from_pdf(resume_file)
            jd_text = extract_text_from_pdf(jd_file)

            resume_clean = preprocess_text(resume_text)
            jd_clean = preprocess_text(jd_text)

            similarity = calculate_similarity(resume_clean, jd_clean)

            suggestions = get_suggestions_from_gemini(resume_text, jd_text)

            st.success("✅ Analysis Complete")
            st.write(f"**Similarity Score:** {similarity:.2f}")
            st.markdown("### 📌 Gemini Suggestions")
            st.markdown(suggestions)

            # ✅ Save history
            new_entry = pd.DataFrame([[resume_clean[:150], jd_clean[:150], round(similarity, 2), suggestions]], 
                                     columns=["Resume Snippet", "JD Snippet", "Similarity", "Suggestions"])
            if os.path.exists("history.csv"):
                history = pd.read_csv("history.csv")
                history = pd.concat([history, new_entry], ignore_index=True)
            else:
                history = new_entry
            history.to_csv("history.csv", index=False)

    else:
        st.warning("⚠️ Please upload both resume and job description.")

# ✅ History viewer
if st.checkbox("Show Matching History"):
    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        st.dataframe(df)
    else:
        st.info("No history found.")
