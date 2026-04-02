
import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Research Paper Quality Prediction",
    page_icon="📄",
    layout="wide"
)

MODEL_PATH = "models/binary_hybrid_logistic.pkl"

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1.05rem;
    color: #555;
    margin-bottom: 1.2rem;
}
.card {
    background: #f7f8fb;
    border: 1px solid #e5e7eb;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_model_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_embedder(name):
    return SentenceTransformer(name)

# -----------------------------
# PDF helpers
# -----------------------------
def safe_read_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        page_count = len(reader.pages)
        text_parts = []
        for page in reader.pages:
            try:
                txt = page.extract_text()
                if txt:
                    text_parts.append(txt)
            except Exception:
                continue
        return "\\n".join(text_parts), page_count
    except Exception:
        return "", 0

def extract_title_and_abstract(text):
    if not text.strip():
        return "", ""

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    title = lines[0][:250] if lines else ""

    abstract = ""
    m = re.search(r"abstract\\s*(.*?)(introduction|keywords|\\n\\n)", text, re.I | re.S)
    if m:
        abstract = m.group(1).strip()
    else:
        abstract = text[:1500].strip()

    return title, abstract[:3000]

# -----------------------------
# Feature helpers
# -----------------------------
def word_count(text):
    return len(re.findall(r"\\b\\w+\\b", text))

def sentence_count(text):
    return len([x for x in re.split(r"[.!?]+", text) if x.strip()])

def unique_word_count(text):
    return len(set(re.findall(r"\\b\\w+\\b", text.lower())))

def avg_word_length(text):
    words = re.findall(r"\\b\\w+\\b", text)
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))

def vocabulary_richness(text):
    wc = word_count(text)
    uwc = unique_word_count(text)
    return float(uwc / wc) if wc else 0.0

def lexical_density(text):
    words = re.findall(r"\\b\\w+\\b", text.lower())
    if not words:
        return 0.0
    stopwords = {
        "the","a","an","and","or","but","if","while","with","to","of","in","on",
        "for","at","by","is","are","was","were","be","been","being","this","that"
    }
    content = [w for w in words if w not in stopwords]
    return float(len(content) / len(words))

def keyword_count(text, keywords):
    t = text.lower()
    total = 0
    for kw in keywords:
        total += len(re.findall(rf"\\b{re.escape(kw.lower())}\\b", t))
    return total

def presence(text, keywords):
    return 1.0 if keyword_count(text, keywords) > 0 else 0.0

def build_features(full_text, title, abstract, page_count, citation_count):
    combined = f"{title} {abstract} {full_text}".strip()
    return {
        "page_count": float(page_count),
        "abstract_present": 1.0 if abstract.strip() else 0.0,
        "word_count": float(word_count(full_text)),
        "unique_word_count": float(unique_word_count(full_text)),
        "vocabulary_richness": float(vocabulary_richness(full_text)),
        "avg_word_length": float(avg_word_length(full_text)),
        "lexical_density": float(lexical_density(full_text)),
        "sentence_count": float(sentence_count(full_text)),
        "introduction_present": presence(full_text, ["introduction"]),
        "literature_review_present": presence(full_text, ["literature review", "related work"]),
        "methodology_present": presence(full_text, ["methodology", "methods"]),
        "results_present": presence(full_text, ["results"]),
        "discussion_present": presence(full_text, ["discussion"]),
        "conclusion_present": presence(full_text, ["conclusion", "conclusions"]),
        "references_present": presence(full_text, ["references", "bibliography"]),
        "section_count": float(sum([
            presence(full_text, ["introduction"]),
            presence(full_text, ["related work", "literature review"]),
            presence(full_text, ["methodology", "methods"]),
            presence(full_text, ["results"]),
            presence(full_text, ["discussion"]),
            presence(full_text, ["conclusion", "conclusions"]),
            presence(full_text, ["references", "bibliography"])
        ])),
        "experiment_mentions": float(keyword_count(combined, ["experiment", "experiments"])),
        "dataset_mentions": float(keyword_count(combined, ["dataset", "datasets"])),
        "evaluation_mentions": float(keyword_count(combined, ["evaluation", "evaluate"])),
        "validation_mentions": float(keyword_count(combined, ["validation", "validated"])),
        "benchmark_mentions": float(keyword_count(combined, ["benchmark", "benchmarks"])),
        "statistical_terms_count": float(keyword_count(combined, ["statistical", "significant", "regression", "variance"])),
        "p_value_mentions": float(keyword_count(combined, ["p-value", "p value", "p<", "p >"])),
        "confidence_interval_mentions": float(keyword_count(combined, ["confidence interval"])),
        "ablation_mentions": float(keyword_count(combined, ["ablation"])),
        "baseline_mentions": float(keyword_count(combined, ["baseline", "baselines"])),
        "reproducibility_terms_count": float(keyword_count(combined, ["reproducibility", "reproducible"])),
        "theorem_count": float(keyword_count(combined, ["theorem", "theorems"])),
        "lemma_count": float(keyword_count(combined, ["lemma", "lemmas"])),
        "proof_count": float(keyword_count(combined, ["proof", "proofs"])),
        "proposition_count": float(keyword_count(combined, ["proposition", "propositions"])),
        "corollary_count": float(keyword_count(combined, ["corollary", "corollaries"])),
        "algorithm_count": float(keyword_count(combined, ["algorithm", "algorithms"])),
        "pseudocode_present": presence(combined, ["pseudocode"]),
        "complexity_mentions": float(keyword_count(combined, ["complexity", "time complexity"])),
        "formal_definition_count": float(keyword_count(combined, ["definition", "definitions"])),
        "formula_density": 0.0,
        "novelty_keywords_count": float(keyword_count(combined, ["novel", "new", "proposed"])),
        "research_gap_mentions": float(keyword_count(combined, ["research gap", "gap in the literature"])),
        "new_method_mentions": float(keyword_count(combined, ["new method", "proposed method", "novel approach"])),
        "limitation_discussion_presence": presence(combined, ["limitation", "limitations"]),
        "future_work_mentions": float(keyword_count(combined, ["future work"])),
        "contribution_mentions": float(keyword_count(combined, ["contribution", "contributions"])),
        "sample_size_mentions": float(keyword_count(combined, ["sample size"])),
        "survey_mentions": float(keyword_count(combined, ["survey"])),
        "interview_mentions": float(keyword_count(combined, ["interview", "interviews"])),
        "case_study_mentions": float(keyword_count(combined, ["case study", "case studies"])),
        "fieldwork_mentions": float(keyword_count(combined, ["fieldwork"])),
        "real_world_mentions": float(keyword_count(combined, ["real world", "real-world"])),
        "github_link_present": 1.0 if "github.com" in combined.lower() else 0.0,
        "code_link_present": 1.0 if ("github.com" in combined.lower() or "code available" in combined.lower()) else 0.0,
        "Institution UKPRN code": 0.0,
        "Unit of assessment number": 11.0,
        "Year": 0.0,
        "Citation count": float(citation_count),
        "pdf_found": 1.0
    }

def prepare_input(model_bundle, title, abstract, numeric_inputs, categorical_inputs):
    embedder = load_embedder(model_bundle.get("embedder_name", "all-MiniLM-L6-v2"))
    combined_text = f"{title} {abstract}".strip()
    text_emb = embedder.encode([combined_text], convert_to_numpy=True, normalize_embeddings=True)

    numeric_features = model_bundle.get("numeric_features", [])
    categorical_features = model_bundle.get("categorical_features", [])

    if numeric_features:
        num_row = pd.DataFrame([{f: numeric_inputs.get(f, np.nan) for f in numeric_features}])
        num_imputer = model_bundle.get("num_imputer")
        scaler = model_bundle.get("scaler")
        num_arr = num_imputer.transform(num_row) if num_imputer is not None else num_row.values
        num_arr = scaler.transform(num_arr) if scaler is not None else num_arr
    else:
        num_arr = np.zeros((1, 0))

    if categorical_features:
        cat_row = pd.DataFrame([{f: categorical_inputs.get(f, "") for f in categorical_features}])
        cat_imputer = model_bundle.get("cat_imputer")
        ohe = model_bundle.get("ohe")
        cat_raw = cat_imputer.transform(cat_row) if cat_imputer is not None else cat_row.values
        cat_arr = ohe.transform(cat_raw) if ohe is not None else np.zeros((1, 0))
    else:
        cat_arr = np.zeros((1, 0))

    return np.hstack([text_emb, cat_arr, num_arr])

# -----------------------------
# App layout
# -----------------------------
st.markdown('<div class="main-title">Research Paper Quality Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict whether an individual paper is likely to be 4★ or not 4★ using textual, structural, and metadata-based features.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["Home", "Predict", "About"])

with tab1:
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Project Overview")
        st.write(
            "This web application is part of an MSc project on machine learning-based "
            "research paper quality prediction. It analyses paper text, PDF-derived features, "
            "and metadata to estimate the likely paper quality outcome."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("How it works")
        st.write(
            "1. Upload a research paper PDF\\n"
            "2. Extract title, abstract, and structural indicators\\n"
            "3. Add metadata such as citation count and publisher\\n"
            "4. Run the trained model\\n"
            "5. View the predicted result and confidence"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Current Model")
        st.write("Binary classification: **4★ vs Not 4★**")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    uploaded_pdf = st.file_uploader("Upload research paper PDF", type=["pdf"])

    extracted_text = ""
    page_count = 0
    title_default = ""
    abstract_default = ""

    if uploaded_pdf is not None:
        extracted_text, page_count = safe_read_pdf(uploaded_pdf)
        title_default, abstract_default = extract_title_and_abstract(extracted_text)
        st.success("PDF uploaded and processed successfully.")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Paper Title", value=title_default)
        abstract = st.text_area("Abstract / Summary", value=abstract_default, height=220)

    with col2:
        citation_count = st.number_input("Citation Count", min_value=0, value=0, step=1)
        publisher = st.text_input("Publisher")
        institution_name = st.text_input("Institution Name")
        main_panel = st.text_input("Main Panel", value="B")
        uoa_name = st.text_input("Unit of Assessment Name", value="Computer Science and Informatics")
        open_access_status = st.selectbox(
            "Open Access Status",
            ["Unknown", "Open", "Closed", "Hybrid", "Gold", "Green"]
        )

    if st.button("Predict", type="primary"):
        try:
            model_bundle = load_model_bundle()

            numeric_inputs = build_features(
                full_text=extracted_text,
                title=title,
                abstract=abstract,
                page_count=page_count,
                citation_count=citation_count
            )

            categorical_inputs = {
                "Institution name": institution_name,
                "Main panel": main_panel,
                "Unit of assessment name": uoa_name,
                "Publisher": publisher,
                "Open access status": open_access_status
            }

            X = prepare_input(model_bundle, title, abstract, numeric_inputs, categorical_inputs)

            clf = model_bundle["classifier"]
            pred = clf.predict(X)[0]
            prob = clf.predict_proba(X)[0]
            classes = list(clf.classes_)

            confidence = float(np.max(prob))
            pred_label = "4★" if int(pred) == 1 else "Not 4★"

            st.subheader("Prediction Result")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Predicted Label", pred_label)
            with c2:
                st.metric("Confidence", f"{confidence:.2%}")

            prob_df = pd.DataFrame({
                "Class": ["Not 4★" if int(c) == 0 else "4★" for c in classes],
                "Probability": prob
            })

            st.subheader("Class Probabilities")
            st.dataframe(prob_df, use_container_width=True)
            st.bar_chart(prob_df.set_index("Class"))

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About this application")
    st.write(
        "This prototype demonstrates deployment of the research model developed in the MSc project. "
        "It is designed to support paper-level prediction using uploaded PDFs and associated metadata."
    )
    st.markdown('</div>', unsafe_allow_html=True)
