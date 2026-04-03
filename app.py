import os
import re
import pickle
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Research Paper Quality Prediction System",
    page_icon="📄",
    layout="wide"
)

MODEL_PATH = "binary_hybrid_logistic.pkl"


# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.4rem;
    }
    .section-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background-color: #f8f9fb;
        border: 1px solid #e8e8e8;
        margin-bottom: 1rem;
    }
    .result-good {
        padding: 1rem;
        border-radius: 12px;
        background-color: #eaf7ee;
        border: 1px solid #b8e0c2;
        color: #145a32;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-warn {
        padding: 1rem;
        border-radius: 12px;
        background-color: #fff4e5;
        border: 1px solid #f5c27a;
        color: #7a4b00;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


@st.cache_resource
def load_embedder(embedder_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(embedder_name)


def extract_pdf_text(uploaded_file) -> str:
    try:
        file_bytes = uploaded_file.read()
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages).strip()
    except Exception as e:
        st.warning(f"Could not fully read PDF text: {e}")
        return ""


def safe_float(value, default=0.0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_combined_text(title: str, abstract_or_text: str) -> str:
    title = clean_text(title)
    abstract_or_text = clean_text(abstract_or_text)
    return f"{title}. {abstract_or_text}".strip()


def prepare_single_input_dataframe(
    title: str,
    combined_text: str,
    citation_count,
    publisher: str,
    institution_name: str,
    main_panel: str,
    uoa_name: str,
    open_access_status: str,
):
    row = {
        "Title": title,
        "combined_text": combined_text,
        "Citation count": safe_float(citation_count, 0),
        "Publisher": publisher,
        "Institution name": institution_name,
        "Main panel": main_panel,
        "Unit of assessment name": uoa_name,
        "Open access status": open_access_status,
    }
    return pd.DataFrame([row])


def build_feature_matrix(df_input: pd.DataFrame, bundle: dict):
    embedder_name = bundle.get("embedder_name", "all-MiniLM-L6-v2")
    text_feature = bundle.get("text_feature", "combined_text")
    numeric_features = bundle.get("numeric_features", [])
    categorical_features = bundle.get("categorical_features", [])

    num_imputer = bundle.get("num_imputer", None)
    scaler = bundle.get("scaler", None)
    classifier = bundle.get("classifier", None)
    cat_imputer = bundle.get("cat_imputer", None)
    ohe = bundle.get("ohe", None)

    if classifier is None:
        raise ValueError("Classifier not found in model bundle.")

    # text embeddings
    embedder = load_embedder(embedder_name)
    text_values = df_input[text_feature].fillna("").astype(str).tolist()
    X_text = embedder.encode(text_values, convert_to_numpy=True)

    # numeric features
    X_num = np.empty((len(df_input), 0))
    if numeric_features:
        X_num_df = df_input[numeric_features].copy()
        if num_imputer is not None:
            X_num_df = pd.DataFrame(
                num_imputer.transform(X_num_df),
                columns=numeric_features
            )
        if scaler is not None:
            X_num = scaler.transform(X_num_df)
        else:
            X_num = X_num_df.to_numpy(dtype=float)

    # categorical features
    X_cat = np.empty((len(df_input), 0))
    if categorical_features:
        X_cat_df = df_input[categorical_features].copy()

        if cat_imputer is not None:
            X_cat_df = pd.DataFrame(
                cat_imputer.transform(X_cat_df),
                columns=categorical_features
            )
        else:
            X_cat_df = X_cat_df.fillna("Unknown")

        if ohe is not None:
            X_cat = ohe.transform(X_cat_df)
            if hasattr(X_cat, "toarray"):
                X_cat = X_cat.toarray()
        else:
            X_cat = np.empty((len(df_input), 0))

    # combine all
    X_parts = [arr for arr in [X_text, X_num, X_cat] if arr.shape[1] > 0]
    X_final = np.hstack(X_parts)

    return X_final, classifier


def predict_paper(
    title: str,
    abstract_or_text: str,
    citation_count,
    publisher: str,
    institution_name: str,
    main_panel: str,
    uoa_name: str,
    open_access_status: str,
):
    bundle = load_model_bundle()

    combined_text = make_combined_text(title, abstract_or_text)

    df_input = prepare_single_input_dataframe(
        title=title,
        combined_text=combined_text,
        citation_count=citation_count,
        publisher=publisher,
        institution_name=institution_name,
        main_panel=main_panel,
        uoa_name=uoa_name,
        open_access_status=open_access_status,
    )

    X_final, classifier = build_feature_matrix(df_input, bundle)

    pred = classifier.predict(X_final)[0]

    if hasattr(classifier, "predict_proba"):
        prob = classifier.predict_proba(X_final)[0]
        confidence = float(np.max(prob))
    else:
        confidence = None

    return int(pred), confidence, combined_text


# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">Research Paper Quality Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict whether an individual paper is likely to be 4★ or not 4★ using textual, structural, and metadata-based inputs.</div>',
    unsafe_allow_html=True
)

tab_home, tab_predict, tab_about = st.tabs(["Home", "Predict", "About"])


# =========================================================
# HOME TAB
# =========================================================
with tab_home:
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Project Overview")
        st.write(
            """
            This web application is part of an MSc project on machine learning-based research paper
            quality prediction. It supports binary classification of papers as **4★** or **Not 4★**
            using paper text and metadata such as citation count, publisher, institution, main panel,
            unit of assessment, and open access status.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("How it works")
        st.markdown(
            """
            1. Enter a paper title and either paste text or upload a PDF  
            2. Provide metadata used by the model  
            3. Run prediction  
            4. View the predicted class and confidence
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Current Model")
        st.write("**Binary classification:** 4★ vs Not 4★")
        st.write("**Focus:** UOA 11 — Computer Science and Informatics")
        st.write("**Deployment note:** This is a demonstration system for project presentation and evaluation.")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PREDICT TAB
# =========================================================
with tab_predict:
    st.subheader("Enter Paper Details")

    left_col, right_col = st.columns(2)

    with left_col:
        title = st.text_input("Paper Title", "")
        uploaded_pdf = st.file_uploader("Upload Paper PDF (optional)", type=["pdf"])
        manual_text = st.text_area(
            "Abstract / Extracted Text / Key Paper Content",
            height=220,
            placeholder="Paste abstract or important paper text here..."
        )

    with right_col:
        citation_count = st.number_input("Citation Count", min_value=0.0, value=0.0, step=1.0)
        publisher = st.text_input("Publisher", "Unknown")
        institution_name = st.text_input("Institution Name", "Unknown")
        main_panel = st.text_input("Main Panel", "B")
        uoa_name = st.text_input("Unit of Assessment Name", "Computer Science and Informatics")

        open_access_status = st.selectbox(
            "Open Access Status",
            [
                "Compliant",
                "Out of scope for open access requirements",
                "Not compliant",
                "Other exception",
                "Unknown"
            ],
            index=0
        )

    text_for_prediction = manual_text

    if uploaded_pdf is not None:
        with st.spinner("Reading PDF..."):
            pdf_text = extract_pdf_text(uploaded_pdf)
        if pdf_text.strip():
            text_for_prediction = pdf_text
            st.success("PDF text extracted and will be used for prediction.")
        else:
            st.warning("PDF text could not be extracted. The manually entered text will be used instead.")

    predict_button = st.button("Predict")

    if predict_button:
        if not title.strip():
            st.error("Please enter the paper title.")
        elif not text_for_prediction.strip():
            st.error("Please provide paper text or upload a readable PDF.")
        else:
            try:
                with st.spinner("Running prediction..."):
                    pred, confidence, combined_text = predict_paper(
                        title=title,
                        abstract_or_text=text_for_prediction,
                        citation_count=citation_count,
                        publisher=publisher,
                        institution_name=institution_name,
                        main_panel=main_panel,
                        uoa_name=uoa_name,
                        open_access_status=open_access_status,
                    )

                st.markdown("---")
                st.subheader("Prediction Result")

                if pred == 1:
                    st.markdown(
                        '<div class="result-good">Predicted Class: 4★ Paper</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-warn">Predicted Class: Not 4★ Paper</div>',
                        unsafe_allow_html=True
                    )

                if confidence is not None:
                    st.write(f"**Confidence:** {confidence:.2%}")

                with st.expander("Show combined text used for prediction"):
                    st.write(combined_text[:5000])

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# =========================================================
# ABOUT TAB
# =========================================================
with tab_about:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("About this Project")
    st.write(
        """
        This application was developed as part of an MSc project focused on predicting the likely quality
        category of individual research papers. The project explores how textual, structural, and contextual
        metadata can be combined within a machine learning pipeline to support research paper evaluation.
        """
    )
    st.write(
        """
        The current deployed version uses a binary target:
        **4★ vs Not 4★**.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Future Extensions")
    st.markdown(
        """
        - richer metadata integration  
        - paper history storage  
        - similarity analysis and plagiarism-related checks  
        - dashboard integration for paper-level analytics
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
