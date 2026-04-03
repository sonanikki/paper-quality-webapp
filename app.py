import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st


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
# RESOURCES
# =========================================================
@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_embedder(embedder_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(embedder_name)


# =========================================================
# PDF / TEXT HELPERS
# =========================================================
def extract_pdf_text_and_pages(uploaded_file):
    try:
        from io import BytesIO
        from pypdf import PdfReader

        file_bytes = uploaded_file.read()
        reader = PdfReader(BytesIO(file_bytes))
        page_count = len(reader.pages)

        pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)

        return "\n".join(pages).strip(), page_count

    except ModuleNotFoundError:
        st.error("PDF reading dependency is missing in the deployed environment.")
        return "", 0

    except Exception as e:
        st.warning(f"Could not fully read PDF text: {e}")
        return "", 0


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_float(value, default=0.0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def tokenize_words(text: str):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def split_sentences(text: str):
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "while", "of", "to", "in", "on",
    "for", "with", "at", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "as", "it", "its", "their", "our", "we", "they", "he", "she", "you",
    "i", "his", "her", "them", "than", "then", "which", "who", "whom", "what",
    "when", "where", "why", "how", "can", "could", "should", "would", "may",
    "might", "will", "shall", "do", "does", "did", "done", "have", "has", "had"
}

ACADEMIC_WORDS = {
    "analysis", "approach", "assessment", "concept", "data", "design", "development",
    "evaluation", "evidence", "experiment", "framework", "hypothesis", "implementation",
    "investigation", "method", "methodology", "model", "objective", "outcome",
    "performance", "process", "research", "result", "study", "system", "theory",
    "validation", "significance", "rigour", "contribution"
}

SECTION_PATTERNS = {
    "abstract_present": r"\babstract\b",
    "introduction_present": r"\bintroduction\b",
    "literature_review_present": r"(literature review|related work|background)",
    "methodology_present": r"(methodology|methods|materials and methods|approach)",
    "results_present": r"\bresults?\b",
    "discussion_present": r"\bdiscussion\b",
    "conclusion_present": r"(conclusion|conclusions|concluding remarks)",
    "references_present": r"(references|bibliography)"
}

COUNT_PATTERNS = {
    "experiment_mentions": r"\bexperiment(s)?\b",
    "dataset_mentions": r"\bdataset(s)?\b",
    "evaluation_mentions": r"\bevaluation\b",
    "validation_mentions": r"\bvalidation\b",
    "benchmark_mentions": r"\bbenchmark(s)?\b",
    "statistical_terms_count": r"\b(statistical|regression|anova|variance|significant)\b",
    "p_value_mentions": r"\bp\s*[<=>]\s*0\.\d+|\bp-value\b",
    "confidence_interval_mentions": r"\bconfidence interval(s)?\b|\bCI\b",
    "ablation_mentions": r"\bablation\b",
    "baseline_mentions": r"\bbaseline(s)?\b",
    "reproducibility_terms_count": r"\b(reproducibility|reproducible|replication|replicable)\b",
    "theorem_count": r"\btheorem(s)?\b",
    "lemma_count": r"\blemma(s)?\b",
    "proof_count": r"\bproof(s)?\b",
    "proposition_count": r"\bproposition(s)?\b",
    "corollary_count": r"\bcorollary\b",
    "algorithm_count": r"\balgorithm(s)?\b",
    "complexity_mentions": r"\bcomplexity\b|\bo\([n0-9log\+\-\*\/\^\s]+\)",
    "formal_definition_count": r"\bdefinition(s)?\b",
    "novelty_keywords_count": r"\bnovel|new|original|innovative|proposed\b",
    "research_gap_mentions": r"\b(gap in the literature|research gap|existing gap)\b",
    "new_method_mentions": r"\bproposed method|new method|novel method\b",
    "future_work_mentions": r"\bfuture work\b",
    "contribution_mentions": r"\bcontribution(s)?\b",
    "sample_size_mentions": r"\bsample size\b|\bn\s*=\s*\d+\b",
    "survey_mentions": r"\bsurvey(s)?\b",
    "interview_mentions": r"\binterview(s)?\b",
    "case_study_mentions": r"\bcase study|case studies\b",
    "fieldwork_mentions": r"\bfieldwork\b",
    "real_world_mentions": r"\breal[- ]world\b"
}


def count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def syllable_count(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(syllable_count(w) for w in words)
    return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))


def detect_section_count(text: str) -> int:
    count = 0
    for _, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            count += 1
    return count


def binary_present(pattern: str, text: str) -> int:
    return int(bool(re.search(pattern, text, flags=re.IGNORECASE)))


def passive_voice_ratio(text: str) -> float:
    matches = re.findall(r"\b(is|are|was|were|been|be|being)\s+\w+ed\b", text.lower())
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    return len(matches) / len(sentences)


def punctuation_density(text: str) -> float:
    if not text:
        return 0.0
    punct = len(re.findall(r"[,\.;:!?()\[\]{}\-]", text))
    return punct / max(len(text), 1)


def github_link_present(text: str) -> int:
    return int(bool(re.search(r"github\.com", text, flags=re.IGNORECASE)))


def code_link_present(text: str) -> int:
    return int(bool(re.search(r"(github\.com|gitlab\.com|bitbucket\.org|code available|source code)", text, flags=re.IGNORECASE)))


def pseudocode_present(text: str) -> int:
    return int(bool(re.search(r"\bpseudocode\b", text, flags=re.IGNORECASE)))


def formula_density(text: str) -> float:
    formulas = len(re.findall(r"[=+\-/*^<>≤≥∑∫λμσπ]", text))
    words = len(tokenize_words(text))
    return formulas / max(words, 1)


def academic_word_frequency(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    academic = sum(1 for w in words if w in ACADEMIC_WORDS)
    return academic / len(words)


def lexical_density(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    content_words = [w for w in words if w not in STOPWORDS]
    return len(content_words) / len(words)


def vocabulary_richness(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def build_engineered_features(raw_text: str, page_count: int, title: str):
    text = clean_text(raw_text)
    words = tokenize_words(text)
    sentences = split_sentences(text)

    word_count = len(words)
    unique_word_count = len(set(words))
    sentence_count = len(sentences)

    avg_sentence_length = word_count / sentence_count if sentence_count else 0.0
    max_sentence_length = max((len(tokenize_words(s)) for s in sentences), default=0)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0

    features = {
        "paper_id": "USER_INPUT",
        "Title": title,
        "Institution UKPRN code": 0,
        "Year": 0,
        "pdf_found": int(page_count > 0),
        "page_count": page_count,
        "abstract_text": text[:2000],
        "abstract_present": binary_present(r"\babstract\b", text),
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "vocabulary_richness": vocabulary_richness(text),
        "avg_word_length": avg_word_length,
        "lexical_density": lexical_density(text),
        "academic_word_frequency": academic_word_frequency(text),
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "max_sentence_length": max_sentence_length,
        "readability_score": flesch_reading_ease(text),
        "passive_voice_ratio": passive_voice_ratio(text),
        "punctuation_density": punctuation_density(text),
        "section_count": detect_section_count(text),
        "pseudocode_present": pseudocode_present(text),
        "formula_density": formula_density(text),
        "github_link_present": github_link_present(text),
        "code_link_present": code_link_present(text),
    }

    for feature_name, pattern in SECTION_PATTERNS.items():
        features[feature_name] = binary_present(pattern, text)

    for feature_name, pattern in COUNT_PATTERNS.items():
        features[feature_name] = count_matches(pattern, text)

    features["limitation_discussion_presence"] = binary_present(r"\blimitation(s)?\b", text)

    return features


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
    engineered_features: dict,
    bundle: dict,
):
    row = dict(engineered_features)

    row["Title"] = title
    row["combined_text"] = combined_text
    row["Citation count"] = safe_float(citation_count, 0)
    row["Publisher"] = publisher
    row["Institution name"] = institution_name
    row["Main panel"] = main_panel
    row["Unit of assessment name"] = uoa_name
    row["Open access status"] = open_access_status

    numeric_features = bundle.get("numeric_features", [])
    categorical_features = bundle.get("categorical_features", [])

    for col in numeric_features:
        if col not in row:
            row[col] = 0.0

    for col in categorical_features:
        if col not in row:
            row[col] = "Unknown"

    if "combined_text" not in row:
        row["combined_text"] = combined_text

    return pd.DataFrame([row])


def build_feature_matrix(df_input: pd.DataFrame, bundle: dict):
    embedder_name = bundle.get("embedder_name", "all-MiniLM-L6-v2")
    text_feature = bundle.get("text_feature", "combined_text")
    numeric_features = bundle.get("numeric_features", [])
    categorical_features = bundle.get("categorical_features", [])

    scaler = bundle.get("scaler", None)
    classifier = bundle.get("classifier", None)
    ohe = bundle.get("ohe", None)

    if classifier is None:
        raise ValueError("Classifier not found in model bundle.")

    # TEXT
    embedder = load_embedder(embedder_name)
    text_values = df_input[text_feature].fillna("").astype(str).tolist()
    X_text = embedder.encode(text_values, convert_to_numpy=True)

    # NUMERIC
    X_num = np.empty((len(df_input), 0))
    if numeric_features:
        X_num_df = df_input.reindex(columns=numeric_features, fill_value=0).copy()
        X_num_df = X_num_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if scaler is not None:
            X_num = scaler.transform(X_num_df)
        else:
            X_num = X_num_df.to_numpy(dtype=float)

    # CATEGORICAL
    X_cat = np.empty((len(df_input), 0))
    if categorical_features:
        X_cat_df = df_input.reindex(columns=categorical_features, fill_value="Unknown").copy()
        X_cat_df = X_cat_df.fillna("Unknown").astype(str)

        if ohe is not None:
            X_cat = ohe.transform(X_cat_df)
            if hasattr(X_cat, "toarray"):
                X_cat = X_cat.toarray()
        else:
            X_cat = np.empty((len(df_input), 0))

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
    engineered_features: dict,
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
        engineered_features=engineered_features,
        bundle=bundle,
    )

    X_final, classifier = build_feature_matrix(df_input, bundle)
    pred = classifier.predict(X_final)[0]

    confidence = None
    if hasattr(classifier, "predict_proba"):
        prob = classifier.predict_proba(X_final)[0]
        confidence = float(np.max(prob))

    return int(pred), confidence, combined_text, df_input


# =========================================================
# UI HEADER
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
            using paper text, structural indicators, and metadata such as citation count, publisher,
            institution, main panel, unit of assessment, and open access status.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("How it works")
        st.markdown(
            """
            1. Upload a paper PDF or paste paper text  
            2. The system extracts text and derives document-level features  
            3. Metadata is combined with these extracted features  
            4. The trained model predicts whether the paper is likely 4★ or not 4★
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Current Model")
        st.write("**Binary classification:** 4★ vs Not 4★")
        st.write("**Focus:** UOA 11 — Computer Science and Informatics")
        st.write("**Inference mode:** PDF/text + metadata + engineered document features")
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
    detected_page_count = 0

    if uploaded_pdf is not None:
        with st.spinner("Reading PDF..."):
            pdf_text, detected_page_count = extract_pdf_text_and_pages(uploaded_pdf)

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
                engineered_features = build_engineered_features(
                    raw_text=text_for_prediction,
                    page_count=detected_page_count,
                    title=title
                )

                with st.spinner("Running prediction..."):
                    pred, confidence, combined_text, debug_df = predict_paper(
                        title=title,
                        abstract_or_text=text_for_prediction,
                        citation_count=citation_count,
                        publisher=publisher,
                        institution_name=institution_name,
                        main_panel=main_panel,
                        uoa_name=uoa_name,
                        open_access_status=open_access_status,
                        engineered_features=engineered_features,
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

                with st.expander("Show engineered features used for inference"):
                    st.dataframe(debug_df.T)

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
        category of individual research papers. The project combines metadata, textual content, and
        engineered document features within a machine learning pipeline to support paper-level evaluation.
        """
    )
    st.write("**Current target:** 4★ vs Not 4★")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Future Extensions")
    st.markdown(
        """
        - richer metadata integration  
        - persistent submission storage  
        - similarity analysis and plagiarism-related checks  
        - dashboard integration for paper-level analytics
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
