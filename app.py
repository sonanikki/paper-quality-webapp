import os
import re
import pickle
import importlib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Research Paper Quality Prediction System",
    page_icon="📄",
    layout="wide"
)

MODEL_PATH = "binary_hybrid_logistic.pkl"
LOOKUP_PATH = "metadata_lookup.csv"


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
# BASIC HELPERS
# =========================================================
def normalize_text(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_filename(name: str) -> str:
    name = os.path.basename(str(name or "").strip().lower())
    name = re.sub(r"_dup\d+(?=\.pdf$)", "", name)
    return name


def extract_paper_id(value: str) -> str:
    m = re.search(r"(P\d+)", str(value or ""), re.IGNORECASE)
    return m.group(1).upper() if m else ""


def safe_float(value, default=0.0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def split_sentences(text: str):
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


# =========================================================
# RUNTIME CHECK
# =========================================================
def check_runtime_dependencies():
    results = {}
    package_names = ["sklearn", "pypdf", "sentence_transformers", "torch"]

    for pkg in package_names:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "version unknown")
            results[pkg] = f"OK ({version})"
        except Exception as e:
            results[pkg] = f"ERROR: {e}"

    return results


# =========================================================
# LOAD RESOURCES
# =========================================================
@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except Exception as e:
        raise RuntimeError(f"Failed to load model bundle: {e}")


@st.cache_resource
def load_embedder(embedder_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(embedder_name)


@st.cache_data
def load_metadata_lookup():
    if not os.path.exists(LOOKUP_PATH):
        return pd.DataFrame()

    df = pd.read_csv(LOOKUP_PATH)

    for col in ["paper_id", "Title", "pdf_file"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df


# =========================================================
# LOOKUP MATCHING
# =========================================================
def find_metadata_match(title: str, uploaded_filename: str, lookup_df: pd.DataFrame):
    if lookup_df.empty:
        return None

    norm_title = normalize_text(title)
    norm_file = normalize_filename(uploaded_filename)
    paper_id = extract_paper_id(uploaded_filename)

    temp = lookup_df.copy()

    if "paper_id" in temp.columns:
        temp["_paper_id"] = temp["paper_id"].astype(str).str.upper().str.strip()
        pid_matches = temp[temp["_paper_id"] == paper_id]
        if len(pid_matches) > 0:
            return pid_matches.iloc[0].to_dict()

    if "pdf_file" in temp.columns:
        temp["_norm_pdf"] = temp["pdf_file"].apply(normalize_filename)
        file_matches = temp[temp["_norm_pdf"] == norm_file]
        if len(file_matches) > 0:
            return file_matches.iloc[0].to_dict()

    if norm_title and "Title" in temp.columns:
        temp["_norm_title"] = temp["Title"].apply(normalize_text)

        exact_title_matches = temp[temp["_norm_title"] == norm_title]
        if len(exact_title_matches) > 0:
            return exact_title_matches.iloc[0].to_dict()

        partial_title_matches = temp[temp["_norm_title"].str.contains(norm_title, na=False)]
        if len(partial_title_matches) > 0:
            return partial_title_matches.iloc[0].to_dict()

    return None


# =========================================================
# PDF HELPERS
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
            try:
                txt = page.extract_text()
                if txt:
                    pages.append(txt)
            except Exception:
                continue

        return "\n".join(pages).strip(), page_count

    except ModuleNotFoundError as e:
        st.error(f"PDF reading dependency is missing in the deployed environment: {e}")
        return "", 0
    except Exception as e:
        st.warning(f"Could not fully read PDF text: {e}")
        return "", 0


def extract_title_and_abstract(text: str):
    if not text or not text.strip():
        return "", ""

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    guessed_title = lines[0][:250] if lines else ""

    abstract = ""
    m = re.search(
        r"abstract\s*(.*?)(introduction|keywords|\n\n)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if m:
        abstract = m.group(1).strip()
    else:
        abstract = text[:3000].strip()

    return guessed_title, abstract[:3000]


# =========================================================
# FEATURE ENGINEERING
# =========================================================
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
    for pattern in SECTION_PATTERNS.values():
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
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0

    features = {
        "paper_id": "USER_INPUT",
        "Title": title,
        "pdf_found": int(page_count > 0),
        "page_count": page_count,
        "abstract_text": text[:3000],
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


# =========================================================
# MODEL PREP
# =========================================================
def make_combined_text(title: str, abstract_text: str) -> str:
    return f"{clean_text(title)} {clean_text(abstract_text)}".strip()


def prepare_single_input_dataframe(
    title: str,
    abstract_text: str,
    citation_count,
    publisher: str,
    institution_name: str,
    institution_ukprn_code,
    main_panel: str,
    uoa_name: str,
    open_access_status: str,
    year,
    engineered_features: dict,
    bundle: dict,
):
    row = dict(engineered_features)

    row["Title"] = title
    row["abstract_text"] = abstract_text
    row["combined_text"] = make_combined_text(title, abstract_text)
    row["Citation count"] = safe_float(citation_count, 0.0)
    row["Publisher"] = publisher
    row["Institution name"] = institution_name
    row["Institution UKPRN code"] = safe_int(institution_ukprn_code, 0)
    row["Main panel"] = main_panel
    row["Unit of assessment name"] = uoa_name
    row["Open access status"] = open_access_status
    row["Year"] = safe_int(year, 0)

    numeric_features = bundle.get("numeric_features", [])
    categorical_features = bundle.get("categorical_features", [])

    for col in numeric_features:
        if col not in row:
            row[col] = 0.0

    for col in categorical_features:
        if col not in row:
            row[col] = "Unknown"

    return pd.DataFrame([row])


def build_feature_matrix(df_input: pd.DataFrame, bundle: dict):
    embedder_name = bundle.get("embedder_name", "all-MiniLM-L6-v2")
    text_feature = bundle.get("text_feature", "combined_text")
    numeric_features = bundle.get("numeric_features", [])
    categorical_features = bundle.get("categorical_features", [])

    cat_imputer = bundle.get("cat_imputer", None)
    num_imputer = bundle.get("num_imputer", None)
    scaler = bundle.get("scaler", None)
    classifier = bundle.get("classifier", None)
    ohe = bundle.get("ohe", None)

    if classifier is None:
        raise ValueError("Classifier not found in model bundle.")

    embedder = load_embedder(embedder_name)
    text_values = df_input[text_feature].fillna("").astype(str).tolist()
    X_text = embedder.encode(
        text_values,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    X_cat = np.empty((len(df_input), 0))
    if categorical_features:
        X_cat_df = df_input.reindex(columns=categorical_features, fill_value="Unknown").copy()
        X_cat_df = X_cat_df.fillna("Unknown").astype(str)

        if cat_imputer is not None:
            X_cat_raw = cat_imputer.transform(X_cat_df)
        else:
            X_cat_raw = X_cat_df.values

        if ohe is not None:
            X_cat = ohe.transform(X_cat_raw)
            if hasattr(X_cat, "toarray"):
                X_cat = X_cat.toarray()

    X_num = np.empty((len(df_input), 0))
    if numeric_features:
        X_num_df = df_input.reindex(columns=numeric_features, fill_value=0).copy()
        X_num_df = X_num_df.apply(pd.to_numeric, errors="coerce")

        if num_imputer is not None:
            X_num_imputed = num_imputer.transform(X_num_df)
        else:
            X_num_imputed = X_num_df.fillna(0.0).to_numpy(dtype=float)

        if scaler is not None:
            X_num = scaler.transform(X_num_imputed)
        else:
            X_num = np.asarray(X_num_imputed, dtype=float)

    X_parts = [arr for arr in [X_text, X_cat, X_num] if arr.shape[1] > 0]
    if not X_parts:
        raise ValueError("No features were generated for prediction.")

    X_final = np.hstack(X_parts)

    debug_info = {
        "text_feature_name": text_feature,
        "embedder_name": embedder_name,
        "X_text_shape": X_text.shape,
        "X_cat_shape": X_cat.shape,
        "X_num_shape": X_num.shape,
        "X_final_shape": X_final.shape,
        "X_text_sum": float(np.sum(X_text)),
        "X_cat_sum": float(np.sum(X_cat)) if X_cat.size else 0.0,
        "X_num_sum": float(np.sum(X_num)) if X_num.size else 0.0,
        "X_final_sum": float(np.sum(X_final)),
        "X_final_mean": float(np.mean(X_final)),
        "X_final_std": float(np.std(X_final)),
    }

    return X_final, classifier, debug_info


def predict_paper(
    title: str,
    abstract_text: str,
    citation_count,
    publisher: str,
    institution_name: str,
    institution_ukprn_code,
    main_panel: str,
    uoa_name: str,
    open_access_status: str,
    year,
    engineered_features: dict,
):
    bundle = load_model_bundle()

    df_input = prepare_single_input_dataframe(
        title=title,
        abstract_text=abstract_text,
        citation_count=citation_count,
        publisher=publisher,
        institution_name=institution_name,
        institution_ukprn_code=institution_ukprn_code,
        main_panel=main_panel,
        uoa_name=uoa_name,
        open_access_status=open_access_status,
        year=year,
        engineered_features=engineered_features,
        bundle=bundle,
    )

    X_final, classifier, debug_info = build_feature_matrix(df_input, bundle)
    pred = classifier.predict(X_final)[0]

    confidence = None
    probabilities = None
    classifier_classes = None

    if hasattr(classifier, "classes_"):
        classifier_classes = classifier.classes_

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X_final)[0]
        confidence = float(np.max(probabilities))

    return int(pred), confidence, df_input, classifier_classes, probabilities, debug_info


# =========================================================
# UI
# =========================================================
st.markdown('<div class="main-title">Research Paper Quality Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict whether an individual paper is likely to be 4★ or not 4★ using textual, structural, and metadata-based inputs.</div>',
    unsafe_allow_html=True
)

tab_home, tab_predict, tab_about = st.tabs(["Home", "Predict", "About"])

with tab_home:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Project Overview")
    st.write(
        "This version uses metadata lookup for known dataset papers and manual metadata fallback for new papers."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with tab_predict:
    st.subheader("Enter Paper Details")

    lookup_df = load_metadata_lookup()

    left_col, right_col = st.columns(2)

    with left_col:
        title_input = st.text_input("Paper Title", "")
        uploaded_pdf = st.file_uploader("Upload Paper PDF (optional)", type=["pdf"])
        manual_text = st.text_area(
            "Abstract / Extracted Text / Key Paper Content",
            height=220,
            placeholder="Paste abstract or important paper text here..."
        )

    text_for_features = manual_text
    abstract_for_model = manual_text
    detected_page_count = 0
    final_title = title_input.strip()
    uploaded_filename = uploaded_pdf.name if uploaded_pdf is not None else ""

    if uploaded_pdf is not None:
        with st.spinner("Reading PDF..."):
            pdf_text, detected_page_count = extract_pdf_text_and_pages(uploaded_pdf)

        if pdf_text.strip():
            guessed_title, guessed_abstract = extract_title_and_abstract(pdf_text)

            if not final_title and guessed_title:
                final_title = guessed_title

            if guessed_abstract:
                abstract_for_model = guessed_abstract
                text_for_features = pdf_text
                st.success("PDF text extracted and abstract-style content will be used for prediction.")
            else:
                text_for_features = pdf_text
                abstract_for_model = manual_text if manual_text.strip() else pdf_text[:3000]
                st.warning("PDF abstract could not be isolated clearly. Fallback text will be used.")
        else:
            st.warning("PDF text could not be extracted. The manually entered text will be used instead.")

    matched_metadata = find_metadata_match(final_title, uploaded_filename, lookup_df)

    with right_col:
        st.markdown("### Metadata")

        default_citation = matched_metadata.get("Citation count", 0) if matched_metadata else 0
        default_publisher = matched_metadata.get("Publisher", "Unknown") if matched_metadata else "Unknown"
        default_institution = matched_metadata.get("Institution name", "Unknown") if matched_metadata else "Unknown"
        default_ukprn = matched_metadata.get("Institution UKPRN code", 0) if matched_metadata else 0
        default_main_panel = matched_metadata.get("Main panel", "B") if matched_metadata else "B"
        default_uoa = matched_metadata.get("Unit of assessment name", "Computer Science and Informatics") if matched_metadata else "Computer Science and Informatics"
        default_oa = matched_metadata.get("Open access status", "Unknown") if matched_metadata else "Unknown"
        default_year = matched_metadata.get("Year", 0) if matched_metadata else 0

        if matched_metadata:
            st.success("Known paper matched in metadata lookup. Real metadata has been loaded.")
        else:
            st.info("No lookup match found. Please enter metadata manually for best results.")

        citation_count = st.number_input(
            "Citation Count",
            min_value=0.0,
            value=float(safe_float(default_citation, 0)),
            step=1.0
        )
        publisher = st.text_input("Publisher", str(default_publisher))
        institution_name = st.text_input("Institution Name", str(default_institution))
        institution_ukprn_code = st.text_input("Institution UKPRN Code", str(default_ukprn))
        main_panel = st.text_input("Main Panel", str(default_main_panel))
        uoa_name = st.text_input("Unit of Assessment Name", str(default_uoa))
        year = st.number_input(
            "Year",
            min_value=0,
            max_value=2100,
            value=safe_int(default_year, 0),
            step=1
        )

        oa_options = [
            "Compliant",
            "Out of scope for open access requirements",
            "Not compliant",
            "Other exception",
            "Unknown"
        ]
        default_oa = str(default_oa) if str(default_oa) in oa_options else "Unknown"
        open_access_status = st.selectbox(
            "Open Access Status",
            oa_options,
            index=oa_options.index(default_oa)
        )

    if st.button("Predict"):
        if not final_title.strip():
            st.error("Please enter the paper title, or upload a PDF with a readable first line/title.")
        elif not abstract_for_model.strip():
            st.error("Please provide paper text or upload a readable PDF.")
        else:
            try:
                engineered_features = build_engineered_features(
                    raw_text=text_for_features,
                    page_count=detected_page_count,
                    title=final_title
                )

                with st.spinner("Running prediction..."):
                    pred, confidence, debug_df, classifier_classes, probabilities, debug_info = predict_paper(
                        title=final_title,
                        abstract_text=abstract_for_model,
                        citation_count=citation_count,
                        publisher=publisher,
                        institution_name=institution_name,
                        institution_ukprn_code=institution_ukprn_code,
                        main_panel=main_panel,
                        uoa_name=uoa_name,
                        open_access_status=open_access_status,
                        year=year,
                        engineered_features=engineered_features,
                    )

                st.markdown("---")
                st.subheader("Prediction Result")

                # Training target: is_4_star = (label == 4).astype(int)
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

                with st.expander("Show prediction debug info"):
                    st.write("Raw prediction:", pred)
                    st.write("Classifier classes:", classifier_classes)
                    st.write("Prediction probabilities:", probabilities)
                    st.json(debug_info)

                with st.expander("Show input row used for inference"):
                    st.dataframe(debug_df.T)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab_about:
    with st.expander("Runtime dependency check"):
        st.json(check_runtime_dependencies())
