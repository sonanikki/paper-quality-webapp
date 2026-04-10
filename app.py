import os
import re
import json
import uuid
import pickle
import importlib
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Research Paper Quality Prediction System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "binary_hybrid_logistic.pkl"
LOOKUP_PATH = "metadata_lookup.csv"

APP_ASSISTANT_SYSTEM_PROMPT = """
You are a friendly in-app assistant for a Streamlit application called
'Research Paper Quality Prediction System'.

Your job is ONLY to help users with:
- how to use the app
- what the prediction workflow means
- why metadata is needed
- what 4★ vs Not 4★ means in this project
- what citation count / publisher / institution / open access fields are for
- basic demo or viva guidance
- how PDF upload and paper processing work

Do NOT mention internal datasets, backend lookup files, hidden matching rules,
or implementation details unless the user explicitly asks for technical details.

Do NOT answer unrelated general knowledge questions in depth.
If the user asks something outside the app's scope, politely redirect them
back to app-related help.

Keep answers clear, supportive, and concise.
"""


# =========================================================
# SESSION STATE
# =========================================================
if "helper_messages" not in st.session_state:
    st.session_state["helper_messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello — I’m your app guide. Ask me how to use the system, "
                "what metadata means, or how to present the app in your demo."
            ),
        }
    ]

if "assistant_previous_response_id" not in st.session_state:
    st.session_state["assistant_previous_response_id"] = None

if "assistant_session_id" not in st.session_state:
    st.session_state["assistant_session_id"] = str(uuid.uuid4())

if "use_gpt_helper" not in st.session_state:
    st.session_state["use_gpt_helper"] = True


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0f0d 0%, #101915 45%, #0c120f 100%);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero-box {
        background: linear-gradient(135deg, #050807 0%, #0c1511 45%, #123222 100%);
        color: white;
        padding: 2rem 2rem 1.6rem 2rem;
        border-radius: 24px;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.2rem;
        animation: fadeUp 0.7s ease;
        border: 1px solid rgba(74, 222, 128, 0.16);
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.35rem;
        color: #ecfdf5;
    }

    .hero-subtitle {
        font-size: 1.03rem;
        line-height: 1.65;
        opacity: 0.96;
        max-width: 980px;
        color: #d1fae5;
    }

    .section-card {
        background: rgba(16, 24, 20, 0.95);
        backdrop-filter: blur(6px);
        padding: 1.15rem 1.25rem;
        border-radius: 20px;
        border: 1px solid rgba(74, 222, 128, 0.16);
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
        animation: fadeUp 0.65s ease;
        color: #ecfdf5;
    }

    .mini-card {
        background: linear-gradient(180deg, #0f1713 0%, #13201a 100%);
        padding: 1rem 1rem 0.9rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(74, 222, 128, 0.16);
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.25);
        height: 100%;
        transition: all 0.25s ease;
        color: #ecfdf5;
    }

    .mini-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.34);
    }

    .stat-card {
        background: linear-gradient(180deg, #0d1411 0%, #15211b 100%);
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(74, 222, 128, 0.20);
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        text-align: center;
        animation: fadeUp 0.6s ease;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #4ade80;
        margin-bottom: 0.2rem;
    }

    .stat-label {
        font-size: 0.95rem;
        color: #bbf7d0;
        font-weight: 500;
    }

    .status-good {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: #052e16;
        color: #bbf7d0;
        border: 1px solid #22c55e;
        font-weight: 600;
        margin-bottom: 0.9rem;
        box-shadow: 0 6px 14px rgba(34, 197, 94, 0.14);
    }

    .status-info {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: #0f172a;
        color: #86efac;
        border: 1px solid #16a34a;
        font-weight: 600;
        margin-bottom: 0.9rem;
        box-shadow: 0 6px 14px rgba(22, 163, 74, 0.12);
    }

    .status-warn {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: #1c1917;
        color: #facc15;
        border: 1px solid #65a30d;
        font-weight: 600;
        margin-bottom: 0.9rem;
        box-shadow: 0 6px 14px rgba(101, 163, 13, 0.12);
    }

    .result-good {
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(180deg, #052e16 0%, #14532d 100%);
        border: 1px solid #22c55e;
        color: #dcfce7;
        font-size: 1.1rem;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(34, 197, 94, 0.16);
    }

    .result-warn {
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(180deg, #172018 0%, #243c25 100%);
        border: 1px solid #65a30d;
        color: #ecfccb;
        font-size: 1.1rem;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(101, 163, 13, 0.14);
    }

    .glass-note {
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: rgba(13, 20, 17, 0.88);
        border: 1px solid rgba(74, 222, 128, 0.14);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.24);
        color: #d1fae5;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .soft-heading {
        font-size: 1.08rem;
        font-weight: 700;
        color: #4ade80;
        margin-bottom: 0.55rem;
    }

    .small-muted {
        color: #bbf7d0;
        font-size: 0.94rem;
        line-height: 1.6;
    }

    .footer-note {
        text-align: center;
        color: #86efac;
        margin-top: 2rem;
        font-size: 0.9rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        border: none;
        font-weight: 700;
        color: #04110a;
        background: linear-gradient(135deg, #22c55e 0%, #4ade80 100%);
        box-shadow: 0 10px 22px rgba(34, 197, 94, 0.22);
        transition: all 0.25s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(34, 197, 94, 0.30);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030706 0%, #08110d 45%, #0c1a13 100%);
    }

    [data-testid="stSidebar"] * {
        color: #ecfdf5 !important;
    }

    [data-testid="stMetric"] {
        background: rgba(12, 18, 15, 0.95);
        border: 1px solid rgba(74, 222, 128, 0.14);
        padding: 0.7rem;
        border-radius: 16px;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background-color: #101915 !important;
        color: #ecfdf5 !important;
        border: 1px solid #1f7a46 !important;
        border-radius: 12px !important;
    }

    [data-baseweb="select"] > div {
        background-color: #101915 !important;
        color: #ecfdf5 !important;
        border: 1px solid #1f7a46 !important;
        border-radius: 12px !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: inherit;
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


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


def safe_text(value, default=""):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def split_sentences(text: str):
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def get_secret_value(key_name: str, default: str = "") -> str:
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, default)


def gpt_helper_ready() -> bool:
    api_key = get_secret_value("OPENAI_API_KEY", "")
    return OPENAI_AVAILABLE and bool(api_key)


# =========================================================
# RUNTIME CHECK
# =========================================================
def check_runtime_dependencies():
    results = {}
    package_names = ["sklearn", "pypdf", "sentence_transformers", "torch", "openai"]

    for pkg in package_names:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "version unknown")
            results[pkg] = f"OK ({version})"
        except Exception as e:
            results[pkg] = f"ERROR: {e}"

    results["gpt_helper_configured"] = "Yes" if gpt_helper_ready() else "No"
    return results


# =========================================================
# LOAD RESOURCES
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
 sentence_transformers import SentenceTransformer
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
# MATCHING
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
        re.IGNORECASE | re.DOTALL,
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
    "might", "will", "shall", "do", "does", "did", "done", "have", "has", "had",
}

ACADEMIC_WORDS = {
    "analysis", "approach", "assessment", "concept", "data", "design", "development",
    "evaluation", "evidence", "experiment", "framework", "hypothesis", "implementation",
    "investigation", "method", "methodology", "model", "objective", "outcome",
    "performance", "process", "research", "result", "study", "system", "theory",
    "validation", "significance", "rigour", "contribution",
}

SECTION_PATTERNS = {
    "abstract_present": r"\babstract\b",
    "introduction_present": r"\bintroduction\b",
    "literature_review_present": r"(literature review|related work|background)",
    "methodology_present": r"(methodology|methods|materials and methods|approach)",
    "results_present": r"\bresults?\b",
    "discussion_present": r"\bdiscussion\b",
    "conclusion_present": r"(conclusion|conclusions|concluding remarks)",
    "references_present": r"(references|bibliography)",
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
    "real_world_mentions": r"\breal[- ]world\b",
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
        normalize_embeddings=True,
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
# ASSISTANT UI
# =========================================================
def render_avatar_assistant(messages, title="Ava · App Guide", subtitle="Ask me about uploads, metadata, or demo tips."):
    messages_json = json.dumps(messages, ensure_ascii=False)

    html = f"""
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: Inter, Arial, sans-serif;
        }}

        .wrap {{
            display: flex;
            align-items: center;
            gap: 18px;
            background: linear-gradient(135deg, rgba(12,20,15,0.96), rgba(20,35,28,0.96));
            border: 1px solid rgba(74,222,128,0.16);
            border-radius: 24px;
            padding: 20px;
            box-shadow: 0 14px 28px rgba(0,0,0,0.22);
            overflow: hidden;
        }}

        .avatar-shell {{
            position: relative;
            width: 110px;
            height: 110px;
            flex-shrink: 0;
            animation: popIn 0.8s ease;
        }}

        .halo {{
            position: absolute;
            inset: 0;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(34,197,94,0.30), rgba(34,197,94,0.03) 68%);
            animation: pulse 2.2s infinite ease-in-out;
        }}

        .avatar {{
            position: absolute;
            inset: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg, #166534 0%, #22c55e 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 46px;
            color: white;
            box-shadow: 0 10px 22px rgba(34,197,94,0.22);
            animation: floaty 3s infinite ease-in-out;
        }}

        .bubble {{
            flex: 1;
            background: #0f1713;
            border-radius: 22px;
            padding: 16px 18px;
            border: 1px solid rgba(74,222,128,0.16);
            box-shadow: 0 10px 20px rgba(0,0,0,0.16);
            min-height: 110px;
        }}

        .name {{
            font-size: 16px;
            font-weight: 800;
            color: #ecfdf5;
            margin-bottom: 6px;
        }}

        .subtitle {{
            font-size: 13px;
            color: #bbf7d0;
            margin-bottom: 8px;
        }}

        .text {{
            font-size: 16px;
            color: #f0fdf4;
            line-height: 1.6;
            min-height: 48px;
            transition: opacity 0.35s ease;
        }}

        .dots {{
            margin-top: 8px;
            display: flex;
            gap: 6px;
        }}

        .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #86efac;
            animation: blink 1.4s infinite ease-in-out;
        }}

        .dot:nth-child(2) {{ animation-delay: 0.2s; }}
        .dot:nth-child(3) {{ animation-delay: 0.4s; }}

        @keyframes popIn {{
            from {{ transform: scale(0.75); opacity: 0; }}
            to {{ transform: scale(1); opacity: 1; }}
        }}

        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 0.8; }}
            50% {{ transform: scale(1.12); opacity: 0.35; }}
            100% {{ transform: scale(1); opacity: 0.8; }}
        }}

        @keyframes floaty {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-6px); }}
            100% {{ transform: translateY(0px); }}
        }}

        @keyframes blink {{
            0%, 80%, 100% {{ opacity: 0.3; transform: scale(0.9); }}
            40% {{ opacity: 1; transform: scale(1.1); }}
        }}
    </style>
    </head>
    <body>
        <div class="wrap">
            <div class="avatar-shell">
                <div class="halo"></div>
                <div class="avatar">👩‍💻</div>
            </div>
            <div class="bubble">
                <div class="name">{title}</div>
                <div class="subtitle">{subtitle}</div>
                <div class="text" id="assistantText"></div>
                <div class="dots">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        </div>

        <script>
            const messages = {messages_json};
            const el = document.getElementById("assistantText");
            let idx = 0;

            function showMessage() {{
                el.style.opacity = 0.15;
                setTimeout(() => {{
                    el.innerText = messages[idx];
                    el.style.opacity = 1;
                    idx = (idx + 1) % messages.length;
                }}, 180);
            }}

            showMessage();
            setInterval(showMessage, 3200);
        </script>
    </body>
    </html>
    """
    components.html(html, height=200)


# =========================================================
# ASSISTANT LOGIC
# =========================================================
def local_app_help(question: str) -> str:
    q = normalize_text(question)

    if any(x in q for x in ["how do i use", "how to use", "how can i use", "start", "begin", "upload"]):
        return (
            "Go to the Predict page, upload a PDF, review the extracted text, "
            "check the metadata fields, and then run the prediction."
        )

    if "metadata" in q or "citation" in q or "publisher" in q or "institution" in q or "open access" in q:
        return (
            "Metadata gives the hybrid model extra context beyond the paper text. "
            "In this project, fields like citation count, publisher, institution, main panel, "
            "UOA name, open access status, and year can support the prediction."
        )

    if "4★" in question or "4 star" in q or "not 4" in q:
        return (
            "This app performs binary classification. It predicts whether a paper is likely to be "
            "4★ or Not 4★ within the current project framing for UOA 11."
        )

    if "demo" in q or "viva" in q or "presentation" in q:
        return (
            "For the best demo, upload a paper, show the extracted information, explain the metadata role, "
            "and then run the model prediction."
        )

    if "pdf" in q or "abstract" in q or "text extraction" in q:
        return (
            "When you upload a PDF, the app tries to extract page text, guess the title, and isolate abstract-style content. "
            "If extraction is weak, the metadata fields can still help complete the prediction workflow."
        )

    return (
        "I can help with using the app, metadata meaning, prediction flow, and demo tips. "
        "Try asking something like: 'How do I use this app?' or 'Why is metadata needed?'"
    )


def ask_gpt_helper(question: str, current_page: str, model_exists: bool, lookup_exists: bool) -> str:
    api_key = get_secret_value("OPENAI_API_KEY", "")
    model_name = get_secret_value("OPENAI_MODEL", "gpt-4.1-mini")

    if not OPENAI_AVAILABLE:
        raise RuntimeError("The openai package is not installed.")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)

    app_context = f"""
Current page: {current_page}
Model file present: {'Yes' if model_exists else 'No'}
Metadata lookup available: {'Yes' if lookup_exists else 'No'}
Session ID: {st.session_state['assistant_session_id']}
"""

    kwargs = {
        "model": model_name,
        "instructions": APP_ASSISTANT_SYSTEM_PROMPT,
        "input": f"{app_context}\nUser question: {question}",
        "max_output_tokens": 260,
    }

    previous_response_id = st.session_state.get("assistant_previous_response_id")
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    response = client.responses.create(**kwargs)

    st.session_state["assistant_previous_response_id"] = getattr(response, "id", None)

    output_text = getattr(response, "output_text", "")
    if output_text and str(output_text).strip():
        return str(output_text).strip()

    return "I could not generate a reply just now. Please try again."


def respond_from_assistant(question: str, current_page: str, model_exists: bool, lookup_exists: bool) -> str:
    if st.session_state.get("use_gpt_helper", True) and gpt_helper_ready():
        try:
            return ask_gpt_helper(question, current_page, model_exists, lookup_exists)
        except Exception as e:
            st.error(f"GPT helper runtime error: {e}")
            fallback = local_app_help(question)
            return (
                fallback
                + "\n\n_(The GPT helper was unavailable just now, so I answered using the built-in app guide.)_"
            )

    return local_app_help(question)


def reset_assistant_chat():
    st.session_state["helper_messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello — I’m your app guide. Ask me how to use the system, "
                "what metadata means, or how to present the app in your demo."
            ),
        }
    ]
    st.session_state["assistant_previous_response_id"] = None


def handle_assistant_prompt(prompt: str, current_page: str, model_exists: bool, lookup_exists: bool):
    prompt = (prompt or "").strip()
    if not prompt:
        return

    st.session_state["helper_messages"].append({"role": "user", "content": prompt})
    answer = respond_from_assistant(prompt, current_page, model_exists, lookup_exists)
    st.session_state["helper_messages"].append({"role": "assistant", "content": answer})


def render_help_chat(current_page: str, model_exists: bool, lookup_exists: bool):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Ask the App Assistant")

    mode_text = "GPT helper enabled" if gpt_helper_ready() and st.session_state["use_gpt_helper"] else "Built-in help mode"
    st.caption(f"Assistant mode: {mode_text}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.checkbox("Use GPT helper when available", key="use_gpt_helper")
    with c2:
        if st.button("Clear assistant chat", key=f"clear_chat_{current_page}"):
            reset_assistant_chat()
            st.rerun()

    st.write("Quick help prompts:")
    q1, q2, q3, q4 = st.columns(4)

    if q1.button("How do I use the app?", key=f"q1_{current_page}"):
        handle_assistant_prompt("How do I use this app?", current_page, model_exists, lookup_exists)
        st.rerun()

    if q2.button("Why is metadata needed?", key=f"q2_{current_page}"):
        handle_assistant_prompt("Why is metadata needed?", current_page, model_exists, lookup_exists)
        st.rerun()

    if q3.button("What is the prediction workflow?", key=f"q3_{current_page}"):
        handle_assistant_prompt("What is the prediction workflow?", current_page, model_exists, lookup_exists)
        st.rerun()

    if q4.button("How should I demo this?", key=f"q4_{current_page}"):
        handle_assistant_prompt("How should I demo this app in my viva?", current_page, model_exists, lookup_exists)
        st.rerun()

    for msg in st.session_state["helper_messages"]:
        role = msg.get("role", "assistant")
        if role not in ["user", "assistant"]:
            role = "assistant"
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

    with st.form(key=f"assistant_form_{current_page}", clear_on_submit=True):
        typed_prompt = st.text_input(
            "Type your question here",
            placeholder="Ask about the app, metadata, prediction flow, or demo tips...",
        )
        submitted = st.form_submit_button("Send")

    if submitted and typed_prompt.strip():
        handle_assistant_prompt(typed_prompt, current_page, model_exists, lookup_exists)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "About"])

st.sidebar.markdown("---")
st.sidebar.markdown("## System Status")

model_exists = os.path.exists(MODEL_PATH)
lookup_exists = os.path.exists(LOOKUP_PATH)

if model_exists:
    st.sidebar.success("Model file detected")
else:
    st.sidebar.error("Model file missing")

if lookup_exists:
    st.sidebar.success("Metadata service available")
else:
    st.sidebar.warning("Metadata service unavailable")

st.sidebar.markdown("---")
st.sidebar.markdown("## Assistant Status")

if gpt_helper_ready():
    st.sidebar.success("GPT helper ready")
else:
    st.sidebar.info("GPT helper not configured")

st.sidebar.markdown("---")
st.sidebar.markdown("## Quick Notes")
st.sidebar.info(
    "Upload a paper, review the extracted information, confirm or complete metadata, and run the prediction."
)


# =========================================================
# HOME
# =========================================================
if page == "Home":
    st.markdown(
        """
    <div class="hero-box">
        <div class="hero-title">Research Paper Quality Prediction System</div>
        <div class="hero-subtitle">
            A hybrid web application for predicting whether an individual research paper is likely to be
            <b>4★</b> or <b>Not 4★</b>, using textual content, engineered PDF-based indicators, and metadata features.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_avatar_assistant(
        messages=[
            "Welcome! I can guide you through the paper prediction workflow.",
            "Upload a paper to begin processing.",
            "Metadata can support the prediction alongside the paper text.",
            "Use the help chat below if you have any doubts about the app or demo.",
        ],
        title="Ava · Welcome Guide",
        subtitle="Your animated app assistant",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
        <div class="stat-card">
            <div class="stat-value">UOA 11</div>
            <div class="stat-label">Project Focus</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
        <div class="stat-card">
            <div class="stat-value">Binary</div>
            <div class="stat-label">Classification Task</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
        <div class="stat-card">
            <div class="stat-value">Hybrid</div>
            <div class="stat-label">Prediction Approach</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-heading'>Project Overview</div>", unsafe_allow_html=True)
    st.write(
        "This system supports individual research paper assessment through a hybrid modelling approach. "
        "It combines extracted paper content, structural indicators from PDFs, and contextual metadata "
        "to produce a classification result."
    )
    st.write(
        "When available, metadata can be filled automatically. Otherwise, metadata can be reviewed and completed "
        "as part of the prediction workflow."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
        <div class="mini-card">
            <div class="soft-heading">Paper Upload</div>
            <div class="small-muted">
                Upload a research paper PDF and allow the system to extract title, abstract-style text,
                and structural information for the hybrid prediction workflow.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
        <div class="mini-card">
            <div class="soft-heading">Metadata Completion</div>
            <div class="small-muted">
                Metadata fields can be reviewed and completed before prediction so the model can use
                both contextual and document-derived information.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-heading'>Workflow</div>", unsafe_allow_html=True)
    st.write("1. Upload a paper PDF")
    st.write("2. Extract paper text and abstract-style content")
    st.write("3. Review or complete metadata")
    st.write("4. Run the hybrid prediction model")
    st.write("5. Review the result and supporting details")
    st.markdown("</div>", unsafe_allow_html=True)

    render_help_chat("Home", model_exists, lookup_exists)


# =========================================================
# PREDICT
# =========================================================
elif page == "Predict":
    st.markdown(
        """
    <div class="hero-box">
        <div class="hero-title">Prediction Workspace</div>
        <div class="hero-subtitle">
            Upload a research paper PDF, review the extracted paper information, confirm metadata,
            and run the hybrid model to predict whether the paper is likely to be 4★ or Not 4★.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_avatar_assistant(
        messages=[
            "Start by entering a title or uploading a PDF.",
            "The app will extract text and try to populate metadata when available.",
            "You can review or complete the metadata fields before prediction.",
            "Use the help chat below if you want explanation or demo tips.",
        ],
        title="Ava · Prediction Guide",
        subtitle="Helping you through the prediction process",
    )

    lookup_df = load_metadata_lookup()

    top_left, top_right = st.columns([1.4, 1])

    with top_left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Paper Input")
        title_input = st.text_input("Paper Title", "")
        uploaded_pdf = st.file_uploader("Upload Paper PDF", type=["pdf"])
        manual_text = st.text_area(
            "Abstract / Extracted Text / Key Paper Content",
            height=240,
            placeholder="Paste abstract or important paper text here if needed...",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Input Guidance")
        st.markdown(
            """
        <div class="glass-note">
            <b>Best results:</b><br>
            • upload a readable PDF<br>
            • keep the title accurate<br>
            • review metadata before prediction<br>
            • use manual text if PDF extraction is weak
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
                st.markdown(
                    '<div class="status-good">PDF text extracted successfully. Extracted content will be used for prediction.</div>',
                    unsafe_allow_html=True,
                )
            else:
                text_for_features = pdf_text
                abstract_for_model = manual_text if manual_text.strip() else pdf_text[:3000]
                st.markdown(
                    '<div class="status-warn">PDF text was extracted, but the abstract could not be isolated clearly. Fallback text will be used.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="status-warn">PDF text could not be extracted. The manually entered text will be used instead.</div>',
                unsafe_allow_html=True,
            )

    matched_metadata = find_metadata_match(final_title, uploaded_filename, lookup_df)

    if matched_metadata:
        st.markdown(
            '<div class="status-good">Paper information was identified successfully. Available metadata has been filled in.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-info">You can review and complete the metadata fields before running prediction.</div>',
            unsafe_allow_html=True,
        )

    metric_1, metric_2, metric_3 = st.columns(3)
    with metric_1:
        st.metric("Detected Pages", detected_page_count)
    with metric_2:
        st.metric("PDF Uploaded", "Yes" if uploaded_pdf is not None else "No")
    with metric_3:
        st.metric("Metadata Filled", "Auto / Manual" if uploaded_pdf is not None else "Pending")

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Extracted Paper Information")

        if final_title:
            st.write(f"**Detected title:** {final_title}")
        else:
            st.write("**Detected title:** Not available yet")

        if uploaded_filename:
            st.write(f"**Uploaded file:** {uploaded_filename}")

        preview_text = clean_text(abstract_for_model)[:1200]
        if preview_text:
            st.text_area("Text preview used for model input", preview_text, height=220, disabled=True)
        else:
            st.info("No paper text is currently available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Metadata")

        default_citation = matched_metadata.get("Citation count", 0) if matched_metadata else 0
        default_publisher = matched_metadata.get("Publisher", "") if matched_metadata else ""
        default_institution = matched_metadata.get("Institution name", "") if matched_metadata else ""
        default_ukprn = matched_metadata.get("Institution UKPRN code", 0) if matched_metadata else 0
        default_main_panel = matched_metadata.get("Main panel", "") if matched_metadata else ""
        default_uoa = (
            matched_metadata.get("Unit of assessment name", "Computer Science and Informatics")
            if matched_metadata
            else "Computer Science and Informatics"
        )
        default_oa = matched_metadata.get("Open access status", "Unknown") if matched_metadata else "Unknown"
        default_year = matched_metadata.get("Year", 0) if matched_metadata else 0

        citation_count = st.number_input(
            "Citation Count",
            min_value=0.0,
            value=float(safe_float(default_citation, 0)),
            step=1.0,
        )
        publisher = st.text_input("Publisher", safe_text(default_publisher, ""))
        institution_name = st.text_input("Institution Name", safe_text(default_institution, ""))
        institution_ukprn_code = st.text_input("Institution UKPRN Code", str(default_ukprn if default_ukprn else ""))
        main_panel = st.text_input("Main Panel", safe_text(default_main_panel, ""))
        uoa_name = st.text_input("Unit of Assessment Name", safe_text(default_uoa, "Computer Science and Informatics"))
        year = st.number_input(
            "Year",
            min_value=0,
            max_value=2100,
            value=safe_int(default_year, 0),
            step=1,
        )

        oa_options = [
            "Compliant",
            "Out of scope for open access requirements",
            "Not compliant",
            "Other exception",
            "Unknown",
        ]
        default_oa = str(default_oa) if str(default_oa) in oa_options else "Unknown"
        open_access_status = st.selectbox(
            "Open Access Status",
            oa_options,
            index=oa_options.index(default_oa),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run Prediction"):
        if not final_title.strip():
            st.error("Please enter the paper title, or upload a PDF with a readable first line/title.")
        elif not abstract_for_model.strip():
            st.error("Please provide paper text or upload a readable PDF.")
        else:
            try:
                engineered_features = build_engineered_features(
                    raw_text=text_for_features,
                    page_count=detected_page_count,
                    title=final_title,
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

                if pred == 1:
                    st.markdown(
                        '<div class="result-good">Predicted Class: 4★ Paper</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-warn">Predicted Class: Not 4★ Paper</div>',
                        unsafe_allow_html=True,
                    )

                if confidence is not None:
                    st.write(f"**Confidence:** {confidence:.2%}")

                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric("Prediction", "4★" if pred == 1 else "Not 4★")
                with result_col2:
                    st.metric("Confidence", f"{confidence:.2%}" if confidence is not None else "N/A")
                with result_col3:
                    st.metric("Processed", datetime.now().strftime("%H:%M:%S"))

                with st.expander("Show prediction debug info"):
                    st.write("Raw prediction:", pred)
                    st.write("Classifier classes:", classifier_classes)
                    st.write("Prediction probabilities:", probabilities)
                    st.json(debug_info)

                with st.expander("Show input row used for inference"):
                    st.dataframe(debug_df.T, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    render_help_chat("Predict", model_exists, lookup_exists)


# =========================================================
# ABOUT
# =========================================================
elif page == "About":
    st.markdown(
        """
    <div class="hero-box">
        <div class="hero-title">About the Project</div>
        <div class="hero-subtitle">
            This application forms part of an MSc project on research paper quality prediction
            for Unit of Assessment 11: Computer Science and Informatics.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_avatar_assistant(
        messages=[
            "This project focuses on predicting 4★ versus Not 4★ papers.",
            "The model combines text, engineered PDF indicators, and metadata.",
            "The system can use metadata alongside extracted paper content.",
            "You can ask me questions about the app in the chat below.",
        ],
        title="Ava · Project Guide",
        subtitle="Explaining the project and the app",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="mini-card">
            <div class="soft-heading">Research Context</div>
            <div class="small-muted">
                The project investigates whether the quality of an individual research paper can be predicted
                using a hybrid combination of textual features, engineered PDF-based indicators, and metadata.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="mini-card">
            <div class="soft-heading">Current Scope</div>
            <div class="small-muted">
                The present version focuses on binary classification: predicting whether a paper is likely to be
                4★ or Not 4★ using a hybrid prediction workflow.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Runtime Dependency Check")
    st.json(check_runtime_dependencies())
    st.markdown("</div>", unsafe_allow_html=True)

    render_help_chat("About", model_exists, lookup_exists)

st.markdown(
    """
    <div class="footer-note">
        Research Paper Quality Prediction System · Streamlit Web Application
    </div>
    """,
    unsafe_allow_html=True,
)
