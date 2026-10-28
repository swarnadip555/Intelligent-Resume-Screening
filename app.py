"""
Streamlit Web Application: Intelligent Resume Screening & Job Matching System
An Industry-Grade GenAI Practice Prototype.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Offline cache flags
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.data_loader import (
    load_and_clean_dataset,
    calculate_experience_statistics,
    calculate_category_distribution,
    analyze_skill_frequencies
)
from src.tokenizer_analysis import (
    get_tokenizer,
    inspect_text_tokenization,
    get_tokenization_concept_table
)
from src.classifier import (
    train_and_evaluate_classifier,
    predict_category
)
from src.semantic_matcher import (
    get_embedder,
    match_resumes_to_job,
    compute_pairwise_similarity
)
from src.feedback_assistant import (
    generate_feedback
)

# Page configuration
st.set_page_config(
    page_title="Intelligent Resume Screening & Job Matching",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 6px;
        padding: 1rem;
        color: #92400E;
        font-size: 0.95rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    csv_path = os.path.join("data", "resumes.csv")
    if not os.path.exists(csv_path):
        import generate_dataset
        generate_dataset.main()
    return load_and_clean_dataset(csv_path)


@st.cache_resource
def load_models():
    tokenizer = get_tokenizer("distilbert/distilbert-base-uncased")
    embedder = get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, embedder


def load_job_descriptions():
    jd_path = os.path.join("data", "job_descriptions.json")
    with open(jd_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load data and models
df = load_data()
job_descriptions = load_job_descriptions()
tokenizer, embedder = load_models()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/resume.png", width=70)
st.sidebar.title("Resume AI System")
st.sidebar.markdown("**Industry:** HR Tech & Recruitment  \n**Tech:** NumPy, Pandas, RF, HF, Transformers")

section = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard & Analytics",
        "🔤 Hugging Face Tokenizer Lab",
        "🎯 Random Forest Classifier",
        "🔍 Semantic Job Matcher",
        "💡 GenAI Feedback Assistant",
        "🛡️ Responsible AI & Governance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ **Decision-Support Only:** Designed to assist recruiters, not to automate rejection or acceptance."
)

# ----------------------------------------------------
# 1. Dashboard & Analytics (Task A & E)
# ----------------------------------------------------
if section == "📊 Dashboard & Analytics":
    st.markdown('<div class="main-header">Candidate Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Task A & Task E — Exploratory Data Analysis and Workforce Insights</div>', unsafe_allow_html=True)

    stats = calculate_experience_statistics(df)
    cat_summary = calculate_category_distribution(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", len(df))
    with col2:
        st.metric("Mean Experience", f"{stats['mean_experience']:.1f} yrs")
    with col3:
        st.metric("Median Experience", f"{stats['median_experience']:.1f} yrs")
    with col4:
        st.metric("Job Categories", len(cat_summary))

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Category Distribution")
        if os.path.exists("charts/01_category_distribution.png"):
            st.image("charts/01_category_distribution.png", use_container_width=True)
        else:
            st.dataframe(cat_summary)

    with chart_col2:
        st.subheader("Experience Distribution (Years)")
        if os.path.exists("charts/02_experience_distribution.png"):
            st.image("charts/02_experience_distribution.png", use_container_width=True)

    st.markdown("---")
    st.subheader("Candidate Pool Data Explorer")
    selected_cat = st.selectbox("Filter by Category", ["All"] + list(cat_summary.index))
    filtered_df = df if selected_cat == "All" else df[df["job_category"] == selected_cat]

    st.dataframe(
        filtered_df[["candidate_id", "job_category", "years_experience", "education_level", "skills", "text_length"]],
        use_container_width=True
    )


# ----------------------------------------------------
# 2. Tokenizer Lab (Task B)
# ----------------------------------------------------
elif section == "🔤 Hugging Face Tokenizer Lab":
    st.markdown('<div class="main-header">Hugging Face Tokenizer Laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Task B — DistilBERT WordPiece Tokenization & Token IDs Analysis</div>', unsafe_allow_html=True)

    default_text = "Python, SQL, Pandas, CI/CD, and PyTorch experience with 5 years in production."
    user_text = st.text_area("Input Text to Tokenize:", value=default_text, height=100)

    if st.button("Tokenize Text", type="primary"):
        analysis = inspect_text_tokenization(user_text, tokenizer)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Character Count", len(user_text))
        with col2:
            st.metric("Word Count", len(user_text.split()))
        with col3:
            st.metric("Subword Token Count", analysis["token_count"])

        st.markdown("#### Subword Tokens Extracted")
        # Visual tokens pills
        token_html = " ".join([
            f'<span style="background-color:#E0E7FF; color:#3730A3; padding:4px 8px; margin:3px; border-radius:4px; display:inline-block; font-family:monospace; font-weight:600;">{t}</span>'
            for t in analysis["tokens"]
        ])
        st.markdown(token_html, unsafe_allow_html=True)

        st.markdown("#### Input Token IDs (Vocabulary Tensor Indices)")
        st.code(str(analysis["input_ids"]), language="python")

        st.markdown("#### Special Tokens Decoded (Includes `[CLS]` and `[SEP]`)")
        st.code(str(analysis["special_tokens_decoded"]), language="python")

    st.markdown("---")
    st.subheader("Theoretical NLP Foundation")
    st.dataframe(get_tokenization_concept_table(), use_container_width=True)


# ----------------------------------------------------
# 3. Random Forest Classifier (Task C & E)
# ----------------------------------------------------
elif section == "🎯 Random Forest Classifier":
    st.markdown('<div class="main-header">Random Forest Job-Category Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Task C & Task E — Supervised TF-IDF + Random Forest Pipeline</div>', unsafe_allow_html=True)

    model_path = os.path.join("data", "rf_classifier.joblib")
    if not os.path.exists(model_path):
        with st.spinner("Training Random Forest Classifier..."):
            train_and_evaluate_classifier(df, save_path=model_path)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Test Resume Category Prediction")
        sample_choice = st.selectbox(
            "Or pick from candidate pool:",
            ["Custom Input"] + df["candidate_id"].head(20).tolist()
        )

        if sample_choice != "Custom Input":
            matched_row = df[df["candidate_id"] == sample_choice].iloc[0]
            input_resume = matched_row["resume_text"]
            st.caption(f"Actual Category: **{matched_row['job_category']}**")
        else:
            input_resume = "Skilled in Python, SQL, Tableau dashboards, and performing statistical A/B tests."

        eval_text = st.text_area("Resume Content:", value=input_resume, height=180)

        if st.button("Predict Job Category", type="primary"):
            pred, probs = predict_category(eval_text, model_path=model_path)

            st.success(f"**Predicted Category:** {pred}")

            prob_df = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
            prob_df = prob_df.sort_values(by="Probability", ascending=False)

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(prob_df["Category"], prob_df["Probability"], color="#3b82f6", edgecolor="black")
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Confidence Probability")
            plt.tight_layout()
            st.pyplot(fig)

    with col2:
        st.subheader("Model Diagnostic Confusion Matrix")
        if os.path.exists("charts/05_confusion_matrix.png"):
            st.image("charts/05_confusion_matrix.png", use_container_width=True)


# ----------------------------------------------------
# 4. Semantic Job Matcher (Task D & E)
# ----------------------------------------------------
elif section == "🔍 Semantic Job Matcher":
    st.markdown('<div class="main-header">Semantic Job Matching System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Task D & Task E — Pretrained Sentence Embeddings (`all-MiniLM-L6-v2`)</div>', unsafe_allow_html=True)

    target_category = st.selectbox("Select Target Job Role:", list(job_descriptions.keys()))
    current_jd = job_descriptions[target_category]

    jd_text = st.text_area("Job Description:", value=current_jd["description"], height=120)

    top_k = st.slider("Number of Candidates to Rank:", min_value=5, max_value=25, value=10)

    if st.button("Calculate Semantic Similarity", type="primary"):
        with st.spinner("Generating embeddings & computing cosine similarities..."):
            ranked = match_resumes_to_job(df, jd_text, embedder=embedder, top_n=top_k)

            st.markdown(f"### Top {top_k} Candidates for **{current_jd['title']}**")

            for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                sim = row["similarity_score"]
                is_match = row["job_category"].lower() in target_category.lower()

                with st.expander(f"#{rank} | {row['candidate_id']} - {row['job_category']} (Similarity: {sim:.4f})"):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.write(f"**Experience:** {row['years_experience']} years")
                        st.write(f"**Education:** {row['education_level']}")
                        st.write(f"**Skills:** {row['skills']}")
                    with col_b:
                        st.write(f"**Resume Snippet:** {row['resume_text']}")

            # Matplotlib bar of top candidates
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_sub = ranked.head(10).sort_values(by="similarity_score", ascending=True)
            labels = [f"{r['candidate_id']} ({r['job_category']})" for _, r in plot_sub.iterrows()]
            colors = ["#10b981" if r["job_category"] == target_category else "#f59e0b" for _, r in plot_sub.iterrows()]
            ax.barh(labels, plot_sub["similarity_score"], color=colors, edgecolor="black")
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Cosine Similarity Score")
            ax.set_title("Candidate Similarity Alignment")
            plt.tight_layout()
            st.pyplot(fig)


# ----------------------------------------------------
# 5. GenAI Feedback Assistant (Section 6)
# ----------------------------------------------------
elif section == "💡 GenAI Feedback Assistant":
    st.markdown('<div class="main-header">GenAI Candidate Feedback Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Section 6 — Ethical Decision Support, Gap Analysis & Upskilling Guidance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        selected_cid = st.selectbox("Select Candidate:", df["candidate_id"].tolist())
        target_role = st.selectbox("Target Position:", list(job_descriptions.keys()))

    cand_row = df[df["candidate_id"] == selected_cid].iloc[0].to_dict()
    job_info = job_descriptions[target_role]

    st.markdown("---")
    if st.button("Generate Candidate Evaluation Feedback", type="primary"):
        feedback, log_entry = generate_feedback(cand_row, job_info)

        st.markdown(feedback)

        with st.expander("Audit Log Entry (Transparency & Traceability)"):
            st.json(log_entry)


# ----------------------------------------------------
# 6. Responsible AI & Governance (Section 7)
# ----------------------------------------------------
elif section == "🛡️ Responsible AI & Governance":
    st.markdown('<div class="main-header">Responsible AI, Ethics & Compliance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Section 7 — Mitigating Bias and Protecting Candidate Rights in Recruitment</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. The Decision Support Mandate
    Automated resume rejection systems have caused significant public and legal controversy. This prototype is architected with strict technical guardrails:
    - **No Autonomous Reject/Accept:** The system outputs relevance signals for human recruiters.
    - **Explainability:** Predictions show exact confidence distributions and evidenced skills rather than a black-box decision.

    ### 2. Elimination of Demographic Bias
    - **Feature Whitelisting:** Our models are trained exclusively on job-related skills, experience, and educational domains.
    - **PII Stripping:** Real names, gender indicators, photographs, phone numbers, and physical home addresses are deliberately omitted.
    - **Proxy Mitigation:** Recruiters must remain vigilant against proxy variables (such as graduation years indicating candidate age).

    ### 3. Data Privacy & Synthetic Security
    - All 250 profiles in `resumes.csv` are 100% synthetically generated.
    - No candidate personal data is sent to untrusted third-party APIs.
    """)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Mandatory Compliance Warning:</strong> This application is a technical practice prototype. Never deploy automated machine learning models to eliminate job applicants without compliance with the EU AI Act (High-Risk AI Systems), NYC Local Law 144, and federal Equal Employment Opportunity Commission (EEOC) audits.
    </div>
    """, unsafe_allow_html=True)

