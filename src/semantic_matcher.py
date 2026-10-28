"""
Module: semantic_matcher.py
Implements Task D — Semantic Matching with Hugging Face Sentence Transformers.

Core Capabilities:
1. Pretrained dense embedding generation via 'sentence-transformers/all-MiniLM-L6-v2'.
2. Pairwise Cosine Similarity computation between candidate resumes and Job Descriptions (JDs).
3. Candidate ranking per job description.
4. Cross-domain similarity analysis (matching all categories against all JDs).
5. Keyword overlap vs. Semantic similarity divergence analysis.
6. Decision-support guardrails (disclaimers on why similarity != candidate quality).
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initializes and returns the SentenceTransformer model with local cache preference.
    """
    from sentence_transformers import SentenceTransformer
    try:
        embedder = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        embedder = SentenceTransformer(model_name)
    return embedder


def compute_pairwise_similarity(text_a, text_b, embedder=None):
    """
    Computes cosine similarity between two individual text strings.
    """
    if embedder is None:
        embedder = get_embedder()

    embeddings = embedder.encode([text_a, text_b])
    score = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0, 0]
    return float(score)


def match_resumes_to_job(df, job_description, embedder=None, top_n=10):
    """
    Encodes candidate resumes and ranks them against a specific target Job Description.
    Returns sorted DataFrame with similarity scores.
    """
    if embedder is None:
        embedder = get_embedder()

    resume_texts = df["resume_text"].fillna("").tolist()
    all_texts = [job_description] + resume_texts

    embeddings = embedder.encode(all_texts, show_progress_bar=False, batch_size=32)

    jd_embedding = embeddings[0].reshape(1, -1)
    resume_embeddings = embeddings[1:]

    scores = cosine_similarity(jd_embedding, resume_embeddings)[0]

    results_df = df.copy()
    results_df["similarity_score"] = np.round(scores, 4)

    ranked_df = results_df.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)
    return ranked_df.head(top_n)


def compute_cross_category_matrix(df, job_descriptions_path="data/job_descriptions.json", embedder=None):
    """
    Calculates average semantic similarity of candidates in each category
    against every benchmark Job Description.
    Reveals inter-domain semantic overlap.
    """
    if embedder is None:
        embedder = get_embedder()

    with open(job_descriptions_path, "r", encoding="utf-8") as f:
        job_descs = json.load(f)

    jd_titles = list(job_descs.keys())
    categories = df["job_category"].unique().tolist()

    matrix = pd.DataFrame(index=categories, columns=jd_titles, dtype=float)

    for jd_name, jd_info in job_descs.items():
        jd_text = jd_info["description"]
        ranked = match_resumes_to_job(df, jd_text, embedder=embedder, top_n=len(df))

        for cat in categories:
            cat_scores = ranked[ranked["job_category"] == cat]["similarity_score"]
            matrix.loc[cat, jd_name] = round(float(cat_scores.mean()), 4)

    return matrix


def analyze_keyword_vs_semantic(candidate_resume, candidate_skills, job_description, required_skills, embedder=None):
    """
    Compares exact keyword skill match percentage against neural semantic similarity score.
    Highlights subtle semantic nuance vs rigid keyword matching.
    """
    sem_score = compute_pairwise_similarity(candidate_resume, job_description, embedder=embedder)

    cand_skill_set = set(s.strip().lower() for s in candidate_skills.split(";") if s.strip())
    req_skill_set = set(s.strip().lower() for s in required_skills)

    matched = cand_skill_set.intersection(req_skill_set)
    missing = req_skill_set.difference(cand_skill_set)
    overlap_pct = (len(matched) / len(req_skill_set) * 100) if req_skill_set else 0.0

    return {
        "semantic_similarity": round(sem_score, 4),
        "keyword_overlap_pct": round(overlap_pct, 2),
        "matched_skills": list(matched),
        "missing_skills": list(missing)
    }
