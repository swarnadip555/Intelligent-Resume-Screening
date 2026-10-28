"""
Pipeline Runner for Intelligent Resume Screening & Job Matching System.
Executes Tasks A through E and GenAI extension end-to-end.
"""

import os
# Prioritize local HF cache for fast execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import numpy as np
import pandas as pd

from src.data_loader import (
    load_and_clean_dataset,
    calculate_experience_statistics,
    calculate_category_distribution,
    analyze_skill_frequencies,
    print_task_a_summary
)
from src.tokenizer_analysis import (
    get_tokenizer,
    inspect_text_tokenization,
    add_token_count_feature,
    analyze_technical_jargon,
    get_tokenization_concept_table
)
from src.classifier import (
    train_and_evaluate_classifier,
    predict_category
)
from src.semantic_matcher import (
    get_embedder,
    compute_pairwise_similarity,
    match_resumes_to_job,
    compute_cross_category_matrix,
    analyze_keyword_vs_semantic
)
from src.visualizer import (
    plot_category_distribution,
    plot_experience_distribution,
    plot_top_skills,
    plot_similarity_scores,
    plot_confusion_matrix,
    plot_text_length_vs_tokens
)
from src.feedback_assistant import (
    generate_feedback
)


def run_full_pipeline():
    print("=" * 80)
    print("INTELLIGENT RESUME SCREENING & JOB MATCHING SYSTEM — PIPELINE EXECUTION")
    print("=" * 80)

    # 1. TASK A: Data Loading & Exploration
    csv_path = os.path.join("data", "resumes.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Generating now...")
        import generate_dataset
        generate_dataset.main()

    df = load_and_clean_dataset(csv_path)
    print_task_a_summary(df)

    # 2. TASK B: Tokenizer Analysis
    print("\n" + "=" * 80)
    print("TASK B: HUGGING FACE TOKENIZATION & TOKEN ID ANALYSIS")
    print("=" * 80)
    tokenizer = get_tokenizer("distilbert/distilbert-base-uncased")
    sample_text = "Python, SQL, Pandas and data visualization experience."
    tok_analysis = inspect_text_tokenization(sample_text, tokenizer)

    print(f"Sample Text: '{sample_text}'")
    print(f"Tokens: {tok_analysis['tokens']}")
    print(f"Token IDs: {tok_analysis['input_ids']}")
    print(f"Token Count: {tok_analysis['token_count']}")

    print("\nTechnical Jargon Tokenization Behavior:")
    jargon_df = analyze_technical_jargon(tokenizer)
    for _, row in jargon_df.iterrows():
        print(f"• '{row['Text']}' -> {len(row['Tokens'])} tokens: {row['Tokens']}")

    df = add_token_count_feature(df, tokenizer=tokenizer)
    print(f"\nAdded 'token_count' feature across {len(df)} candidate records.")
    print(f"Mean Token Count: {np.mean(df['token_count']):.1f} tokens")
    print(f"Min/Max Token Count: [{np.min(df['token_count'])}, {np.max(df['token_count'])}]")

    print("\nConcept Comparison Table (Characters vs Words vs Tokens vs IDs):")
    print(get_tokenization_concept_table().to_string(index=False))

    # 3. TASK C: Random Forest Classifier
    print("\n" + "=" * 80)
    print("TASK C: RANDOM FOREST JOB-CATEGORY CLASSIFIER")
    print("=" * 80)
    rf_results = train_and_evaluate_classifier(
        df,
        test_size=0.2,
        random_state=42,
        save_path="data/rf_classifier.joblib"
    )

    print(f"Model Accuracy on Test Set: {rf_results['accuracy'] * 100:.2f}%")
    print(f"5-Fold Cross-Validation Macro F1: {rf_results['cv_mean'] * 100:.2f}% (+/- {rf_results['cv_std'] * 100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(rf_results["classification_report_text"])

    print("Confusion Matrix:")
    print(rf_results["confusion_matrix"])

    print("\nTop 10 Most Predictive N-grams:")
    for feat, imp in rf_results["top_features"][:10]:
        print(f"  - {feat:20s}: {imp:.4f}")

    # 4. TASK D: Semantic Matching with SentenceTransformers
    print("\n" + "=" * 80)
    print("TASK D: SEMANTIC MATCHING WITH HUGGING FACE SENTENCE TRANSFORMERS")
    print("=" * 80)
    embedder = get_embedder("sentence-transformers/all-MiniLM-L6-v2")

    with open(os.path.join("data", "job_descriptions.json"), "r", encoding="utf-8") as f:
        job_descs = json.load(f)

    target_category = "Data Analyst"
    target_jd = job_descs[target_category]["description"]
    print(f"Matching resumes against Target Job Description: '{job_descs[target_category]['title']}'")
    print(f"JD Snippet: {target_jd[:120]}...")

    ranked_candidates = match_resumes_to_job(df, target_jd, embedder=embedder, top_n=10)
    print(f"\nTop 5 Ranked Candidates for {target_category}:")
    print(ranked_candidates[["candidate_id", "job_category", "years_experience", "similarity_score"]].head(5).to_string(index=False))

    print("\nCross-Category Semantic Similarity Matrix (Mean Cosine Similarity):")
    cross_matrix = compute_cross_category_matrix(df, embedder=embedder)
    print(cross_matrix.to_string())

    # 5. TASK E: Matplotlib Visualizations
    print("\n" + "=" * 80)
    print("TASK E: GENERATING MATPLOTLIB VISUALIZATIONS")
    print("=" * 80)
    os.makedirs("charts", exist_ok=True)
    c1 = plot_category_distribution(df, "charts/01_category_distribution.png")
    c2 = plot_experience_distribution(df, "charts/02_experience_distribution.png")
    c3 = plot_top_skills(df, top_n=15, save_path="charts/03_top_skills.png")
    c4 = plot_similarity_scores(ranked_candidates, job_title="Data Analyst", top_n=10, save_path="charts/04_similarity_scores.png")
    c5 = plot_confusion_matrix(rf_results["confusion_matrix"], save_path="charts/05_confusion_matrix.png")
    c6 = plot_text_length_vs_tokens(df, "charts/06_text_vs_tokens.png")
    print("All 6 visualization charts generated successfully!")

    # 6. GENAI EXTENSION: Resume Feedback Assistant
    print("\n" + "=" * 80)
    print("GENAI EXTENSION: RESUME FEEDBACK ASSISTANT")
    print("=" * 80)
    sample_candidate = df.iloc[0].to_dict()
    target_job = job_descs["Data Scientist"]
    feedback, log_entry = generate_feedback(sample_candidate, target_job, log_path="reports/genai_prompt_log.json")
    print(f"Generated Feedback for Candidate {sample_candidate['candidate_id']} applying for {target_job['title']}:\n")
    print(feedback)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
