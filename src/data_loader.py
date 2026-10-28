"""
Module: data_loader.py
Implements Task A — NumPy and Pandas Exploratory Data Analysis.

Key Operations:
1. Load resumes.csv and validate schema/integrity.
2. Check for missing values and duplicates.
3. Compute summary statistics (mean, median years experience using NumPy).
4. Engineer text_length feature (character count).
5. Compute job category distribution (counts & percentages).
6. Perform skill frequency analysis against a standardized taxonomy.
"""

import os
import json
import numpy as np
import pandas as pd


def load_and_clean_dataset(csv_path="data/resumes.csv"):
    """
    Loads dataset, removes duplicate candidates, and fills null resume texts.
    Returns cleaned DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    initial_count = len(df)
    # Deduplicate by candidate_id
    df = df.drop_duplicates(subset="candidate_id").reset_index(drop=True)
    deduped_count = len(df)

    # Impute missing resume text with empty string if any
    df["resume_text"] = df["resume_text"].fillna("")

    # Engineer text_length feature (character count)
    df["text_length"] = df["resume_text"].str.len()

    # Engineer word_count feature (space-separated word count)
    df["word_count"] = df["resume_text"].apply(lambda t: len(t.split()))

    print(f"Loaded {deduped_count} records (dropped {initial_count - deduped_count} duplicates).")
    return df


def calculate_experience_statistics(df):
    """
    Computes numerical statistics on years of experience using NumPy.
    """
    exp_series = df["years_experience"].dropna()
    stats = {
        "mean_experience": float(np.mean(exp_series)),
        "median_experience": float(np.median(exp_series)),
        "std_experience": float(np.std(exp_series)),
        "min_experience": float(np.min(exp_series)),
        "max_experience": float(np.max(exp_series)),
        "p25_experience": float(np.percentile(exp_series, 25)),
        "p75_experience": float(np.percentile(exp_series, 75))
    }
    return stats


def calculate_category_distribution(df):
    """
    Calculates absolute count and percentage of resumes across job categories.
    """
    counts = df["job_category"].value_counts()
    percentages = df["job_category"].value_counts(normalize=True) * 100
    summary_df = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": percentages.round(2)
    })
    return summary_df


def analyze_skill_frequencies(df, taxonomy_path="data/skills_taxonomy.json", top_n=20):
    """
    Extracts and aggregates skill occurrences across all candidate profiles.
    Returns a DataFrame of top skills and their domain breakdowns.
    """
    # Load taxonomy if available
    known_skills = None
    if os.path.exists(taxonomy_path):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            tax_data = json.load(f)
            known_skills = set(tax_data.get("all_skills", []))

    all_extracted_skills = []
    for skills_field in df["skills"].dropna():
        items = [s.strip() for s in skills_field.split(";") if s.strip()]
        for item in items:
            if known_skills is None or item in known_skills:
                all_extracted_skills.append(item)

    skill_counts = pd.Series(all_extracted_skills).value_counts()
    top_skills_df = pd.DataFrame({
        "Skill": skill_counts.index[:top_n],
        "Frequency": skill_counts.values[:top_n],
        "Candidate_Percentage (%)": ((skill_counts.values[:top_n] / len(df)) * 100).round(2)
    })
    return top_skills_df


def print_task_a_summary(df):
    """
    Console inspection routine matching Task A requirements.
    """
    print("=" * 60)
    print("TASK A: NUMPY & PANDAS EXPLORATION SUMMARY")
    print("=" * 60)
    print("\n1. First 5 Rows:")
    print(df.head(5)[["candidate_id", "job_category", "years_experience", "text_length"]])

    print(f"\n2. Dataset Shape: {df.shape}")

    print("\n3. Missing Value Analysis:")
    print(df.isnull().sum())

    stats = calculate_experience_statistics(df)
    print(f"\n4. Experience Statistics (NumPy):")
    print(f"   Mean:   {stats['mean_experience']:.2f} years")
    print(f"   Median: {stats['median_experience']:.2f} years")
    print(f"   Range:  [{stats['min_experience']:.0f}, {stats['max_experience']:.0f}] years")

    print("\n5. Job Category Distribution:")
    print(calculate_category_distribution(df))

    print("\n6. Top 10 Most Common Skills:")
    print(analyze_skill_frequencies(df, top_n=10))
    print("=" * 60)


if __name__ == "__main__":
    df = load_and_clean_dataset("data/resumes.csv")
    print_task_a_summary(df)
