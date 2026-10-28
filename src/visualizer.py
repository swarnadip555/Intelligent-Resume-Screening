"""
Module: visualizer.py
Implements Task E — Matplotlib Visualizations and Business Insights.

Charts Generated:
1. Category Distribution: Bar chart showing resume counts & balance.
2. Experience Distribution: Histogram with KDE, mean, and median markers.
3. Top Skills Frequency: Horizontal bar chart of the top 15 skills across the corpus.
4. Semantic Matching Scores: Ranked candidate comparison against benchmark job description.
5. Confusion Matrix: Heatmap evaluating Random Forest classification accuracy.
6. Text Length vs Token Count: Scatter / joint distribution illustrating tokenization density.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set consistent aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif"
})


def plot_category_distribution(df, save_path="charts/01_category_distribution.png"):
    """
    Bar chart: Number of resumes in each job category.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    counts = df["job_category"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b5c8f", "#3c8dbc", "#46a5b6", "#5cb85c"]
    bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor="black", alpha=0.85)

    ax.set_title("Distribution of Resumes Across Job Categories", pad=15, fontweight="bold")
    ax.set_xlabel("Job Category", labelpad=10)
    ax.set_ylabel("Number of Resumes", labelpad=10)
    ax.set_ylim(0, max(counts.values) * 1.18)

    # Annotate bar counts and percentages
    total = len(df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            h + 1.2,
            f"{int(h)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_experience_distribution(df, save_path="charts/02_experience_distribution.png"):
    """
    Histogram: Years of experience with mean & median indicators.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    exp = df["years_experience"].dropna()

    mean_exp = np.mean(exp)
    median_exp = np.median(exp)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(exp.min() - 0.5, exp.max() + 1.5, 1)
    ax.hist(exp, bins=bins, color="#3c8dbc", edgecolor="black", alpha=0.75, rwidth=0.85)

    # Add vertical lines for mean and median
    ax.axvline(mean_exp, color="#d9534f", linestyle="--", linewidth=2.2, label=f"Mean: {mean_exp:.2f} yrs")
    ax.axvline(median_exp, color="#2e6da4", linestyle=":", linewidth=2.5, label=f"Median: {median_exp:.1f} yrs")

    ax.set_title("Candidate Experience Distribution (Years)", pad=15, fontweight="bold")
    ax.set_xlabel("Years of Professional Experience", labelpad=10)
    ax.set_ylabel("Candidate Count", labelpad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_top_skills(df, top_n=15, save_path="charts/03_top_skills.png"):
    """
    Horizontal Bar Chart: Most common skills across all candidate profiles.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_skills = []
    for s_field in df["skills"].dropna():
        items = [s.strip() for s in s_field.split(";") if s.strip()]
        all_skills.extend(items)

    counts = pd.Series(all_skills).value_counts().head(top_n).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(counts.index, counts.values, color="#4682b4", edgecolor="#1c3b5e", alpha=0.85)

    ax.set_title(f"Top {top_n} Most Common Skills in Candidate Pool", pad=15, fontweight="bold")
    ax.set_xlabel("Frequency Count", labelpad=10)
    ax.set_ylabel("Skill", labelpad=10)
    ax.set_xlim(0, max(counts.values) * 1.15)

    for bar in bars:
        w = bar.get_width()
        pct = (w / len(df)) * 100
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2., f"{int(w)} ({pct:.0f}%)", va="center", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_similarity_scores(ranked_df, job_title="Data Analyst", top_n=10, save_path="charts/04_similarity_scores.png"):
    """
    Bar Chart: Resume-to-job-description semantic similarity scores for top candidate examples.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    top_candidates = ranked_df.head(top_n).copy()

    # Create label: ID + Job Category
    top_candidates["label"] = top_candidates["candidate_id"] + " (" + top_candidates["job_category"] + ")"
    top_candidates = top_candidates.sort_values(by="similarity_score", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Color bar based on matching category
    colors = [
        "#2ca02c" if cat.lower() in job_title.lower() or job_title.lower() in cat.lower() else "#ff7f0e"
        for cat in top_candidates["job_category"]
    ]

    bars = ax.barh(top_candidates["label"], top_candidates["similarity_score"], color=colors, edgecolor="black", alpha=0.85)

    ax.set_title(f"Top Semantic Matching Scores vs. '{job_title}' Job Description", pad=15, fontweight="bold")
    ax.set_xlabel("Cosine Similarity Score (MiniLM Embeddings)", labelpad=10)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="#888", linestyle=":", label="Baseline threshold (0.50)")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2., f"{w:.3f}", va="center", fontweight="bold", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_confusion_matrix(cm_df, labels=None, save_path="charts/05_confusion_matrix.png"):
    """
    Confusion Matrix Heatmap: Random Forest Classifier predictions vs Ground Truth.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_matrix = cm_df.values
    classes = cm_df.columns.tolist() if labels is None else labels

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(cm_matrix, cmap="Blues", alpha=0.85)
    fig.colorbar(cax, shrink=0.8)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="left")
    ax.set_yticklabels(classes)

    ax.set_title("Random Forest: Confusion Matrix", pad=25, fontweight="bold")
    ax.set_xlabel("Predicted Category", labelpad=12, fontweight="bold")
    ax.set_ylabel("True Category", labelpad=12, fontweight="bold")

    # Annotate matrix values
    thresh = cm_matrix.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_matrix[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_text_length_vs_tokens(df, save_path="charts/06_text_vs_tokens.png"):
    """
    Scatter & Regression Chart: Resume Character Length vs Token Count.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if "token_count" not in df.columns:
        print("token_count column missing, skipping text_vs_tokens chart.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = df["job_category"].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for cat, color in zip(categories, colors):
        sub = df[df["job_category"] == cat]
        ax.scatter(sub["text_length"], sub["token_count"], label=cat, color=color, alpha=0.7, edgecolors="none", s=45)

    ax.set_title("Resume Text Character Length vs. Token Count", pad=15, fontweight="bold")
    ax.set_xlabel("Character Count (text_length)", labelpad=10)
    ax.set_ylabel("WordPiece Token Count (token_count)", labelpad=10)
    ax.legend(title="Job Category", frameon=True, facecolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
    return save_path
