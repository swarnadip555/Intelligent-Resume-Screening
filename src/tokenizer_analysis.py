"""
Module: tokenizer_analysis.py
Implements Task B — Hugging Face Tokenization and Token ID Analysis.

Core Objectives:
1. Load Hugging Face AutoTokenizer ('distilbert/distilbert-base-uncased' or fallback).
2. Tokenize sample resume text, extract subword tokens and token IDs.
3. Analyze tokenization behavior on punctuation, acronyms (e.g. CI/CD, TCP/IP, MLOps),
   and technical skills (e.g. PyTorch, scikit-learn, PostgreSQL).
4. Compute and compare token counts for short vs. long resumes.
5. Create a vectorized `token_count` feature on DataFrame.
6. Provide structured theoretical comparison:
   Characters vs. Words vs. Tokens vs. Token IDs.
"""

import os
import pandas as pd


def get_tokenizer(model_name="distilbert/distilbert-base-uncased"):
    """
    Loads Hugging Face AutoTokenizer with offline caching support.
    """
    from transformers import AutoTokenizer
    try:
        # Prefer local cached files to prevent network timeout in air-gapped/sandboxed environments
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def inspect_text_tokenization(text, tokenizer=None):
    """
    Breaks down a text into subword tokens, token IDs, and special tokens.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    tokens = tokenizer.tokenize(text)
    encoded = tokenizer(text, truncation=True, max_length=512)
    input_ids = encoded["input_ids"]

    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return {
        "raw_text": text,
        "tokens": tokens,
        "token_count": len(tokens),
        "input_ids": input_ids,
        "input_ids_count": len(input_ids),  # Includes [CLS] and [SEP]
        "special_tokens_decoded": decoded_tokens
    }


def add_token_count_feature(df, tokenizer=None, text_col="resume_text"):
    """
    Vectorizes token count across all candidate resumes in the DataFrame.
    Adds `token_count` and `token_to_word_ratio` columns.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    def count_tokens(text):
        if not text:
            return 0
        return len(tokenizer.tokenize(str(text)))

    df_copy = df.copy()
    df_copy["token_count"] = df_copy[text_col].apply(count_tokens)

    if "word_count" in df_copy.columns:
        df_copy["token_to_word_ratio"] = (
            df_copy["token_count"] / df_copy["word_count"].replace(0, 1)
        ).round(3)

    return df_copy


def analyze_technical_jargon(tokenizer=None):
    """
    Demonstrates how technical skills, acronyms, and punctuation are split by WordPiece.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    jargon_samples = [
        "Python, SQL, Pandas and data visualization experience.",
        "CI/CD, Kubernetes, Docker, and RESTful microservices.",
        "PyTorch, TensorFlow, Scikit-Learn, and MLOps.",
        "TCP/IP, Active Directory, DNS/DHCP, and VPN configuration."
    ]

    analysis = []
    for sample in jargon_samples:
        tokens = tokenizer.tokenize(sample)
        analysis.append({
            "Text": sample,
            "Tokens": tokens,
            "Subword_Count": len(tokens),
            "Has_Subword_Prefix": any(t.startswith("##") for t in tokens)
        })

    return pd.DataFrame(analysis)


def get_tokenization_concept_table():
    """
    Returns comparative explanation of characters, words, tokens, and token IDs.
    """
    concepts = [
        {
            "Concept": "Character",
            "Definition": "The smallest atomic orthographic unit (letters, digits, punctuation, spaces).",
            "Example ('Python 3.11')": "['P', 'y', 't', 'h', 'o', 'n', ' ', '3', '.', '1', '1'] (11 chars)",
            "Pros / Cons in NLP": "Eliminates Out-Of-Vocabulary (OOV) but creates overly long sequences and lacks word semantic grounding."
        },
        {
            "Concept": "Word",
            "Definition": "Whitespace/punctuation-separated linguistic unit.",
            "Example ('Python 3.11')": "['Python', '3.11'] (2 words)",
            "Pros / Cons in NLP": "Intuitive human semantics, but massive vocabulary size and fails entirely on unseen words/typos."
        },
        {
            "Concept": "Subword Token",
            "Definition": "Frequent words or statistical sub-word segments (WordPiece/BPE) prefixed by ## when continuing.",
            "Example ('Python 3.11')": "['python', '3', '.', '11'] (4 tokens)",
            "Pros / Cons in NLP": "Best of both: bounds vocabulary size (~30,522) while gracefully splitting compound or rare words."
        },
        {
            "Concept": "Token ID",
            "Definition": "Unique numerical integer index in model vocabulary tensor input.",
            "Example ('Python 3.11')": "[101, 18750, 1017, 1012, 2340, 102] ([CLS] ... [SEP])",
            "Pros / Cons in NLP": "Direct numerical matrix input required by PyTorch/TensorFlow neural network embedding layers."
        }
    ]
    return pd.DataFrame(concepts)
