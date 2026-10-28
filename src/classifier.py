"""
Module: classifier.py
Implements Task C — Random Forest Job-Category Classifier.

Key Responsibilities:
1. Feature extraction via TF-IDF (unigrams + bigrams, max_features=3000).
2. Random Forest Classification (n_estimators=100, balanced class weights).
3. Stratified train/test splitting (80/20) with strict data leak prevention.
4. Comprehensive evaluation:
   - Precision, Recall, F1-score, Support per category
   - Macro and weighted averages
   - Confusion matrix extraction
   - Top predictive n-grams per job category
5. Model saving and loading via joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_pipeline(max_features=3000, ngram_range=(1, 2), n_estimators=100, random_state=42):
    """
    Constructs a scikit-learn pipeline ensuring vectorizer is fitted strictly on training data.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english"
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])
    return pipeline


def train_and_evaluate_classifier(df, test_size=0.2, random_state=42, save_path="data/rf_classifier.joblib"):
    """
    Trains the Random Forest model and calculates evaluation metrics.
    """
    df_clean = df.dropna(subset=["job_category"]).copy()
    X = df_clean["resume_text"].fillna("")
    y = df_clean["job_category"]

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model = build_pipeline(random_state=random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)

    # Classification metrics
    report_dict = classification_report(y_test, predictions, output_dict=True)
    report_text = classification_report(y_test, predictions)
    acc = accuracy_score(y_test, predictions)

    # Confusion matrix
    labels = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # 5-fold cross validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")

    # Extract top informative features
    feature_names = model.named_steps["tfidf"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = [(feature_names[i], importances[i]) for i in top_indices]

    results = {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": pred_probs,
        "accuracy": acc,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm_df,
        "labels": labels,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "top_features": top_features
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Trained model saved to: {save_path}")

    return results


def predict_category(text, model_path="data/rf_classifier.joblib"):
    """
    Inference helper: predicts category and probability confidence.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    model = joblib.load(model_path)
    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
    return pred, prob_dict
