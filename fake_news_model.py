"""
fake_news_model.py — Binary Fake News Detection Pipeline
Implements 5-stage pipeline: Data Gathering → Cleaning → Feature Engineering → Model → Metrics
"""

import os, re, pickle, json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from pydantic import BaseModel, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

FAKE_NEWS_LABEL_MAP = {0: "Real", 1: "Fake"}
MODELS_PATH = "models/"
FAKE_VECTORIZER_PATH = "models/fake_news_vectorizer.pkl"
FAKE_MODEL_PATH = "models/fake_news_model.pkl"
FAKE_EVAL_PATH = "models/fake_news_evaluation.json"

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn","said","says","new","also","would","could","may","one","two",
    "three","us",
}

# ============================================================================
# STAGE 1 — DATA GATHERING: Pydantic Model & Data Loading
# ============================================================================

class NewsArticleRecord(BaseModel):
    """Pydantic schema for validating fake news dataset records."""
    title: str
    text: str
    subject: str
    label: int

    class Config:
        str_strip_whitespace = True


def validate_record(row: pd.Series) -> Tuple[bool, str]:
    """
    Validate a single row using Pydantic model.
    Returns (is_valid, error_message).
    """
    try:
        NewsArticleRecord(
            title=row.get("title", ""),
            text=row.get("text", ""),
            subject=row.get("subject", ""),
            label=row.get("label", 0)
        )
        return True, ""
    except ValidationError as e:
        return False, str(e)


def load_fake_news_data(true_csv: str, fake_csv: str) -> pd.DataFrame:
    """
    STAGE 1: Load ISOT Fake News Dataset from two CSVs (True.csv, Fake.csv).
    - Combine both datasets
    - Add binary label column (0=Real, 1=Fake)
    - Validate sample of rows using Pydantic model
    """
    print("\n[STAGE 1] DATA GATHERING")
    print(f"  Loading True news from: {true_csv}")
    print(f"  Loading Fake news from: {fake_csv}")
    
    # Load both CSVs
    df_true = pd.read_csv(true_csv)
    df_fake = pd.read_csv(fake_csv)
    
    # Normalize column names (strip whitespace)
    df_true.columns = [c.strip().lower() for c in df_true.columns]
    df_fake.columns = [c.strip().lower() for c in df_fake.columns]
    
    # Add label column
    df_true["label"] = 0  # Real
    df_fake["label"] = 1  # Fake
    
    # Combine datasets
    df = pd.concat([df_true, df_fake], ignore_index=True)
    print(f"  Total records loaded: {len(df):,} (Real: {len(df_true):,}, Fake: {len(df_fake):,})")
    
    # Validate sample of 10 random rows
    print(f"  Validating sample of rows...")
    sample_indices = np.random.choice(len(df), min(10, len(df)), replace=False)
    invalid_count = 0
    
    for idx in sample_indices:
        is_valid, error_msg = validate_record(df.iloc[idx])
        if not is_valid:
            print(f"    ⚠ Invalid row {idx}: {error_msg}")
            invalid_count += 1
    
    if invalid_count == 0:
        print(f"  ✓ All {len(sample_indices)} sample rows validated successfully")
    else:
        print(f"  ⚠ {invalid_count}/{len(sample_indices)} sample rows had validation warnings")
    
    return df


# ============================================================================
# STAGE 2 — DATA CLEANING: Preprocessing & Deduplication
# ============================================================================

def preprocess(text: str) -> str:
    """
    Preprocess text: lowercase, strip HTML tags, remove punctuation, filter stopwords.
    Mirrors the preprocess() function from model.py for consistency.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(t for t in text.split() if t not in STOPWORDS and len(t) > 2)


def clean_fake_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 2: Data Cleaning
    - Drop rows with null title or text
    - Drop duplicate articles (based on text column)
    - Remove articles shorter than 20 characters
    - Apply preprocessing (lowercase, strip HTML, remove punctuation)
    """
    print("\n[STAGE 2] DATA CLEANING")
    initial_count = len(df)
    
    # Drop rows with null title or text
    print(f"  Dropping rows with null title or text...")
    df = df.dropna(subset=["title", "text"])
    print(f"    Records after null drop: {len(df):,} (removed {initial_count - len(df):,})")
    
    # Drop duplicates based on text column
    print(f"  Dropping duplicate articles...")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    print(f"    Records after dedup: {len(df):,} (removed {before_dedup - len(df):,})")
    
    # Remove short articles (< 20 chars)
    print(f"  Removing articles shorter than 20 characters...")
    before_length = len(df)
    df["text_length"] = df["text"].str.len()
    df = df[df["text_length"] >= 20]
    print(f"    Records after length filter: {len(df):,} (removed {before_length - len(df):,})")
    
    # Apply preprocessing
    print(f"  Applying preprocessing (lowercase, strip HTML, remove punctuation)...")
    df["text_cleaned"] = df["text"].apply(preprocess)
    df["title_cleaned"] = df["title"].apply(preprocess)
    
    print(f"  ✓ Data cleaning complete. Final record count: {len(df):,}")
    return df


# ============================================================================
# STAGE 3 — FEATURE ENGINEERING: TF-IDF + Scalar Features
# ============================================================================

def engineer_features(df: pd.DataFrame) -> Tuple[sp.csr_matrix, TfidfVectorizer]:
    """
    STAGE 3: Feature Engineering
    - Combine title + text into single field
    - Apply TF-IDF vectorization (max_features=15000, ngram_range=(1,2), sublinear_tf=True)
    - Engineer scalar features: text_length (word count) and exclamation_count
    - Combine using scipy.sparse.hstack
    - Save vectorizer as models/fake_news_vectorizer.pkl
    """
    print("\n[STAGE 3] FEATURE ENGINEERING")
    
    # Combine title and text
    print(f"  Combining title + text fields...")
    df["combined_text"] = df["title_cleaned"] + " " + df["text_cleaned"]
    
    # TF-IDF Vectorization
    print(f"  Applying TF-IDF vectorization (max_features=15000, ngram=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(df["combined_text"])
    print(f"    TF-IDF matrix shape: {X_tfidf.shape}")
    
    # Engineer scalar features
    print(f"  Engineering scalar features...")
    
    # Feature 1: Word count (text_length)
    text_length = df["combined_text"].str.split().str.len().values.reshape(-1, 1)
    # Normalize by dividing by max to keep values in reasonable range
    text_length = text_length / (text_length.max() + 1)
    
    # Feature 2: Exclamation count
    exclamation_count = df["text"].str.count("!").values.reshape(-1, 1)
    # Normalize
    exclamation_count = exclamation_count / (exclamation_count.max() + 1)
    
    print(f"    Scalar features: text_length, exclamation_count")
    
    # Combine TF-IDF with scalar features using sparse matrix stacking
    print(f"  Combining TF-IDF with scalar features using scipy.sparse.hstack...")
    scalar_features = sp.hstack([
        sp.csr_matrix(text_length),
        sp.csr_matrix(exclamation_count)
    ])
    X_combined = sp.hstack([X_tfidf, scalar_features])
    print(f"    Final feature matrix shape: {X_combined.shape}")
    
    # Save vectorizer
    os.makedirs(MODELS_PATH, exist_ok=True)
    with open(FAKE_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  ✓ Vectorizer saved to {FAKE_VECTORIZER_PATH}")
    
    return X_combined, vectorizer


# ============================================================================
# STAGE 4 — MODEL IMPLEMENTATION: Train Logistic Regression
# ============================================================================

def train_fake_model(X: sp.csr_matrix, y: np.ndarray) -> LogisticRegression:
    """
    STAGE 4: Model Implementation
    - 80/20 train/test split, stratified
    - Train Logistic Regression (C=5.0, max_iter=1000, solver=lbfgs)
    - Save model as models/fake_news_model.pkl
    """
    print("\n[STAGE 4] MODEL IMPLEMENTATION")
    
    # Train/test split
    print(f"  Splitting data: 80% train, 20% test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"    Train set: {len(y_train):,} samples")
    print(f"    Test set: {len(y_test):,} samples")
    
    # Train Logistic Regression
    print(f"  Training Logistic Regression (C=5.0, max_iter=1000, solver=lbfgs)...", end=" ", flush=True)
    model = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_train, y_train)
    print("✓")
    
    # Save model
    with open(FAKE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved to {FAKE_MODEL_PATH}")
    
    return model, X_train, X_test, y_train, y_test


# ============================================================================
# STAGE 5 — PERFORMANCE METRICS: Evaluation & ROC Curve
# ============================================================================

def compute_metrics(model: LogisticRegression, X_test: sp.csr_matrix, y_test: np.ndarray) -> Dict[str, Any]:
    """
    STAGE 5: Performance Metrics
    - Compute: accuracy, precision, recall, F1-score, ROC-AUC
    - Generate confusion matrix and classification report
    - Generate ROC curve data points (fpr, tpr, thresholds as lists)
    - Save all metrics to models/fake_news_evaluation.json
    """
    print("\n[STAGE 5] PERFORMANCE METRICS")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    print(f"  Computing metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"    Accuracy:  {accuracy*100:.2f}%")
    print(f"    Precision: {precision*100:.2f}%")
    print(f"    Recall:    {recall*100:.2f}%")
    print(f"    F1-Score:  {f1*100:.2f}%")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    print(f"    Confusion Matrix:\n      {cm}")
    
    # ROC curve data
    print(f"  Generating ROC curve data...")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist()
    }
    
    # Build evaluation data
    eval_data = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(roc_auc, 4),
        "report": report,
        "confusion_matrix": cm,
        "labels": ["Real", "Fake"],
        "roc_curve": roc_data
    }
    
    # Save evaluation data
    # Clean non-JSON-compliant floats (inf, nan) before saving
    import math
    def clean_floats(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: clean_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_floats(v) for v in obj]
        return obj

    eval_data = clean_floats(eval_data)

    # Save evaluation data
    with open(FAKE_EVAL_PATH, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"  ✓ Metrics saved to {FAKE_EVAL_PATH}")
    
    return eval_data


# ============================================================================
# HELPER FUNCTIONS: Model Management
# ============================================================================

def fake_models_exist() -> bool:
    """Check if all required fake news model files exist."""
    return (os.path.exists(FAKE_VECTORIZER_PATH) and 
            os.path.exists(FAKE_MODEL_PATH) and 
            os.path.exists(FAKE_EVAL_PATH))


def load_fake_model() -> Tuple[LogisticRegression, TfidfVectorizer, Dict[str, Any]]:
    """
    Load the trained fake news model, vectorizer, and evaluation metrics.
    Returns: (model, vectorizer, eval_data)
    """
    if not fake_models_exist():
        raise FileNotFoundError("Fake news model files not found. Run training first.")
    
    with open(FAKE_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    with open(FAKE_VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    
    with open(FAKE_EVAL_PATH, "r") as f:
        eval_data = json.load(f)
    
    return model, vectorizer, eval_data


def predict_fake(text: str, model: LogisticRegression, vectorizer: TfidfVectorizer) -> Dict[str, Any]:
    """
    Predict whether an article is fake news.
    
    Returns dict with:
    - label: "Real" or "Fake"
    - confidence: confidence percentage (0-100)
    - label_text: human-readable label
    - all_probs: dict with probabilities for each class
    - keywords: top TF-IDF terms extracted from the text
    - roc_hint: prediction probability for ROC curve integration
    """
    # Preprocess text
    text_cleaned = preprocess(text)
    
    # Vectorize
    X_vec = vectorizer.transform([text_cleaned])
    
    # Add scalar features (for consistency with training pipeline)
    text_length = np.array([[len(text_cleaned.split()) / 1000]])  # Rough normalization
    exclamation_count = np.array([[text.count("!") / 10]])  # Rough normalization
    
    scalar_features = sp.hstack([
        sp.csr_matrix(text_length),
        sp.csr_matrix(exclamation_count)
    ])
    X_combined = sp.hstack([X_vec, scalar_features])
    
    # Predict
    prediction = model.predict(X_combined)[0]
    proba = model.predict_proba(X_combined)[0]
    confidence = float(proba[int(prediction)])
    
    # Extract keywords (top TF-IDF features)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X_vec.toarray()[0]
    top_indices = tfidf_scores.argsort()[-12:][::-1]
    keywords = [
        {"word": feature_names[i], "score": round(float(tfidf_scores[i]), 4)}
        for i in top_indices if tfidf_scores[i] > 0
    ]
    
    return {
        "label": int(prediction),
        "confidence": round(confidence * 100, 2),
        "label_text": FAKE_NEWS_LABEL_MAP[int(prediction)],
        "all_probs": {
            "Real": round(float(proba[0]) * 100, 2),
            "Fake": round(float(proba[1]) * 100, 2)
        },
        "keywords": keywords,
        "roc_hint": round(float(proba[1]), 4)  # Probability of being fake for ROC curve
    }


# ============================================================================
# MAIN: Train the complete pipeline
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Get file paths from command line arguments or use defaults
    true_csv = sys.argv[1] if len(sys.argv) > 1 else "data/True.csv"
    fake_csv = sys.argv[2] if len(sys.argv) > 2 else "data/Fake.csv"
    
    print("=" * 70)
    print("FAKE NEWS DETECTION PIPELINE")
    print("=" * 70)
    
    try:
        # STAGE 1: Load data
        df = load_fake_news_data(true_csv, fake_csv)
        
        # STAGE 2: Clean data
        df = clean_fake_news_data(df)
        
        # STAGE 3: Engineer features
        X, vectorizer = engineer_features(df)
        y = df["label"].values
        
        # STAGE 4: Train model
        model, X_train, X_test, y_train, y_test = train_fake_model(X, y)
        
        # STAGE 5: Compute metrics
        eval_data = compute_metrics(model, X_test, y_test)
        
        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nModel files saved:")
        print(f"  • {FAKE_MODEL_PATH}")
        print(f"  • {FAKE_VECTORIZER_PATH}")
        print(f"  • {FAKE_EVAL_PATH}")
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print(f"\nPlease ensure the dataset files exist:")
        print(f"  • {true_csv}")
        print(f"  • {fake_csv}")
        print(f"\nDownload the ISOT Fake News Dataset from:")
        print(f"  https://www.kaggle.com/datasets/emineyetis/isot-fake-news-dataset")
