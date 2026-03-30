"""
model.py — NLP pipeline for News Category Classification
Uses TF-IDF + Multinomial Naive Bayes
"""

import os
import re
import pickle
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ─── Constants ──────────────────────────────────────────────────────────────

CATEGORY_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

CATEGORY_ICONS = {
    "World": "🌍",
    "Sports": "⚽",
    "Business": "💼",
    "Sci/Tech": "🔬",
}

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Basic English stopwords (no external dependency)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn", "said", "says", "new",
    "also", "would", "could", "may", "one", "two", "three", "us",
}

# ─── Text Preprocessing ──────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Lowercase → remove punctuation/numbers → remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def combine_fields(row) -> str:
    """Merge title + description into a single text feature."""
    title = str(row.get("Title", ""))
    desc = str(row.get("Description", ""))
    return title + " " + desc


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    df["text"] = df.apply(combine_fields, axis=1).apply(preprocess)
    df["label"] = df["Class Index"].map(CATEGORY_MAP)
    df = df.dropna(subset=["label"])
    return df


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """Train TF-IDF + Multinomial Naive Bayes and save artefacts."""
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorisation
    vectorizer = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # log(tf) dampening
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Naive Bayes
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)

    print("\n" + "=" * 60)
    print("  NEWS CLASSIFIER — MODEL EVALUATION")
    print("=" * 60)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=list(CATEGORY_MAP.values()))
    labels = list(CATEGORY_MAP.values())
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>12}" + "".join(f"{cm[i][j]:>12}" for j in range(len(labels)))
        print(row)
    print("=" * 60 + "\n")

    # ── Persist ─────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("  ✔ Model and vectoriser saved.\n")
    return model, vectorizer


# ─── Prediction ──────────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict(text: str, model, vectorizer) -> dict:
    """Return predicted category, confidence, and per-class probabilities."""
    clean = preprocess(text)
    vec   = vectorizer.transform([clean])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    category  = classes[np.argmax(probs)]
    confidence = float(np.max(probs))

    all_probs = {cls: float(p) for cls, p in zip(classes, probs)}

    return {
        "category":   category,
        "icon":       CATEGORY_ICONS.get(category, "📰"),
        "confidence": round(confidence * 100, 2),
        "all_probs":  {k: round(v * 100, 2) for k, v in all_probs.items()},
    }


# ─── Entry point (train standalone) ─────────────────────────────────────────

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.csv"
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"Loaded {len(df):,} records.")
    train_model(df)
