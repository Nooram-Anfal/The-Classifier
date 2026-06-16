"""
model.py — Multi-model NLP pipeline for News Category Classification
"""

import os, re, pickle, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CATEGORY_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
CATEGORIES   = ["World", "Sports", "Business", "Sci/Tech"]

MODEL_DEFS = {
    "naive_bayes": {
        "label": "Naive Bayes",
        "short": "NB",
        "description": "Multinomial Naive Bayes. Fast, probabilistic, ideal for sparse TF-IDF features.",
    },
    "logistic_regression": {
        "label": "Logistic Regression",
        "short": "LR",
        "description": "Linear classifier with L2 regularisation. Strong baseline, well-calibrated probabilities.",
    },
    "linear_svc": {
        "label": "Linear SVC",
        "short": "SVC",
        "description": "Support Vector Classifier with linear kernel. Maximises classification margin.",
    },
    "sgd": {
        "label": "SGD Classifier",
        "short": "SGD",
        "description": "Stochastic Gradient Descent with log-loss. Fast and scalable to large corpora.",
    },
}

MODELS_PATH     = "models/"
VECTORIZER_PATH = "models/vectorizer.pkl"
EVAL_PATH       = "models/evaluation.json"

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

def preprocess(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(t for t in text.split() if t not in STOPWORDS and len(t) > 2)

def combine_fields(row):
    return str(row.get("Title","")) + " " + str(row.get("Description",""))

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["text"]  = df.apply(combine_fields, axis=1).apply(preprocess)
    df["label"] = df["Class Index"].map(CATEGORY_MAP)
    return df.dropna(subset=["label"])

def _make_models():
    return {
        "naive_bayes": MultinomialNB(alpha=0.1),
        "logistic_regression": LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs"),
        "linear_svc": CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3),
        "sgd": CalibratedClassifierCV(SGDClassifier(loss="log_loss", max_iter=200, random_state=42), cv=3),
    }

def train_all(df):
    os.makedirs(MODELS_PATH, exist_ok=True)
    X, y = df["text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=30_000, ngram_range=(1,2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    with open(VECTORIZER_PATH, "wb") as f: pickle.dump(vectorizer, f)

    models    = _make_models()
    eval_data = {}

    for key, model in models.items():
        print(f"  Training {MODEL_DEFS[key]['label']}...", end=" ", flush=True)
        model.fit(X_train_vec, y_train)
        y_pred  = model.predict(X_test_vec)
        acc     = accuracy_score(y_test, y_pred)
        report  = classification_report(y_test, y_pred, output_dict=True)
        cm      = confusion_matrix(y_test, y_pred, labels=CATEGORIES).tolist()
        eval_data[key] = {"accuracy": round(acc*100,2), "report": report, "confusion": cm, "labels": CATEGORIES}
        print(f"{acc*100:.2f}%")
        with open(f"{MODELS_PATH}{key}.pkl","wb") as f: pickle.dump(model, f)

    with open(EVAL_PATH,"w") as f: json.dump(eval_data, f)
    print("  All models saved.\n")
    return models, vectorizer, eval_data

def load_all_models():
    with open(VECTORIZER_PATH,"rb") as f: vectorizer = pickle.load(f)
    models = {}
    for key in MODEL_DEFS:
        path = f"{MODELS_PATH}{key}.pkl"
        if os.path.exists(path):
            with open(path,"rb") as f: models[key] = pickle.load(f)
    with open(EVAL_PATH) as f: eval_data = json.load(f)
    return models, vectorizer, eval_data

def models_exist():
    return (os.path.exists(VECTORIZER_PATH) and os.path.exists(EVAL_PATH)
            and all(os.path.exists(f"{MODELS_PATH}{k}.pkl") for k in MODEL_DEFS))

def predict_one(text, model_key, models, vectorizer):
    model = models[model_key]
    clean = preprocess(text)
    vec   = vectorizer.transform([clean])
    probs   = model.predict_proba(vec)[0]
    classes = model.classes_
    idx     = int(np.argmax(probs))
    category   = classes[idx]
    confidence = float(probs[idx])
    all_probs  = {cls: round(float(p)*100,2) for cls, p in zip(classes, probs)}

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores  = vec.toarray()[0]
    top_indices   = tfidf_scores.argsort()[-12:][::-1]
    keywords = [{"word": feature_names[i], "score": round(float(tfidf_scores[i]),4)}
                for i in top_indices if tfidf_scores[i] > 0]

    return {
        "category":    category,
        "confidence":  round(confidence*100,2),
        "all_probs":   all_probs,
        "keywords":    keywords,
        "model_key":   model_key,
        "model_label": MODEL_DEFS[model_key]["label"],
    }

def predict_batch(texts, model_key, models, vectorizer):
    results = []
    for text in texts:
        t = text.strip()
        if not t: continue
        r = predict_one(t, model_key, models, vectorizer)
        r["input"] = t
        results.append(r)
    return results

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.csv"
    print(f"\nLoading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"Loaded {len(df):,} records.\n")
    train_all(df)