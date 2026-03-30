# 📰 News Category Classification System
### Naive Bayes · TF-IDF · Flask · scikit-learn

---

## 1. Project Overview

A complete end-to-end NLP system that classifies news articles into one of four categories — **World**, **Sports**, **Business**, or **Sci/Tech** — using a Multinomial Naive Bayes classifier trained on TF-IDF features. The project includes a dark, editorial-themed web frontend built with Flask, HTML, CSS, and vanilla JavaScript.

---

## 2. Dataset Description

**AG News Classification Dataset**  
Source: [Kaggle — amananandrai](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

| Split | Samples |
|-------|---------|
| Train | ~120,000 |
| Test  | 7,600 |

Each record contains:
- `Class Index` — 1 (World), 2 (Sports), 3 (Business), 4 (Sci/Tech)
- `Title` — news headline
- `Description` — short article summary

---

## 3. Methodology — NLP Pipeline

```
Raw CSV → Combine Title + Description → Preprocess → TF-IDF → Naive Bayes → Prediction
```

### Step 1 — Data Loading
- Load CSV with `pandas`
- Map numeric class index → human-readable label
- Concatenate `Title` and `Description` into a single `text` feature

### Step 2 — Text Preprocessing
- Lowercase all text
- Remove punctuation, digits, and special characters
- Filter out English stopwords (built-in set, no external dependency)
- Drop tokens shorter than 3 characters

### Step 3 — Feature Extraction
- **TF-IDF Vectorizer** with:
  - `max_features = 30,000`
  - `ngram_range = (1, 2)` — unigrams and bigrams
  - `sublinear_tf = True` — log-dampened term frequency

### Step 4 — Model Training
- **Multinomial Naive Bayes** with `alpha = 0.1` (Laplace smoothing)
- 80 / 20 stratified train-test split
- Model and vectorizer persisted to `model.pkl` / `vectorizer.pkl`

### Step 5 — Prediction
- User input preprocessed identically to training data
- Transformed via the saved TF-IDF vectorizer
- Model returns predicted class + full probability distribution

---

## 4. Model — Naive Bayes Explained

Multinomial Naive Bayes applies **Bayes' theorem** under the "naive" assumption that features (words) are conditionally independent given the class:

```
P(Class | Text) ∝ P(Class) × ∏ P(word_i | Class)
```

**Why it works for text classification:**
- Text features (TF-IDF weights) are high-dimensional and sparse — NB handles this naturally
- The independence assumption, while unrealistic in theory, holds well enough in practice
- Training is extremely fast (single pass through data)
- Performs competitively with complex models on short text and news datasets
- Probabilistic output provides calibrated confidence scores

---

## 5. Evaluation Results

Trained and evaluated on the AG News test set (7,600 samples, 80/20 split):

```
Accuracy: 89.34%

              precision    recall  f1-score   support
    Business       0.87      0.84      0.85       380
    Sci/Tech       0.86      0.88      0.87       380
      Sports       0.94      0.97      0.96       380
       World       0.91      0.88      0.90       380

    accuracy                           0.89      1520
```

**Confusion Matrix:**

|          | World | Sports | Business | Sci/Tech |
|----------|-------|--------|----------|----------|
| World    | 336   | 17     | 18       | 9        |
| Sports   | 4     | 370    | 3        | 3        |
| Business | 13    | 5      | 318      | 44       |
| Sci/Tech | 16    | 2      | 28       | 334      |

Sports is the easiest to classify (94% F1). Business and Sci/Tech share vocabulary causing the most confusion.

---

## 6. How to Run the Project

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone / download the project
cd news_classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure dataset is in place
#    Place test.csv (and optionally train.csv) inside data/

# 4. Train the model (first run only)
python model.py data/test.csv

# 5. Start the Flask server
python app.py
```

### Open the app
Visit **http://localhost:5000** in your browser.

---

## 7. Project Structure

```
news_classifier/
├── app.py               ← Flask server & routes
├── model.py             ← NLP pipeline (preprocess, train, predict)
├── requirements.txt
├── model.pkl            ← saved Naive Bayes model (auto-generated)
├── vectorizer.pkl       ← saved TF-IDF vectorizer (auto-generated)
├── data/
│   └── test.csv         ← AG News test set
├── templates/
│   └── index.html       ← frontend page
└── static/
    ├── style.css        ← dark editorial theme
    └── script.js        ← frontend JS logic
```

---

## 8. Future Improvements

| Idea | Expected Impact |
|------|----------------|
| Train on full 120k training set | +2–3% accuracy |
| Swap NB for Logistic Regression or LinearSVC | +3–5% accuracy |
| Add BERT / DistilBERT embeddings | +8–12% accuracy |
| Use the article body (not just title+desc) | Better generalisation |
| Add REST API with proper auth | Production-readiness |
| Add batch classification (CSV upload) | Power-user feature |
| Confidence threshold + "uncertain" fallback | More honest predictions |

---

*Built with Flask · scikit-learn · pandas · AG News Dataset*
