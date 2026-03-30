# 📰 The Classifier

A multi-model NLP-powered news classification system with a newspaper-style interface.

This project classifies news articles into four categories — **World, Sports, Business, and Sci/Tech** — using classical machine learning models and an interactive web interface.

---

## What This Project Does

The system provides two main ways to classify news:

### Single Article Classification
- Paste a news article or paragraph
- The system predicts the category
- Displays:
  - Top predicted category
  - Confidence percentage
  - All category probabilities (ranked)

This gives a clear view of how confident the model is and what alternatives it considered.

---

### Batch Headline Classification
- Paste up to **50 headlines** (one per line)
- The system processes all entries at once
- Each headline is classified individually

Useful for testing multiple inputs quickly instead of repeating manual steps.

---

## Models Used

The system compares multiple machine learning models:

- Naive Bayes  
- Logistic Regression  
- Linear SVC  
- SGD Classifier  

All models are trained on the same TF-IDF feature space to ensure fair comparison.

---

## Performance Snapshot

| Model                | Accuracy |
|---------------------|---------|
| Naive Bayes         | 89.28%  |
| Linear SVC          | 88.09%  |
| SGD Classifier      | 87.70%  |
| Logistic Regression | 87.57%  |

Key observation:
- **Naive Bayes performs best overall**
- Sports category is easiest to classify
- Business and Sci/Tech show some overlap

---

## How It Works

1. News text is cleaned (lowercase, punctuation removed)
2. Converted into numerical form using **TF-IDF**
3. Passed through selected model
4. Model returns category probabilities
5. Results are displayed in a structured format

---

## User Interface

The interface is designed like a **newspaper broadsheet**:
- Black & white theme (ink + paper style)
- Structured layout with clear sections
- Icon-based UI (no emojis)
- Multi-tab navigation:
  - Classification
  - Batch Processing
  - Evaluation
  - Methodology

The goal was to make the system feel like a real news tool, not just a model demo.

---

## Features

- Single article classification  
- Batch headline classification (up to 50 inputs)  
- Multi-model comparison  
- Confidence score display  
- Keyword-based influence hints (TF-IDF driven)  
- Evaluation dashboard (accuracy, confusion matrix, F1-score)  
- CSV export for batch results  

---

## Project Structure



project/
│
├── app.py
├── model.py
├── requirements.txt
│
├── models/
├── templates/
├── static/
└── data/


---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python model.py
python app.py
````

Then open:

```
http://localhost:5000
```

---

## Why This Project Matters

This project shows that:

* Classical NLP methods are still highly effective
* Multiple models provide better insight than a single one
* A good interface turns a model into a usable tool

---

## Future Improvements

* Add deep learning models (BERT, LSTM)
* Integrate real-time news APIs
* Deploy as a web application
* Improve preprocessing with lemmatization

---
