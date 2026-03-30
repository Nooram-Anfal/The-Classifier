# 📰 News Classification System (Multi-Model NLP Project)

## 📌 Overview
This project is an end-to-end Natural Language Processing (NLP) system that classifies news articles into predefined categories using multiple machine learning models.

Unlike basic classifiers, this system provides:
- Multi-model comparison
- Interactive web interface
- Evaluation visualization
- Batch processing and export features

The goal is to demonstrate practical implementation of NLP pipelines along with real-world usability.

---

## 📊 Dataset
- Source: AG News Classification Dataset (Kaggle)
- Categories:
  - World
  - Sports
  - Business
  - Sci/Tech

Each record contains:
- Title
- Description
- Label

The title and description are combined for training.

---

## ⚙️ Technologies Used

### Backend
- Python
- Flask
- scikit-learn
- pandas
- numpy

### Frontend
- HTML
- CSS
- JavaScript
- Font Awesome (icons)

---

## 🧠 NLP Pipeline

1. **Data Preprocessing**
   - Lowercasing
   - Removing punctuation
   - Stopword removal

2. **Feature Extraction**
   - TF-IDF Vectorization

3. **Model Training**
   The system uses four machine learning models:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear Support Vector Classifier (Linear SVC)
   - SGD Classifier

4. **Prediction**
   - User input is processed and classified into one of the categories.

---

## 📈 Model Performance

| Model                | Accuracy |
|---------------------|---------|
| Naive Bayes         | 89.34%  |
| Logistic Regression | 87.57%  |
| Linear SVC          | 87.96%  |
| SGD Classifier      | 87.83%  |

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🚀 Features

### 🔍 Single Text Classification
- Enter a news headline or paragraph
- Select a model
- Get predicted category + confidence score

### 🧠 Model Selection
- Switch between 4 models dynamically
- Compare outputs instantly

### 🏷 Keyword Highlighting
- Displays top TF-IDF keywords influencing prediction

### 📊 Evaluation Dashboard
- Model comparison charts
- Confusion matrix visualization
- Per-class performance metrics

### 📦 Batch Classification
- Input multiple headlines (one per line)
- Classify all at once

### 📁 Export Results
- Download batch predictions as CSV file

---

## 🎨 User Interface

The interface is designed with a **newspaper broadsheet aesthetic**:
- Black & white ink theme
- Classic typography (Fraktur masthead + serif body)
- Multi-column layout
- Icon-based design (no emojis)

Sections include:
- Classification
- Batch Processing
- Evaluation
- Methodology

---

## 📁 Project Structure
project/
│
├── app.py
├── model.py
├── requirements.txt
│
├── models/
│ ├── naive_bayes.pkl
│ ├── logistic_regression.pkl
│ ├── linear_svc.pkl
│ ├── sgd.pkl
│ ├── vectorizer.pkl
│ └── evaluation.json
│
├── templates/
│ └── index.html
│
├── static/
│ ├── style.css
│ └── script.js
│
└── data/
└── dataset files


---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Train models (if not already trained)
python model.py
3. Run application
python app.py
4. Open in browser
http://localhost:5000
📌 API Endpoints
/api/predict → Single prediction
/api/batch → Batch classification
/api/evaluation → Model evaluation data
/api/models → Available models
🔍 Methodology Summary

The system uses TF-IDF to convert text into numerical vectors. Multiple classifiers are trained on the same feature space to compare performance.

Naive Bayes performs best due to:

Independence assumption
Efficiency with high-dimensional sparse data

Other models provide competitive results and allow comparative analysis.

🔮 Future Improvements
Add deep learning models (LSTM / BERT)
Real-time news API integration
User authentication & saved results
Improved preprocessing (lemmatization, n-grams)
Deployment on cloud platform
📎 Conclusion

This project demonstrates how classical NLP techniques combined with multiple machine learning models can produce a powerful, interactive, and user-friendly news classification system.

It balances:

Simplicity
Performance
Practical usability

---

## ⚠️ One blunt note (important)

Your project is now:
> **“multi-model evaluation system with UI”**

So if your README stayed basic, examiner would think:
> “UI copy ki hai, samajh nahi aya”

Now?
👉 It clearly shows you understand pipeline + models + system design.

---

## Next step

When you're ready:
👉 send me your **results / screenshots / outputs**

I’ll build you a **clean report (PDF-level content)** that actually scores marks, not just filler.