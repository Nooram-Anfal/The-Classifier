# The Classifier — Multi-Model News Intelligence System

## Overview

This project is an end-to-end Natural Language Processing (NLP) system that does two things: classifies news articles into one of four categories, and detects whether a news article is real or fake. It started as a single classification system and has since grown into a secured, dual-model application backed by user authentication and persistent storage.

The system provides:

- Multi-model news category classification
- Binary fake news detection
- JWT-based user authentication
- Per-user prediction history stored in MongoDB
- Interactive newspaper-style web interface
- Evaluation dashboards for both models, including confusion matrices and ROC curves
- Batch processing and CSV export

The goal is to demonstrate a complete machine learning lifecycle — data gathering, cleaning, feature engineering, model training, and evaluation — wrapped in a secure, production-style web application rather than a standalone notebook or script.

## Datasets

**News Category Classification**
Source: AG News Classification Dataset (Kaggle)
Categories: World, Sports, Business, Sci/Tech
Each record contains a title, description, and label. Title and description are combined before training.

**Fake News Detection**
Source: ISOT Fake News Dataset (Kaggle)
Two source files, True.csv and Fake.csv, combined into a single binary-labeled dataset (Real = 0, Fake = 1).

## Technologies Used

**Backend**
- Python
- FastAPI (migrated from an earlier Flask version)
- Uvicorn (ASGI server)
- scikit-learn
- pandas / numpy
- Motor (async MongoDB driver)
- python-jose (JWT signing and verification)
- passlib with bcrypt (password hashing)
- Pydantic (data validation)
- python-dotenv (environment configuration)

**Frontend**
- HTML / CSS / JavaScript
- Chart.js (ROC curve rendering)
- Font Awesome (icons)

**Database**
- MongoDB (Atlas), accessed asynchronously via Motor

## NLP Pipeline

Both models follow the same five-stage pipeline structure:

**1. Data Gathering**
Raw data loaded from CSV sources. For the fake news module, each record is validated against a Pydantic schema (title, text, subject, label) before entering the pipeline.

**2. Data Cleaning**
- Null and duplicate removal
- Minimum length filtering
- Lowercasing, punctuation stripping, stopword removal

**3. Feature Engineering**
- TF-IDF vectorization (unigrams and bigrams, sublinear term frequency scaling)
- For fake news detection specifically: two additional engineered scalar features, normalized text length and exclamation mark count, concatenated to the TF-IDF matrix using sparse matrix operations

**4. Model Training**
- News category classification: four models trained on a shared TF-IDF representation — Multinomial Naive Bayes, Logistic Regression, Linear SVC, and SGD Classifier
- Fake news detection: a single Logistic Regression classifier trained on the combined TF-IDF and scalar feature matrix

**5. Evaluation**
- Accuracy, Precision, Recall, F1-score for both models
- ROC-AUC and full ROC curve data for the fake news model
- Confusion matrices for both
- All evaluation metrics are computed once during training and persisted, then served live through dedicated API endpoints

## Model Performance

**News Category Classification**

| Model | Accuracy |
|---|---|
| Naive Bayes | 89.34% |
| Linear SVC | 87.96% |
| SGD Classifier | 87.83% |
| Logistic Regression | 87.57% |

**Fake News Detection**

| Metric | Score |
|---|---|
| Accuracy | 99.24% |
| Precision | 99.42% |
| Recall | 98.88% |
| F1-Score | 99.15% |
| ROC-AUC | 0.9995 |

## Authentication & Security

The fake news detection module is gated behind a full authentication system rather than left open:

- **Registration** — usernames and emails are checked for uniqueness, passwords are hashed with bcrypt before storage, raw passwords are never persisted
- **Login** — OAuth2-compliant token endpoint; on success, returns a signed JWT with a short expiry window
- **Protected routes** — prediction, metrics, and history endpoints for the fake news module all require a valid JWT, enforced through FastAPI's dependency injection system (`Depends`)
- **Session handling** — the frontend stores the JWT client-side and attaches it as a Bearer token on every protected request; expired or invalid tokens are rejected with HTTP 401 and the user is redirected to log in again

## Features

**News Category Classification**
- Single text classification with model selection and confidence scores
- Keyword highlighting (top TF-IDF terms influencing the prediction)
- Batch classification (up to 50 headlines at once) with CSV export
- Evaluation dashboard: accuracy comparison, confusion matrix, per-class metrics

**Fake News Detection** *(requires login)*
- Real/Fake verdict with confidence score and contributing keywords
- Evaluation dashboard: accuracy, precision, recall, F1 metric cards
- Confusion matrix visualization
- ROC curve rendered live with Chart.js
- Personal prediction history, persisted in MongoDB and retrievable per user

## User Interface

The interface keeps its original newspaper broadsheet aesthetic across both modules:

- Black and white ink theme
- Fraktur masthead with serif body typography
- Multi-tab layout: Classify, Batch, Evaluation, Fake News, Methodology
- Icon-based design, no emojis

## Project Structure

```
project/
├── app.py                      FastAPI application and all routes
├── model.py                    News category classification pipeline
├── fake_news_model.py          Fake news detection pipeline
├── auth.py                     JWT authentication and user management
├── requirements.txt
├── .env.example
│
├── models/
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── linear_svc.pkl
│   ├── sgd.pkl
│   ├── vectorizer.pkl
│   ├── evaluation.json
│   ├── fake_news_model.pkl
│   ├── fake_news_vectorizer.pkl
│   └── fake_news_evaluation.json
│
├── templates/
│   ├── index.html
│   └── login.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── data/
    └── dataset files (test.csv, True.csv, Fake.csv)
```

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# fill in MONGODB_URI and SECRET_KEY
```

**3. Train models (if not already trained)**
```bash
python model.py data/test.csv
python fake_news_model.py data/True.csv data/Fake.csv
```

**4. Run the application**
```bash
uvicorn app:app --port 5000
```

**5. Open in browser**
```
http://localhost:5000
```

Register an account and log in to access the Fake News module.

## API Endpoints

**Public**
- `GET /api/models` — available news category models
- `GET /api/evaluation` — news category evaluation metrics
- `POST /api/predict` — single news category prediction
- `POST /api/batch` — batch news category classification

**Authentication**
- `POST /auth/register` — create a new account
- `POST /auth/token` — log in, returns JWT
- `GET /auth/me` — current authenticated user

**Protected (requires JWT)**
- `POST /api/fake-news/predict` — fake news verdict, logged to MongoDB
- `GET /api/fake-news/metrics` — fake news evaluation metrics and ROC curve data
- `GET /api/fake-news/history` — current user's last 20 predictions
- `POST /api/retrain` — retrain news category models

## Methodology Summary

Both pipelines convert raw text into TF-IDF vectors before training. For news categorization, four classifiers are trained on the same feature space to allow direct comparison; Naive Bayes performs best, largely due to its efficiency on high-dimensional sparse data despite its independence assumption. For fake news detection, Logistic Regression was chosen for its strong performance on sparse text features and its ability to produce well-calibrated probabilities, which matter when displaying confidence scores to end users.

## Future Improvements

- Transformer-based models (BERT / DistilBERT) for fake news detection as a comparison baseline
- Real-time news API integration for live classification
- Refresh tokens and session revocation
- Rate limiting on public and authenticated endpoints
- Lemmatization and expanded n-gram features
- Cloud deployment (Render / Railway for the API, Vercel-style static hosting if migrated to a separate frontend)

## Conclusion

This project demonstrates a complete machine learning lifecycle, from raw data to a secured, user-facing application. It combines classical NLP techniques across two distinct tasks, a properly authenticated API layer, and persistent storage, while keeping the interaction simple enough that the underlying complexity stays out of the user's way.
