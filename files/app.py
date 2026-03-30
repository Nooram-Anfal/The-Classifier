"""
app.py — Flask backend, multi-model News Classifier
"""
import os
from flask import Flask, render_template, request, jsonify
from model import (load_data, train_all, load_all_models, models_exist,
                   predict_one, predict_batch, MODEL_DEFS, EVAL_PATH)
import json

app = Flask(__name__)

# ── Startup ───────────────────────────────────────────────────────────────────
if models_exist():
    print("Loading saved models...")
    MODELS, VECTORIZER, EVAL_DATA = load_all_models()
else:
    print("Training all models from scratch...")
    df = load_data("data/test.csv")
    MODELS, VECTORIZER, EVAL_DATA = train_all(df)
print("Ready.\n")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/models")
def get_models():
    return jsonify(MODEL_DEFS)

@app.route("/api/evaluation")
def get_evaluation():
    return jsonify(EVAL_DATA)

@app.route("/api/predict", methods=["POST"])
def predict_route():
    data      = request.get_json()
    text      = (data.get("text") or "").strip()
    model_key = data.get("model", "naive_bayes")

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) < 10:
        return jsonify({"error": "Please enter at least 10 characters."}), 400
    if model_key not in MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    result = predict_one(text, model_key, MODELS, VECTORIZER)
    return jsonify(result)

@app.route("/api/batch", methods=["POST"])
def batch_route():
    data      = request.get_json()
    raw       = (data.get("text") or "").strip()
    model_key = data.get("model", "naive_bayes")

    if not raw:
        return jsonify({"error": "No text provided."}), 400
    if model_key not in MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    lines   = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(lines) > 50:
        return jsonify({"error": "Batch limit is 50 lines."}), 400

    results = predict_batch(lines, model_key, MODELS, VECTORIZER)
    return jsonify(results)

@app.route("/api/retrain", methods=["POST"])
def retrain():
    global MODELS, VECTORIZER, EVAL_DATA
    try:
        df = load_data("data/test.csv")
        MODELS, VECTORIZER, EVAL_DATA = train_all(df)
        return jsonify({"status": "Retrained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)