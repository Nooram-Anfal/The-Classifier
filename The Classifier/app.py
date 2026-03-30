"""
app.py — Flask backend for News Category Classifier
"""

import os
from flask import Flask, render_template, request, jsonify
from model import load_data, train_model, load_model, predict, MODEL_PATH, VECTORIZER_PATH

app = Flask(__name__)

# ─── Load or train model on startup ──────────────────────────────────────────

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print("Loading existing model...")
    model, vectorizer = load_model()
else:
    print("No saved model found. Training now...")
    df = load_data("data/test.csv")
    model, vectorizer = train_model(df)

print("Model ready. Starting server...\n")

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) < 10:
        return jsonify({"error": "Please enter a longer news snippet (at least 10 characters)."}), 400

    result = predict(text, model, vectorizer)
    return jsonify(result)


@app.route("/retrain", methods=["POST"])
def retrain():
    """Optional: retrain the model via API call."""
    try:
        global model, vectorizer
        df = load_data("data/test.csv")
        model, vectorizer = train_model(df)
        return jsonify({"status": "Model retrained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
