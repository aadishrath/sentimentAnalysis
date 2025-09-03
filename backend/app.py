from flask import Flask, request, jsonify
from backend.models.lstm.predict_lstm import predict_lstm
from backend.models.tfidf_svm.predict_tfidf_svm import predict_svm
from backend.models.transformer.predict_transformer import predict_transformer
from backend.utils.logger import log_input
from backend.utils.preprocess import clean_text
from flask_cors import CORS
import os
from .config import *

def validate_model_paths():
    paths = [
        TFIDF_VECTORIZER_PATH,
        SVM_MODEL_PATH,
        LSTM_MODEL_PATH,
        LSTM_TOKENIZER_PATH,
        os.path.join(TRANSFORMER_MODEL_DIR, "config.json")
    ]
    for path in paths:
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            exit()
        else:
            print(f"✅ Found: {path}")


# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    # Parse incoming JSON payload
    data = request.json
    model_type = data["model"]
    raw_text = data["text"]

    # Clean the input text (remove noise, lowercase, keep emojis)
    text = clean_text(raw_text)

    # Route to appropriate model based on user selection
    if model_type == "svm":
        score, confidence = predict_svm(text)
    elif model_type == "lstm":
        score, confidence = predict_lstm(text)
    elif model_type == "transformer":
        score, confidence = predict_transformer(text)
    else:
        return jsonify({"error": "Invalid model"}), 400

    # Log input and prediction for analytics
    log_input(raw_text, model_type, score, confidence)

    # Return sentiment score and confidence to frontend
    return jsonify({
        "sentiment": score,
        "confidence": confidence
    })


# Run the Flask server
if __name__ == "__main__":
    print('Validating file paths...')
    validate_model_paths()

    app.run(debug=True)
