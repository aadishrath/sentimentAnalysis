import os

# Base directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = "backend"
MODEL_DIR = os.path.join(PROJECT_ROOT, BASE_DIR, "trained_models")
DATASET_DIR = os.path.join(PROJECT_ROOT, BASE_DIR, "dataset")


# TF-IDF + SVM
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")

# LSTM
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")
LSTM_TOKENIZER_PATH = os.path.join(MODEL_DIR, "lstm_tokenizer.pkl")

# Transformer
TRANSFORMER_MODEL_DIR = os.path.join(MODEL_DIR, "transformer")

# Dataset path
DATASETS = {  # created object to host multiple dataset, making this project more dynamic
    "sentiment140": os.path.join(DATASET_DIR, "sentiment140.csv"),
}


if not os.path.exists(DATASETS['sentiment140']):
    print(f"❌ Missing: {DATASETS['sentiment140']}")
    exit()
else:
    print(f"✅ Found: {DATASETS['sentiment140']}")