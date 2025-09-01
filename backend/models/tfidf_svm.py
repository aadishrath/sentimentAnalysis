import joblib
from backend.config import TFIDF_VECTORIZER_PATH, SVM_MODEL_PATH

# Load pre-trained TF-IDF vectorizer and SVM model
vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)

def predict_svm(text):
    # Convert text to TF-IDF vector
    vec = vectorizer.transform([text])
    # Predict sentiment class
    pred = svm_model.predict(vec)[0]
    # Get confidence score from predicted probabilities
    prob = svm_model.predict_proba(vec)[0]
    confidence = max(prob)
    return int(pred), float(confidence)
