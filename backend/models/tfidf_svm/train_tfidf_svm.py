import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
from tqdm import tqdm

from backend.config import DATASETS, MODEL_DIR
# from ..config import DATASETS, MODEL_DIR

if __name__ == "__main__":
    # Define correct column names
    columns = ["sentiment", "id", "date", "query", "user", "text"]
    vctrzer_dir = MODEL_DIR + '/tfidf_vectorizer.pkl'
    model_dir = MODEL_DIR + '/svm_model.pkl'

    # Load dataset
    print("Loading Sentiment140 dataset...")
    df = pd.read_csv(DATASETS['sentiment140'], encoding="ISO-8859-1", names=columns)

    # Reduced to 3-class and slice for prototyping
    df = df[df["sentiment"].isin([0, 2, 4])].reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle before slicing to preserve class balance
    df = df[:100000]  # Reduced size for memory safety

    label_map = {0: 0, 2: 1, 4: 2}
    df["label"] = df["sentiment"].map(label_map)

    print("Extracting texts and labels...")
    texts = list(tqdm(df["text"], desc="Processing texts"))
    labels = list(tqdm(df["sentiment"], desc="Processing labels"))

    # Convert text to TF-IDF vectors
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    # Train SVM classifier
    print("Training SVM classifier...")
    svm = SVC(probability=True)
    svm.fit(X, labels)

    # Save vectorizer and model
    print("Saving model artifacts...")
    joblib.dump(vectorizer, vctrzer_dir)
    joblib.dump(svm, model_dir)

    print("✅ TF-IDF + SVM training complete.")


# # unable to get 'cuml.svm' installed, hence not able to check below code for GPU training
# import os
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC as CPU_SVC
# import joblib
# from tqdm import tqdm
# from ..config import DATASETS, MODEL_DIR

# # Try importing cuML's GPU SVM
# try:
#     from cuml.svm import SVC as GPU_SVC
#     use_gpu = True
#     print("🟢 GPU detected: using cuML's SVM.")
# except ImportError:
#     use_gpu = False
#     print("⚪️ cuML not available: falling back to scikit-learn's CPU SVM.")

# # Ensure directories exist
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Define column names
# columns = ["sentiment", "id", "date", "query", "user", "text"]
# vctrzer_dir = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
# model_dir = os.path.join(MODEL_DIR, "svm_model.pkl")

# # Load and preprocess dataset
# print("📥 Loading Sentiment140 dataset...")
# df = pd.read_csv(DATASETS['sentiment140'], encoding="ISO-8859-1", names=columns)

# # Optional: reduce to 3-class and slice for prototyping
# df = df[df["sentiment"].isin([0, 2, 4])].reset_index(drop=True)
# df = df[:10000]  # Memory-safe slice

# label_map = {0: 0, 2: 1, 4: 2}
# df["label"] = df["sentiment"].map(label_map)

# # Extract texts and labels
# print("🧹 Extracting texts and labels...")
# texts = list(tqdm(df["text"], desc="Processing texts"))
# labels = list(tqdm(df["label"], desc="Processing labels"))

# # TF-IDF vectorization
# print("✂️ Vectorizing text with TF-IDF...")
# vectorizer = TfidfVectorizer(max_features=5000)
# X = vectorizer.fit_transform(texts)

# # Train SVM
# print("🚀 Training SVM classifier...")
# if use_gpu:
#     X_dense = X.astype("float32").toarray()
#     y = pd.Series(labels).astype("int32")
#     svm = GPU_SVC(probability=True)
#     svm.fit(X_dense, y)
# else:
#     svm = CPU_SVC(probability=True)
#     svm.fit(X, labels)

# # Save artifacts
# print("💾 Saving model artifacts...")
# joblib.dump(vectorizer, vctrzer_dir)
# joblib.dump(svm, model_dir)

# print("✅ TF-IDF + SVM training complete.")
