# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# import joblib
# from tqdm import tqdm
# from ..config import DATASETS

# # Define correct column names
# columns = ["sentiment", "id", "date", "query", "user", "text"]

# # Load dataset
# print("Loading Sentiment140 dataset...")
# df = pd.read_csv(DATASETS['sentiment140'], encoding="ISO-8859-1", names=columns)

# print("Extracting texts and labels...")
# texts = list(tqdm(df["text"], desc="Processing texts"))
# labels = list(tqdm(df["sentiment"], desc="Processing labels"))

# # Convert text to TF-IDF vectors
# print("Vectorizing text with TF-IDF...")
# vectorizer = TfidfVectorizer(max_features=5000)
# X = vectorizer.fit_transform(texts)

# # Train SVM classifier
# print("Training SVM classifier...")
# svm = SVC(probability=True)
# svm.fit(X, labels)

# # Save vectorizer and model
# print("Saving model artifacts...")
# joblib.dump(vectorizer, "trained_models/tfidf_vectorizer.pkl")
# joblib.dump(svm, "trained_models/svm_model.pkl")

# print("✅ TF-IDF + SVM training complete.")




import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import joblib

# Try importing cuML's GPU SVM
try:
    from cuml.svm import SVC as GPU_SVC
    use_gpu = True
    print("GPU detected: using cuML's GPU-accelerated SVM.")
except ImportError:
    from sklearn.svm import SVC as CPU_SVC
    use_gpu = False
    print("cuML not available: falling back to scikit-learn's CPU SVM.")

from ..config import DATASETS

# Define correct column names
columns = ["sentiment", "id", "date", "query", "user", "text"]

# Load dataset
print("Loading Sentiment140 dataset...")
df = pd.read_csv(DATASETS['sentiment140'], encoding="ISO-8859-1", names=columns)

# Extract texts and labels
print("Extracting texts and labels...")
texts = list(tqdm(df["text"], desc="Processing texts"))
labels = list(tqdm(df["sentiment"], desc="Processing labels"))

# Convert text to TF-IDF vectors
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Train SVM classifier
print("Training SVM classifier...")
if use_gpu:
    # cuML expects dense float32 arrays
    X_dense = X.astype("float32").toarray()
    y = pd.Series(labels).astype("int32")
    svm = GPU_SVC(probability=True)
    svm.fit(X_dense, y)
else:
    svm = CPU_SVC(probability=True)
    svm.fit(X, labels)

# Save vectorizer and model
print("Saving model artifacts...")
joblib.dump(vectorizer, "trained_models/tfidf_vectorizer.pkl")
joblib.dump(svm, "trained_models/svm_model.pkl")

print("TF-IDF + SVM training complete.")