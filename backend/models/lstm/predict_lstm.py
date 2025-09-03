import torch
import torch.nn.functional as F
import joblib
import os

from backend.config import LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH
from backend.models.lstm.classes import SentimentLSTM


# from backend.models.train_lstm import SentimentLSTM
# from ..config import LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH

# Check model path
if not os.path.exists(LSTM_MODEL_PATH):
    print(f"❌ Missing: {LSTM_MODEL_PATH}")
    exit()
else:
    print(f"✅ Found: {LSTM_MODEL_PATH}")


# Load trained LSTM model and tokenizer
# lstm_model = torch.load(LSTM_MODEL_PATH, map_location=torch.device("cpu"))
# lstm_model.eval()


vocab = joblib.load(LSTM_TOKENIZER_PATH)
model = SentimentLSTM(vocab_size=len(vocab))
model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Label mapping used during training
label_map = {0: "negative", 1: "neutral", 2: "positive"}

def tokenize(text, max_len=100):
    # Convert text to list of token indices using vocab
    tokens = [vocab.get(w, 0) for w in text.lower().split()]
    padded = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
    return torch.tensor(padded, dtype=torch.long).unsqueeze(0)  # Add batch dimension

def predict_lstm(text):
    input_tensor = tokenize(text)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
        confidence = torch.max(probs).item()
        label = label_map[pred_idx]
    return label, confidence
