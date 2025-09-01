import torch
import torch.nn.functional as F
import joblib
from backend.config import LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH

# Load trained LSTM model and tokenizer vocabulary
lstm_model = torch.load(LSTM_MODEL_PATH)
lstm_model.eval()
vocab = joblib.load(LSTM_TOKENIZER_PATH)

def tokenize(text, max_len=100):
    # Convert text to list of token indices using vocab
    tokens = [vocab.get(w, 0) for w in text.split()]
    padded = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
    return torch.tensor(padded).unsqueeze(0)  # Add batch dimension

def predict_lstm(text):
    input_tensor = tokenize(text)
    with torch.no_grad():
        output = lstm_model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        score = torch.argmax(probs).item() + 1  # Convert 0–4 to 1–5
        confidence = torch.max(probs).item()
    return score, confidence
