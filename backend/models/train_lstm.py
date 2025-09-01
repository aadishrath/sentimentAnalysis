import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import joblib
from ..config import DATASETS

# Define correct column names
columns = ["sentiment", "id", "date", "query", "user", "text"]

# Load dataset
print("Loading Sentiment140 dataset...")
df = pd.read_csv(DATASETS['sentiment140'], encoding="ISO-8859-1", names=columns)

print("Extracting texts and labels...")
texts = df["text"].str.lower()
labels = df["sentiment"] - 1  # Convert 1–5 to 0–4

# Build vocabulary
print("Building vocabulary...")
words = " ".join(texts).split()
vocab = {w: i+1 for i, (w, _) in enumerate(Counter(words).most_common())}
joblib.dump(vocab, "models/lstm_tokenizer.pkl")

# Tokenize and pad
def tokenize(text, max_len=100):
    tokens = [vocab.get(w, 0) for w in text.split()]
    return tokens[:max_len] + [0] * max(0, max_len - len(tokens))

print("Tokenizing...")
X = torch.tensor([tokenize(t) for t in texts.tolist()])
y = torch.tensor(labels.tolist())

# Define LSTM model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, 100)
        self.lstm = nn.LSTM(100, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

print("Building model...")
model = SentimentLSTM(len(vocab))
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train loop
print("Training classifier...")
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Save model
print("Saving model artifacts...")
joblib.dump(vocab, "trained_models/lstm_tokenizer.pkl")
torch.save(model, "trained_models/lstm_model.pt")

print("✅ LSTM training complete.")
