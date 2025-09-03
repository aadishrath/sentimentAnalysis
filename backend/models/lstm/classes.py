import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Define LSTM model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, 100)
        self.lstm = nn.LSTM(100, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = [self.vocab.get(w, 0) for w in self.texts[idx].split()]
        padded = tokens[:self.max_len] + [0] * max(0, self.max_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
