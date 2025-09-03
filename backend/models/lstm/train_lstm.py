import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import joblib
# from lstm.classes import SentimentLSTM, SentimentDataset

from backend.config import DATASETS, MODEL_DIR
from backend.models.lstm.classes import SentimentLSTM, SentimentDataset
# from ..config import DATASETS, MODEL_DIR


if __name__ == "__main__":
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define correct column names
    columns = ["sentiment", "id", "date", "query", "user", "text"]
    tknizer_dir = MODEL_DIR + '/lstm_tokenizer.pkl'
    model_dir = MODEL_DIR + '/lstm_model.pt'

    # Load dataset
    print("Loading Sentiment140 dataset...")
    df = pd.read_csv(DATASETS['sentiment140'], encoding="ISO-8859-1", names=columns)

    # Slicing dataset for prototyping
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle before slicing to preserve class balance
    df = df[:100000]  # Reduced size for memory safety

    print("Extracting texts and labels...")
    texts = df["text"].str.lower().tolist()
    label_map = {0: 0, 2: 1, 4: 2}
    labels = df["sentiment"].map(label_map).tolist()


    # Build vocabulary
    print("Building vocabulary...")
    words = " ".join(texts).split()
    vocab = { w: i+1 for i, (w, _) in enumerate(Counter(words).most_common()) }
    joblib.dump(vocab, tknizer_dir)

    # DataLoader
    print("📦 Preparing batches...")
    dataset = SentimentDataset(texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Initialize model
    print("Building model...")
    model = SentimentLSTM(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    print("Training classifier...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    # Save model
    print("Saving model artifacts...")
    torch.save(model.to('cpu').state_dict(), model_dir)

    print("✅ LSTM training complete.")
