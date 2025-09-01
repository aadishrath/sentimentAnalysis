import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from ..config import DATASETS

# Define correct column names
columns = ["sentiment", "id", "date", "query", "user", "text"]

# Load dataset
print("Loading Sentiment140 dataset...")
df = pd.read_csv(DATASETS['sentiment140'], encoding="ISO-8859-1", names=columns)

print("Extracting texts and labels...")
df["label"] = df["sentiment"] - 1  # Convert 1–5 to 0–4
dataset = df[["text", "label"]]

# Load tokenizer and model
print("Load tokenizer and model...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

tokenized = dataset.map(tokenize, batched=True)

# Training setup
args = TrainingArguments(
    output_dir="models/transformer",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized
)

# Train and save
print("Training classifier...")
trainer.train()

print("Saving model artifacts...")
model.save_pretrained("trained_models/transformer")
tokenizer.save_pretrained("trained_models/transformer")

print("✅ Transformer training complete.")
