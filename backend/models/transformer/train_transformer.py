import pandas as pd
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

from backend.config import DATASETS, MODEL_DIR
# from ..config import DATASETS, MODEL_DIR


if __name__ == "__main__":
    # Define correct column names
    columns = ["sentiment", "id", "date", "query", "user", "text"]
    transformer_dir = MODEL_DIR + '/transformer'

    # Load dataset
    print("Loading Sentiment140 dataset...")
    df = pd.read_csv(DATASETS['sentiment140'], encoding="ISO-8859-1", names=columns)

    # Slicing for prototyping
    df = df[df["sentiment"].isin([0, 2, 4])].reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle before slicing to preserve class balance
    df = df[:100000]  # Reduced size for memory safety

    print("Extracting texts and labels...")
    label_map = {0: 0, 2: 1, 4: 2}
    df["label"] = df["sentiment"].map(label_map)
    # dataset = Dataset.from_pandas(df[["text", "label"]])

    # Split into train and validation
    train_df, val_df = train_test_split(df[["text", "label"]], test_size=0.2, random_state=42)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))


    # Load tokenizer and model
    print("Load tokenizer and model...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Tokenize dataset
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    print("Tokenizing...")
    # tokenized = dataset.map(tokenize, batched=True)
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    # Define metrics
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "precision": precision.compute(predictions=preds, references=labels, average="macro")["precision"],
            "recall": recall.compute(predictions=preds, references=labels, average="macro")["recall"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        }


    # Training setup
    args = TrainingArguments(
        output_dir="models/transformer",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train and save
    print("Training classifier...")
    trainer.train()

    print("Saving model artifacts...")
    model.save_pretrained(transformer_dir)
    tokenizer.save_pretrained(transformer_dir)

    print("✅ Transformer training complete.")
