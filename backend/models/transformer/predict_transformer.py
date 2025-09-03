from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

from backend.config import TRANSFORMER_MODEL_DIR
# from ..config import TRANSFORMER_MODEL_DIR

# Load fine-tuned Transformer model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
transformer_model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
transformer_model.eval()

def predict_transformer(text):
    # Tokenize input text for Transformer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze()
        score = torch.argmax(probs).item() + 1  # Convert 0–4 to 1–5
        confidence = torch.max(probs).item()
    return score, confidence
