import csv
from datetime import datetime

def log_input(text, model, score, confidence):
    # Append input and prediction to analytics.csv for later analysis
    with open("analytics.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), model, text, score, confidence])
