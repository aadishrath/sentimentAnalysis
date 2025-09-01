import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters except emojis and alphanumerics
    text = re.sub(r"[^\w\s\U0001F600-\U0001F64F]", "", text)
    
    # Convert to lowercase
    return text.lower()
