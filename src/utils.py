def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_model(model, file_path):
    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    import joblib
    return joblib.load(file_path)

def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r'<br />', '', text)
    text = re.sub(r'https?\\S+|www\\.\\S+', '', text)
    text = re.sub(r'@\\w+|#\\w+', '', text)
    text = re.sub(r'[^\\w\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def get_sentiment_label(label):
    return "Positive" if label == 1 else "Negative"