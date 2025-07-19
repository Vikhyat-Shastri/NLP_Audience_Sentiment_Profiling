from tensorflow.keras.models import load_model
import joblib
import numpy as np
from src.utils import preprocess_text  # Assuming a utility function for text preprocessing

class SentimentPredictor:
    def __init__(self, model_path, vectorizer_path, data_processing_path):
        self.model = load_model(model_path)  # Load the LSTM model
        self.vectorizer = joblib.load(vectorizer_path)  # Load the TF-IDF vectorizer
        self.data_processing = joblib.load(data_processing_path)  # Load the data processing functions

    def predict(self, review):
        processed_review = self.data_processing(review)  # Preprocess the review
        transformed_review = self.vectorizer.transform([processed_review])  # Transform using TF-IDF
        prediction = self.model.predict(transformed_review)  # Make prediction
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"  # Interpret prediction
        return sentiment

if __name__ == "__main__":
    # Example usage
    model_path = '../models/lstm_model.h5'
    vectorizer_path = '../models/tfidf_vectorizer.pkl'
    data_processing_path = '../models/data_processing.pkl'

    predictor = SentimentPredictor(model_path, vectorizer_path, data_processing_path)

    # Test with a sample review
    sample_review = "This movie was fantastic! I loved every moment of it."
    result = predictor.predict(sample_review)
    print(f"Review: {sample_review}\nSentiment: {result}")