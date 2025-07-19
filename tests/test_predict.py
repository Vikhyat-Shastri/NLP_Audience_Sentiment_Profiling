import pytest
import joblib
from src.predict import predict_sentiment, predict_sentiment_LSTM

@pytest.fixture(scope='module')
def setup_models():
    # Load the logistic regression model and data processing function
    logreg_model = joblib.load('models/logistic_regression_model.pkl')
    data_processing = joblib.load('models/data_processing.pkl')
    return logreg_model, data_processing

def test_predict_sentiment_positive(setup_models):
    logreg_model, data_processing = setup_models
    review = "Absolutely loved this movie! The storyline was engaging."
    processed_review = data_processing(review)
    prediction = predict_sentiment(processed_review)
    assert prediction == "Positive"

def test_predict_sentiment_negative(setup_models):
    logreg_model, data_processing = setup_models
    review = "This was a complete waste of time."
    processed_review = data_processing(review)
    prediction = predict_sentiment(processed_review)
    assert prediction == "Negative"

def test_predict_sentiment_LSTM_positive(setup_models):
    logreg_model, data_processing = setup_models
    review = "An excellent film with brilliant acting."
    prediction = predict_sentiment_LSTM(review)
    assert prediction == "positive"

def test_predict_sentiment_LSTM_negative(setup_models):
    logreg_model, data_processing = setup_models
    review = "I really regret spending money on this."
    prediction = predict_sentiment_LSTM(review)
    assert prediction == "negative"