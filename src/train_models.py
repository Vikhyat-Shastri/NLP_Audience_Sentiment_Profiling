from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import pandas as pd
import numpy as np

def train_models(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear'),
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC()
    }

    results = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        results[model_name] = {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'classification_report': classification_report(y_test, predictions)
        }
        joblib.dump(model, f'../models/{model_name.lower().replace(" ", "_")}_model.pkl')

    return results

def train_lstm_model(X, y, tokenizer, max_length=200):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    X_padded = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_length)

    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=max_length),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_padded, y, epochs=5, batch_size=64, validation_split=0.2)

    model.save('../models/lstm_model.h5')

    return model

def main():
    df = pd.read_csv('../data/IMDB_Dataset.csv')
    X = df['review']
    y = df['sentiment'].replace({'positive': 1, 'negative': 0})

    results = train_models(X, y)
    print(results)

if __name__ == "__main__":
    main()