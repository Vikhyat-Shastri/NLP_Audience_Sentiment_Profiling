# NLP Audience Sentiment Profiling

A scalable, production-ready project for sentiment analysis on IMDB reviews using both classical machine learning and deep learning (LSTM). This repository is modular, recruiter-friendly, and easy to extend for other datasets or use cases.

---

## 🚀 Project Overview

This project demonstrates end-to-end sentiment analysis:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering (TF-IDF, word clouds)
- Model training (Logistic Regression, Naive Bayes, SVM, LSTM)
- Hyperparameter tuning
- Predictive system for real-world reviews

---

## 📁 Folder Structure

```
NLP_Audience_Sentiment_Profiling/
│
├── data/                # Raw datasets
│   └── IMDB_Dataset.csv
│
├── notebooks/           # Jupyter notebooks for workflow and demo
│   └── sentiment_analysis.ipynb
│
├── src/                 # Modular Python scripts
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── predict.py
│   └── utils.py
│
├── models/              # Saved models and transformers
│   ├── logistic_regression_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── data_processing.pkl
│   └── lstm_model.h5
│
├── tests/               # Unit tests
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_train_models.py
│   └── test_predict.py
│
├── requirements.txt     # Python dependencies
├── setup.py             # (Optional) For packaging
└── README.md            # Project documentation
```

---

## 🛠️ Setup Instructions

1. **Clone the repository:**

   ```
   git clone https://github.com/Shastri-727/NLP_Audience_Sentiment_Profiling.git
   cd NLP_Audience_Sentiment_Profiling
   ```

2. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

3. **Download NLTK data (if running scripts):**

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

---

## 📓 How to Run

- **For exploration and demo:**  
  Open and run `notebooks/sentiment_analysis.ipynb` in Jupyter or VS Code.

- **For production/batch:**  
  Run scripts in `src/` as needed:

  ```
  python src/train_models.py
  python src/predict.py
  ```

---

## 🧪 Testing

Run unit tests from the `tests/` folder:

```
pytest tests/
```

---

## ✨ Features

- Clean, modular codebase
- Classical ML and Deep Learning models
- Hyperparameter tuning
- Predictive system for new reviews
- Visualizations (EDA, word clouds)
- Easy to extend for other datasets

---

## 📬 Contact

For questions or collaboration, reach out via [GitHub Issues](https://github.com/Shastri-727/NLP_Audience_Sentiment_Profiling/issues).

---

## 📄 License

MIT License
