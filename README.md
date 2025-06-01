# Natural Language Processing for Audience Sentiment Profiling in Film Reviews

## Overview
This project focuses on performing sentiment analysis on IMDB movie reviews using various machine learning models. The primary goal is to classify movie reviews as positive or negative based on their textual content. Multiple models, including Logistic Regression, Naïve Bayes, Support Vector Machines (SVM), and Deep Learning, were evaluated. Logistic Regression achieved the highest accuracy at 89.4% and was selected as the best-performing model.
Sentiment analysis is a crucial application in Natural Language Processing (NLP) with widespread use in fields such as customer feedback analysis, social media monitoring, and market research. Understanding audience sentiment towards movies helps production studios, critics, and viewers gauge public opinion more effectively. This project demonstrates how machine learning techniques can be applied to process and analyze large-scale textual data efficiently.

## Features
- Processed 50,000+ IMDB movie reviews using NLP techniques, including text cleaning, tokenization, stopword removal, lemmatization, and TF-IDF vectorization.
- Implementation of multiple ML models including Logistic Regression, Naïve Bayes, Support Vector Machines (SVM), and LSTM-based Deep Learning models.
- Performance evaluation using accuracy, precision, recall, and F1-score.
- Exploratory Data Analysis (EDA), including word clouds, frequency distributions, and sentiment-based word analysis.
- Optimized deep learning architecture by training an LSTM-based Recurrent Neural Network (RNN) with word embeddings, achieving 87.2% accuracy.

## Dataset
The dataset used in this project consists of 50,000 IMDB movie reviews labeled as positive or negative. The data is sourced from Kaggle and can be accessed here: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The data is preprocessed before feeding it into the machine learning models.

## Installation
To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk tensorflow keras
```

## Usage
Run the Jupyter Notebook to execute the sentiment analysis pipeline:

```bash
jupyter notebook sentiment_analysis.ipynb
```

## Results
After evaluating different models, Logistic Regression provided the best accuracy (89.4%), making it the preferred model for sentiment classification in this project. The LSTM-based deep learning model achieved an accuracy of 87.2%.

## Future Improvements
- Further optimization of the deep learning model using attention mechanisms and transformer-based models.
- Expanding the dataset for better generalization.
- Deploying the model as a web application for real-time sentiment analysis.

## Contributions
Contributions are welcome! If you have any improvements or suggestions, feel free to fork this repository and submit a pull request.

