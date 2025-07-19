from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import re
import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()  # Convert to lowercase
        text = contractions.fix(text)  # Expand contractions
        text = re.sub(r'<br />', '', text)  # Remove <br /> tags
        text = re.sub(r'https?\\S+|www\\.\\S+', '', text)  # Remove URLs
        text = re.sub(r'@\\w+|#\\w+', '', text)  # Remove mentions & hashtags
        text = re.sub(r'[^\\w\\s]', '', text)  # Remove punctuation
        text = re.sub(r'\\s+', ' ', text).strip()  # Remove extra spaces
        return text

    def remove_stopwords(self, text):
        return " ".join([word for word in text.split() if word not in self.stop_words])

    def lemmatize_text(self, text):
        tokens = word_tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(token) for token in tokens])

    def preprocess(self, text):
        cleaned_text = self.clean_text(text)
        no_stopwords_text = self.remove_stopwords(cleaned_text)
        lemmatized_text = self.lemmatize_text(no_stopwords_text)
        return lemmatized_text

    def preprocess_dataframe(self, df, text_column):
        df[text_column] = df[text_column].apply(self.preprocess)
        return df