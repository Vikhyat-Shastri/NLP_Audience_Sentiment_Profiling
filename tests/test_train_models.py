import unittest
from src.train_models import train_logistic_regression, train_naive_bayes, train_svm, train_lstm
import pandas as pd

class TestTrainModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv('data/IMDB_Dataset.csv')
        cls.X = cls.df['review']
        cls.y = cls.df['sentiment'].replace({'positive': 1, 'negative': 0})

    def test_train_logistic_regression(self):
        model = train_logistic_regression(self.X, self.y)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_train_naive_bayes(self):
        model = train_naive_bayes(self.X, self.y)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_train_svm(self):
        model = train_svm(self.X, self.y)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_train_lstm(self):
        model = train_lstm(self.X, self.y)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

if __name__ == '__main__':
    unittest.main()