import unittest
import fakeredis
import app

from unittest.mock import MagicMock
from unittest.mock import patch
from redis import StrictRedis

class PositiveModel():
    def predict(phrase):
        return 2

class NeutralModel():
    def predict(phrase):
        return 1

class NegativeModel():
    def predict(phrase):
        return 0


class TestGetSentiment(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch('app.get_ml_model')
    def test_get_sentiment_positive(self, mock_get_model):
        mock_get_model.return_value = PositiveModel
        result = app.get_sentiment('I love cheese')
        self.assertEqual(result, 2)

    @patch('app.get_ml_model')
    def test_get_sentiment_negative(self, mock_get_model):
        mock_get_model.return_value = NegativeModel
        result = app.get_sentiment('I hate cheese')
        self.assertEqual(result, 0)

    @patch('app.get_ml_model')
    def test_get_sentiment_neutral(self, mock_get_model):
        mock_get_model.return_value = NeutralModel
        result = app.get_sentiment('I cheese')
        self.assertEqual(result, 1)

if __name__ == '__main__':
    unittest.main()
