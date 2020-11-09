import unittest
import os
import requests

class FlaskTests(unittest.TestCase):
    def setUp(self):
        os.environ['NO_PROXY'] = '0.0.0.0'
        self.get_sentiment = {
            'positive_phrase': 'I love you',
            'neutral_phrase' : 'what is going on ?',
            'negative_phrase': 'I hate you'
        }
        pass
    
    def tearDown(self):
        pass


    def test_a_index(self):
        responce = requests.get('http://localhost:5000')
        self.assertEqual(responce.status_code, 200)

    ######### Positive statement	
    def test_b_get_sentiment(self):

        params = {
            'phrase': self.get_sentiment['positive_phrase'],
            "form_type": "get_sentiment"
        }
        responce = requests.post('http://localhost:5000', data=params)
        self.assertEqual(responce.status_code, 200)
        self.assertIn('POSITIVE! YAY!'.encode(),responce.content)

    ######### Neutral statement
    def test_c_get_sentiment(self):

        params = {
            'phrase': self.get_sentiment['neutral_phrase'],
            "form_type": "get_sentiment"
        }
        responce = requests.post('http://localhost:5000', data=params)
        self.assertEqual(responce.status_code, 200)
        self.assertIn('NEUTRAL! LAME!'.encode(),responce.content)

    ######### Negative statement
    def test_d_get_sentiment(self):

        params = {
            'phrase': self.get_sentiment['negative_phrase'],
            "form_type": "get_sentiment"
        }
        responce = requests.post('http://localhost:5000', data=params)
        self.assertEqual(responce.status_code, 200)
        self.assertIn('NEGATIVE! SO SAD!'.encode(),responce.content)



if __name__ == '__main__':
    unittest.main()