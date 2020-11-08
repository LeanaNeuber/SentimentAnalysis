from flask import Flask, request, render_template
from redis import RedisError, StrictRedis
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import column_or_1d
from sklearn.pipeline import Pipeline

"""
This files is the entry point for the user by serving our html templates.
It also trains our sentiment analysis model that is then used for prediction.
"""
app = Flask(__name__)

def train_model():
    """
    This function trains the sentiment model and stores it to redis
    """
    print('Loading the dataset...')
    dataset = pd.read_csv('app/Sentiment_Reviews.csv',index_col=0)
    X = dataset[['Reviews']]
    y = dataset[['Sentiment']]
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = (le.transform(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

    print('Training the model...')
    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer(use_idf=False)), ('clf-svm', SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.001, random_state=42, max_iter=20))])
    text_clf_svm = text_clf_svm.fit(X_train['Reviews'], y_train)

    print('Storing model to redis...')
    pickled_model = pickle.dumps(text_clf_svm)
    try:
        redis_client.set('ml_model', pickled_model)
    except RedisError as e:
        print('Storing the model was not successful and threw an error.')
        print(e)


def get_sentiment(phrase):
    """
    This function takes a phrase and returns a sentiment.
    It does so by predicting with the pretrained model.
    See above function for that.
    Returns: 1 for neutral, 2 for positive, 0 for negative
    """
    print('Getting the sentiment...')
    try:
        pickled_model = redis_client.get('ml_model')
        model = pickle.loads(pickled_model)
        prediction = model.predict([phrase])
        return int(prediction)
    except RedisError as e:
        print('Getting the sentiment was not successful and threw an error.')
        print(e)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    This is the entry point for our web interface.
    For each POST request sent via our frontend, we take the value as a phrase
    and get the sentiment for it. We then return a new page according
    to the sentiment. The templates can be found in the templates folder.
    """
    if request.method == 'POST':
        phrase = request.form
        if phrase['form_type'] == 'get_sentiment':
            sent = get_sentiment(phrase['phrase'])
            if sent == 1:
                return render_template('index_neutral.html')
            elif sent == 2:
                return render_template('index_positive.html')
            elif sent == 0:
                return render_template('index_negative.html')
    return render_template('index.html')

if __name__ == '__main__':
    redis_client = StrictRedis(host='redis', port=6379)
    train_model()
    app.run(host='0.0.0.0')
