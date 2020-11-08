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
app = Flask(__name__)
# ML code for sentiment analysis, could also be in an extra class
def train_model():
    # do something to train model etc.
    print('###### Load dataset')
    dataset = pd.read_csv('app/Sentiment_Reviews.csv',index_col=0)
    X = dataset[['Reviews']]
    y = dataset[['Sentiment']]
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = (le.transform(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
    print('###### train model')
    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer(use_idf=False)), ('clf-svm', SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.001, random_state=42, max_iter=20))])
    text_clf_svm = text_clf_svm.fit(X_train['Reviews'], y_train)
    # store model to redis with pickle
    print('###### store model to redis with pickle')
    pickled_model = pickle.dumps(text_clf_svm)
    try:
        print('###### redis client set start TEST')
        redis_client.set('test','test_value')
        print('###### redis client set end TEST')
        print('###### redis client set start')
        redis_client.set('ml_model', pickled_model)
        print('###### redis client set end')
        return
    except RedisError:
        return "###### fail1"

# Get sentiment for phrase for pretrained model
def get_sentiment(phrase):
    # Get stored model from redis
    try:
        print('###### redis client')
        client_value = redis_client.get('ml_model')
        print('###### unpacked model')
        unpacked_model = pickle.loads(client_value)
        #apply model on phrase
        print('###### apply model')
        result = unpacked_model.predict([phrase])
        return int(result)
    except RedisError:
        return "###### fail2"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        phrase = request.form
        if phrase['form_type'] == 'get_sentiment':
            sent = get_sentiment(phrase['phrase'])
            print(sent)
    return render_template('index.html')
if __name__ == '__main__':
    redis_client = StrictRedis(host='redis', port=6379)
    train_model()
    app.run(host='0.0.0.0')