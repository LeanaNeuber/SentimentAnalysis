from flask import Flask, request, render_template
from redis import RedisError, StrictRedis
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ML code for sentiment analysis, could also be in an extra class
def train_model():
    # do something to train model etc.
	dataset = pd.read_csv('data.csv')
	vector = TfidfVectorizer(sublinear_tf=True)
	tf_vector =  vector.fit(np.array(dataset.iloc[:, 1]).ravel().astype('U'))
	X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
	y = np.array(dataset.iloc[:, 0]).ravel()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
	model = LogisticRegression(solver='lbfgs')
	model.fit(X_train, y_train)
    # store model to redis with pickle
    pickled_model = pickle.dumps(model)

    try:
        redis_client.set('ml_model', pickled_model)
    except RedisError:
        status = "fail"


# Get sentiment for phrase for pretrained model
def get_sentiment(phrase):
    # Get stored model from redis
    try:
        unpacked_model = pickle.loads(redis_client.get('ml_model'))
		#apply model on phrase
		test_prediction_lr = unpacked_model.predict(phrase)
		test_prediction 
        return test_prediction
    except RedisError:
        return "fail"


@app.route('/', methods=['GET', 'POST'])
def index():
    train_model()
    if request.method == 'POST':
        phrase = request.form
        if phrase['form_type'] == 'get_sentiment':
            
			return get_sentiment(phrase['phrase'])
    return render_template('index.html')

if __name__ == '__main__':
    redis_client = StrictRedis(host='redis', port=6379)
    app.run(host='0.0.0.0')




