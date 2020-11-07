from flask import Flask, request, render_template
from redis import RedisError, StrictRedis
import pickle

app = Flask(__name__)

# ML code for sentiment analysis, could also be in an extra class
def train_model():
    # do something to train model etc.

    # store model to redis with pickle
    model = "model!!!" # this should be the acutal model
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
        return unpacked_model
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




