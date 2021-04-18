from flask import Flask
import os
from nn_evaluate import evaluate
from data import get_finnhub_data
from data import parse_real_time_data
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    current_data = get_finnhub_data()
    list_input, model_input = parse_real_time_data(current_data)
    output = evaluate(model_input)

    return {
        'current': list_input,
        'prediction': output
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run()

