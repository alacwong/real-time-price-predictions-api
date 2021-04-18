from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return {
        'current': [i for i in range(10)],
        'predictions': [i + 100 for i in range(10)]
    }


if __name__ == '__main__':
    app.run()
