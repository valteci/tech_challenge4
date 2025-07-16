from flask import Flask, jsonify
from src.pipeline import Pipeline

app = Flask(__name__)

@app.route('/')
def hello_world():
    pipeline = Pipeline()
    stocks = ['WEGE3.SA', 'PETR4.SA', 'VALE3.SA']
    pipeline._download_data(stocks, '2010-01-01', '2025-01-01')
    pipeline._train_model()

    return '<p>Hello World!</p>'

