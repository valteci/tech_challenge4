from flask import Flask, jsonify
from src.pipeline import Pipeline

app = Flask(__name__)
STOCKS = ['ITUB4.SA']
pipeline = Pipeline()

# retorna a página home
@app.route('/home')
def home():
    pass


#treina o modelo e faz deploy com os novos pesos
@app.route('/train')
def train():
    pipeline._download_data(STOCKS, '2010-01-01', '2025-01-01')
    pipeline._train_model()
    pipeline.deploy_model()
    return 'Treino concluído!'


# prevê o preço da ação
@app.route('/predict')
def predict():
    result = pipeline.predict()
    response = str(result)
    return f'<p>Preço das ações: {response}</p>'


# pega estatísticas do modelo
@app.route('/statistics')
def statistics():
    pass



