from flask import Flask, jsonify, request
from src.pipeline import Pipeline

app         = Flask(__name__)
pipeline    = Pipeline()


#treina o modelo e faz deploy com os novos pesos
@app.route('/train', methods=['POST'])
def train():
    data = request.get_json(silent=True)

    if not data:
        return jsonify(
            {"error": "Voce precisa mandar um json valido."}
        ), 400

    if 'stock' not in data:
        return jsonify(
            {"error": "Campo 'stock' obrigatorio no corpo JSON."}
        ), 400
    
    if 'inicio' not in data:
        return jsonify(
            {"error": "Campo 'inicio' obrigatorio no corpo JSON."}
        ), 400
    
    if 'fim' not in data:
        return jsonify(
            {"error": "Campo 'fim' obrigatorio no corpo JSON."}
        ), 400
    
    inicio: str = data['inicio']
    fim   : str = data['fim']
    stock : str = data['stock']
    
    if not stock.endswith('.SA'):
        stock += '.SA'

    # Baixa os dados e coloca em ./data
    pipeline._download_data([stock], inicio, fim)

    # Treina o modelo e salva em ./saved_weights
    pipeline._train_model()

    # Faz deploy do modelo que foi treinado
    pipeline.deploy_model()

    # Seta o nome da ação para o qual o modelo foi treinado
    pipeline.stock = stock
    
    return jsonify(
        message="Treino concluido com sucesso! O modelo foi atualizado!"
    ), 200


# prevê o preço da ação
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify(
            {"error": "Voce precisa mandar um json valido."}
        ), 400

    if 'stock' not in data:
        return jsonify(
            {"error": "Campo 'stock' obrigatorio no corpo JSON."}
        ), 400
    
    stock: str      = data['stock']

    if not stock.endswith('.SA'):
        stock += '.SA'
        
    pipeline.stock  = stock

    # Se o modelo não tiver ativo, coloca em produção
    if pipeline._deploy is None:
        pipeline.deploy_model()

    # Faz a predição
    result      = pipeline.predict()
    response    = str(result)

    return jsonify({"preco": response}), 200


# Pega todas as estatísticas de todos os experimentos
@app.route('/statistics', methods=['GET'])
def statistics():
    statistics = pipeline.get_statistics()
    return jsonify({"experiments": statistics}), 200


