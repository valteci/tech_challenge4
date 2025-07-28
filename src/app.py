from flask import Flask, jsonify, request
from src.pipeline import Pipeline

app = Flask(__name__)
pipeline = Pipeline()

#treina o modelo e faz deploy com os novos pesos
@app.route('/train', methods=['POST'])
def train():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Voce precisa mandar um json valido."}), 400

    if 'stock' not in data:
        return jsonify({"error": "Campo 'stock' obrigatorio no corpo JSON."}), 400
    
    if 'inicio' not in data:
        return jsonify({"error": "Campo 'inicio' obrigatorio no corpo JSON."}), 400
    
    if 'fim' not in data:
        return jsonify({"error": "Campo 'fim' obrigatorio no corpo JSON."}), 400
    
    inicio: str = data['inicio']
    fim:    str = data['fim']
    stock:  str = data['stock']
    
    if not stock.endswith('.SA'):
        stock += '.SA'

    pipeline._download_data([stock], inicio, fim)
    pipeline._train_model()
    pipeline.deploy_model()

    pipeline.stock = stock
    
    return jsonify(
        {
            "message": "Treino concluido com sucesso! O modelo foi atualizado!"
        }), 200


# prevê o preço da ação
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Voce precisa mandar um json valido."}), 400

    if 'stock' not in data:
        return jsonify({"error": "Campo 'stock' obrigatorio no corpo JSON."}), 400
    
    stock:  str = data['stock']
    
    pipeline.stock = stock


    if pipeline._deploy is None:
        pipeline.deploy_model()

    result = pipeline.predict()
    response = str(result)
    return jsonify(
        {
            "preco": response
        }), 200


# pega estatísticas do modelo
@app.route('/statistics')
def statistics():
    pass



