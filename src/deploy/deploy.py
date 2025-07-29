from src.train.model import StockLSTM
from src.train.hyperparamater import Hparams
from src.train.train import Train
import torch
import pandas as pd
import numpy as np

class Deploy:
    """
    Carrega um modelo LSTM treinado e oferece método de inferência para gerar previsões
    a partir de dados históricos de preços.

    Ao instanciar, a classe:
      - Inicializa a arquitetura StockLSTM com os hiperparâmetros fornecidos.
      - Carrega os pesos do melhor modelo salvo durante o treinamento.
      - Ajusta o modelo para o dispositivo (CPU, CUDA ou MPS) e o coloca em modo de avaliação.

    Através do método `predict`, converte um DataFrame de preços em um tensor, executa
    forward pass no modelo e retorna as previsões como um array NumPy.
    """
    def __init__(self, hparams: Hparams):
        self._hparams   = hparams
        self._model     = StockLSTM(hparams=hparams)

        state_dict = torch.load(
            f'{Train.SAVING_WEIGHTS_PATH}/best_model.pth',
            map_location=hparams.device
        )

        self._model.load_state_dict(state_dict)
        self._model.to(hparams.device)
        self._model.eval()


    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        closes      = input_data['Close'].astype(float).values
        seq_len     = self._hparams.sequence_length

        # pega só os últimos seq_len valores
        seq         = closes[-seq_len:]

        # monta o tensor (1, seq_len, 1)
        x           = torch.tensor(seq, dtype=torch.float32) \
                            .view(1, seq_len, 1) \
                            .to(self._hparams.device)
        # inferência
        self._model.to(self._hparams.device).eval()
        with torch.no_grad():
            y_hat = self._model(x)
        
        # extrai numpy (reshape em 1D)
        preds = y_hat.cpu().numpy().reshape(-1)
        return preds


