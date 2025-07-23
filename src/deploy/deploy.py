from src.train.model import StockLSTM
from src.train.hyperparamater import Hparams
from src.train.train import Train
from src.deploy.fetch import Fetch
import torch
import pandas as pd
import numpy as np

class Deploy:
    def __init__(self, hparams: Hparams):
        # Carrega o modelo
        self._hparams = hparams

        self._model = StockLSTM(hparams=hparams)

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
        # 1) pega só os últimos seq_len valores
        seq         = closes[-seq_len:]
        # 2) monta o tensor (1, seq_len, 1)
        x           = torch.tensor(seq, dtype=torch.float32) \
                            .view(1, seq_len, 1) \
                            .to(self._hparams.device)
        # 3) inferência
        self._model.to(self._hparams.device).eval()
        with torch.no_grad():
            y_hat = self._model(x)
        # 4) extrai numpy (reshape em 1D)
        preds = y_hat.cpu().numpy().reshape(-1)
        return preds


