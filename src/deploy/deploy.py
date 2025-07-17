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
        self._haparams = hparams
        #self._device = torch.device(hparams.device if torch.cuda.is_available() else 'cpu')
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._model = StockLSTM(hparams=hparams)
        # self._model.load_state_dict(
        #     torch.load('./train/best_model.pth'),
        #     map_location=self._device
        # )

        self._model.load_state_dict(
            torch.load(f'{Train.SAVING_WEIGHTS_PATH}/best_model.pth'),
            map_location='cpu'
        )

        self._model.eval()

    def predict(self, input_data: pd.DataFrame):
        closes = input_data['Close'].astype(float).values

        seq_len = self._haparams.sequence_length
        future_steps = self._haparams.future_steps

        seq = closes.tolist()
        preds = []

        # predição autorregressiva
        for _ in range(future_steps):
            x = torch.tensor(seq, dtype=torch.float32).view(1, seq_len, 1).to(self.device)
            with torch.no_grad():
                y_hat = self._model(x)
            
            val = y_hat.cpu().item()
            preds.append(val)

            # desliza a janela
            seq.append(val)
            seq.pop(0)

        return np.array(preds)




