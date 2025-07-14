from train.train import StockLSTM
from train.hyperparamater import Hparams
from test.fetch import Fetch
import torch
import pandas as pd

class Test:
    def __init__(self, hparams: Hparams):
        # Carrega o modelo
        self._haparams = hparams
        self._device = torch.device(hparams.device if torch.cuda.is_available() else 'cpu')


        self._model = StockLSTM(hparams=hparams)
        self._model.load_state_dict(
            torch.load('./train/best_model.pth'),
            map_location=self._device
        )

        self._model.eval()

    def predict(self, input_data: pd.DataFrame):
        closes = input_data['Close'].astype(float).values

        seq_len = self._haparams.sequence_length
        future_steps = self._haparams.future_steps

hparams = Hparams(
    input_size      = 1,
    hidden_size     = 50,
    num_layers      = 2,
    dropout         = 0.2,
    sequence_length = 60,
    future_steps    = 5,
    batch_size      = 32,
    learning_rate   = 1e-3,
    weight_decay    = 1e-5,
    n_epochs        = 100,
    device          = 'cuda'
)


fetch = Fetch('WEGE3.SA', 60, 1)

# df com os últimos 60 preços, tem as colunas: [Date,Close,High,Low,Open,Volume]
input_data = fetch.get_input() 

test = Test(hparams)
previsao = test.predict(input_data)

print('A previsão de preço é essa: ', previsao)
