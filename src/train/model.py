import torch
import torch.nn as nn
import pandas as pd
from hyperparamater import Hparams


class StockLSTM(nn.Module):
    def __init__(self, hparams: Hparams):
        super(StockLSTM, self).__init__()
        
        self._lstm = nn.LSTM(
            input_size=hparams.input_size,
            hidden_size=hparams.hidden_size,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout,
            batch_first= True
        )

        self._fc = nn.Sequential(
            nn.Linear(hparams.hidden_size, hparams.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hparams.hidden_size // 2, hparams.future_steps)
        )


    def forward(self, x):
        out, _ = self._lstm(x)

        last_hidden = out[:, -1, :]

        y_pred = self._fc(last_hidden)
        return y_pred




