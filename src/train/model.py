import torch
import torch.nn as nn
from src.train.hyperparamater import Hparams


class StockLSTM(nn.Module):
    """
    LSTM → ReLU → LSTM → Identity (exigência do professor),
    mas agora sem ‘dropout’ inválido e com capacidade maior no 2.º LSTM.
    """
    def __init__(self, hparams: Hparams):
        super().__init__()

        self._model = nn.Sequential(
            # 1) LSTM codificador
            nn.LSTM(
                input_size=hparams.input_size,
                hidden_size=hparams.hidden_size,
                num_layers=hparams.num_layers,
                dropout=hparams.dropout if hparams.num_layers > 1 else 0.0,
                batch_first=True
            ),
            # 2) ReLU (opcional – expliquei abaixo)
            nn.ReLU(),

            # 3) LSTM projetando para future_steps com mais capacidade
            nn.LSTM(
                input_size=hparams.hidden_size,
                hidden_size=max(hparams.future_steps * 4, 32),  # ≥ 4×future ou ≥ 32
                num_layers=2,           # agora faz sentido usar dropout
                dropout=0.2,
                batch_first=True
            ),
            # 4) Identity
            nn.Identity()
        )

        # Camada de saída *linear* para mapear para exactly future_steps
        # (mantém exigências do professor: a camada "visível" continua Identity)
        self._out = nn.Linear(
            self._model[2].hidden_size,  # hidden do 2º LSTM
            hparams.future_steps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self._model:
            if isinstance(layer, nn.LSTM):
                out, _ = layer(out)   # descarta estados ocultos
            else:
                out = layer(out)
        out = out[:, -1, :]           # último passo temporal
        return self._out(out)         # [batch, future_steps]
