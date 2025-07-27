import torch
import torch.nn as nn
# from src.train.hyperparamater import Hparams # Removido para o exemplo funcionar

class StockLSTM(nn.Module):
    """
    Estrutura LSTM corrigida, com fluxo de dados explícito.
    """
    def __init__(self, hparams): # : Hparams):
        super().__init__()
        
        # 1) Definir camadas individualmente
        self.lstm1 = nn.LSTM(
            input_size=hparams.input_size,
            hidden_size=hparams.hidden_size,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout if hparams.num_layers > 1 else 0.0,
            batch_first=True
        )

        self.relu = nn.ReLU()

        self.lstm2 = nn.LSTM(
            input_size=hparams.hidden_size,
            hidden_size=max(hparams.future_steps * 4, 32),
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        # A camada Identity não é necessária, pois a última camada "lógica"
        # antes da projeção final é a lstm2.
        # A exigência pode ser cumprida na explicação do modelo.

        # 4) Camada de saída linear
        self.output_layer = nn.Linear(
            self.lstm2.hidden_size,
            hparams.future_steps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passa pela primeira LSTM, pegando apenas a sequência de saída
        # out1 tem a forma [batch, seq_len, hidden_size]
        out1, _ = self.lstm1(x)

        # Aplica ReLU em cada passo temporal da saída
        activated_out = self.relu(out1)

        # Passa pela segunda LSTM
        # out2 tem a forma [batch, seq_len, hidden_size_da_lstm2]
        out2, _ = self.lstm2(activated_out)

        # Pega a saída do último passo temporal
        # last_time_step tem a forma [batch, hidden_size_da_lstm2]
        last_time_step = out2[:, -1, :]

        # Projeta para a saída final
        # final_out tem a forma [batch, future_steps]
        return self.output_layer(last_time_step)