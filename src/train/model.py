import torch
import torch.nn as nn
from src.train.hyperparamater import Hparams

class StockLSTM(nn.Module):
    """
    Rede neural LSTM para previsão de preços de ações.

    Este modelo encapsula duas camadas LSTM sequenciais com camadas de dropout 
    intermediárias e uma camada linear de saída, permitindo que, a partir de uma 
    sequência histórica de preços, sejam previstas as cotações para os próximos 
    passos definidos.

    A estrutura inclui:
      - Uma primeira camada LSTM para extrair padrões temporais da sequência de entrada.
      - Camada de Dropout para reduzir overfitting após a primeira LSTM.
      - Segunda camada LSTM com tamanho de estado oculto ajustado conforme o número 
        de passos futuros, apoiada por outro Dropout.
      - Camada linear final que mapeia o último estado oculto para a predição dos 
        valores futuros.

    A saída do modelo é um tensor de formato [batch_size, future_steps], com valores 
    arredondados conforme necessário pelo pipeline de inferência.

    Attributes:
        lstm1 (nn.LSTM): primeira camada LSTM para processar a entrada.
        dp (nn.Dropout): camada de dropout aplicada após lstm1.
        lstm2 (nn.LSTM): segunda camada LSTM para ampliar a capacidade de modelagem.
        dp_out (nn.Dropout): camada de dropout aplicada ao último estado oculto.
        output_layer (nn.Linear): camada linear que gera as previsões para os próximos passos.
    """
    def __init__(self, hparams: Hparams):
        super().__init__()
        
        # 1) Camada LSTM
        self.lstm1 = nn.LSTM(
            input_size=hparams.input_size,
            hidden_size=hparams.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Camada de dropout
        self.dp  = nn.Dropout(0.3)

        # Camada de LSTM
        self.lstm2 = nn.LSTM(
            input_size=hparams.hidden_size,
            hidden_size=max(hparams.future_steps * 4, 32),
            num_layers=1,
            batch_first=True
        )

        # Camada de dropout
        self.dp_out = nn.Dropout(0.1)

        # 4) Camada de saída linear
        self.output_layer = nn.Linear(
            self.lstm2.hidden_size,
            hparams.future_steps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out1, _ = self.lstm1(x)
        out1    = self.dp(out1)
        out2, _ = self.lstm2(out1)

        last_time_step = out2[:, -1, :]
        last_time_step = self.dp_out(last_time_step) # dropout

        # final_out tem a forma [batch, future_steps]
        return self.output_layer(last_time_step)