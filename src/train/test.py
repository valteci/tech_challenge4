import torch
import numpy as np
import pandas as pd
from hyperparamater import Hparams
from model import StockLSTM

# --- 1) Carrega df e seus hyperparâmetros (mesmos de treino)
df = pd.read_csv('raw_data.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

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

# --- 2) Instancia o modelo e carrega pesos
device = torch.device(hparams.device if torch.cuda.is_available() else 'cpu')
model  = StockLSTM(hparams).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# --- 3) Extrai a última janela de preços
close_vals = df['Close'].values.astype(np.float32)
last_seq   = close_vals[-hparams.sequence_length:]         # shape: (sequence_length,)
# transforma em tensor [1, seq_len, 1]
x_input = torch.from_numpy(last_seq).unsqueeze(0).unsqueeze(2).to(device)

# --- 4) Gera previsões
with torch.no_grad():
    preds = model(x_input)              # tensor shape [1, future_steps]
preds = preds.squeeze(0).cpu().numpy()  # shape (future_steps,)

# --- 5) Imprime
for i, p in enumerate(preds, 1):
    print(f"Dia +{i:>2}: Previsão de fechamento = {p:.4f}")
