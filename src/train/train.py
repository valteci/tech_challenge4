import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from hyperparamater import Hparams
from model import StockLSTM
import numpy as np
import pandas as pd


# 1) FUNÇÃO PARA GERAR SEQUÊNCIAS
def create_sequences(df: pd.DataFrame, hparams: Hparams):
    """
    Gera X e y a partir do df.
    X: shape [n_samples, seq_len, input_size]
    y: shape [n_samples, future_steps]
    """
    data = df[['Close']].values.astype(np.float32)  # só Close; mude se quiser multifeature
    seq_len = hparams.sequence_length
    fut    = hparams.future_steps

    X, y = [], []
    for i in range(len(data) - seq_len - fut + 1):
        seq_x = data[i : i + seq_len]             # janela de entrada
        seq_y = data[i + seq_len : i + seq_len + fut].flatten()  # futuro
        X.append(seq_x)
        y.append(seq_y)
    X = np.stack(X)  # [n_samples, seq_len, 1]
    y = np.stack(y)  # [n_samples, future_steps]
    return X, y

# 2) CARREGA DADOS E SPLIT TEMPO-CONSCIENTE
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

X, y = create_sequences(df, hparams)
print(X[1])




exit(0)
n_train = int(0.7 * len(X))  # 70% treino
X_train, y_train = X[:n_train], y[:n_train]
X_val,   y_val   = X[n_train:], y[n_train:]

# 3) TRANSFORMA EM TENSORES E DATALOADER
train_ds = TensorDataset(
    torch.from_numpy(X_train), 
    torch.from_numpy(y_train)
)
val_ds   = TensorDataset(
    torch.from_numpy(X_val), 
    torch.from_numpy(y_val)
)

train_loader = DataLoader(train_ds, batch_size=hparams.batch_size, shuffle=False)
val_loader   = DataLoader(val_ds,   batch_size=hparams.batch_size, shuffle=False)

# 4) CRIA MODELO, OTIMIZADOR E LOOP DE TREINO
device = torch.device(hparams.device if torch.cuda.is_available() else 'cpu')
model  = StockLSTM(hparams).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=hparams.learning_rate,
    weight_decay=hparams.weight_decay
)
criterion = nn.MSELoss()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)                   # saída: [batch, future_steps]
        loss  = criterion(preds, yb)        # yb também [batch, future_steps]
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss  = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

best_val = float('inf')
for epoch in range(1, hparams.n_epochs + 1):
    tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    va_loss = eval_epoch (model,   val_loader,   criterion, device)
    if va_loss < best_val:
        best_val = va_loss
        torch.save(model.state_dict(), 'best_model.pth')
    print(f"Epoch {epoch:03d} — Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}")

print("Treino concluído. Melhor Val Loss:", best_val)