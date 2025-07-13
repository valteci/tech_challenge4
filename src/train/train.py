import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from hyperparamater import Hparams
from model import StockLSTM
import numpy as np
import pandas as pd



class Train:

    DATA_PATH = 'raw_data.csv'

    def __init__(self, hparams: Hparams, seed: int, data: pd.DataFrame = None):
        self._hparams = hparams
        self._data = data
        self._X = []
        self._y = []
        self._X_train = []
        self._X_test = []
        self._y_train = []
        self._y_test = []
        self._train_loader: DataLoader = None
        self._test_loader: DataLoader = None
        self._device = torch.device(hparams.device if torch.cuda.is_available() else 'cpu')
        self._model = StockLSTM(hparams).to(self._device)
        self._loss_function = nn.MSELoss()
        self._seed = seed

        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=self._hparams.learning_rate,
            weight_decay=self._hparams.weight_decay
        )


    
    # 1) CARREGAR DADOS
    def _load_data(self):
        df = pd.read_csv(Train.DATA_PATH, parse_dates=['Date'])
        self._data = df.sort_values('Date').reset_index(drop=True)


    # 2) GERAR SEQUÊNCIAS DE X E Y
    def _create_sequences(self):
        """
        Gera X e y a partir dos dados (_data).
        coloca o resultado nas propriedades _X e _y
        _X: shape [n_samples, seq_len, input_size]
        _y: shape [n_samples, future_steps]
        """
        data = self._data[['Close']].values.astype(np.float32)  # só Close; mude se quiser multifeature
        seq_len = self._hparams.sequence_length
        fut    = self._hparams.future_steps

        self._X, self._y = [], []
        for i in range(len(data) - seq_len - fut + 1):
            seq_x = data[i : i + seq_len]             # janela de entrada
            seq_y = data[i + seq_len : i + seq_len + fut].flatten()  # futuro
            self._X.append(seq_x)
            self._y.append(seq_y)
        self._X = np.stack(self._X)  # [n_samples, seq_len, 1]
        self._y = np.stack(self._y)  # [n_samples, future_steps]


    # 3) DIVISÃO EM TREINO E TESTE
    def _train_test_split(self, train_size: float):

        if train_size <= 0:
            raise ValueError('train_size deve ser > 0')

        n_train = int(train_size * len(self._X))

        self._X_train = self._X[:n_train]
        self._y_train = self._y[:n_train]

        self._X_test = self._X[n_train:]
        self._y_test = self._y[n_train:]
        

    # 4) CARREGAR OS DADOS NO DATALOADER
    def _load_data_loader(self):
        train_ds = TensorDataset(
            torch.from_numpy(self._X_train), 
            torch.from_numpy(self._y_train)
        )

        test_ds = TensorDataset(
            torch.from_numpy(self._X_test), 
            torch.from_numpy(self._y_test)
        )

        self._train_loader = DataLoader(
            train_ds,
            batch_size=self._hparams.batch_size,
            shuffle=False
        )

        self._test_loader = DataLoader(
            test_ds,
            batch_size=self._hparams.batch_size,
            shuffle=False
        )
        

    # 5) TREINANDO 1 EPOCA
    def _train_epoch(self):
        self._model.train()
        total_loss = 0.0
        for xb, yb in self._train_loader:
            xb, yb = xb.to(self._device), yb.to(self._device)
            self._optimizer.zero_grad()
            preds = self._model(xb)
            loss  = self._loss_function(preds, yb)        # yb também [batch, future_steps]
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(self._train_loader.dataset)


    # 6) TESTANDO 1 EPOCA
    def _eval_epoch(self):
        self._model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in self._test_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                preds = self._model(xb)
                loss  = self._loss_function(preds, yb)
                total_loss += loss.item() * xb.size(0)
        return total_loss / len(self._test_loader.dataset)


    # 7) TRAIN
    def _train(self):
        best_val = float('inf')
        for epoch in range(1, self._hparams.n_epochs + 1):
            tr_loss = self._train_epoch()
            va_loss = self._eval_epoch()
            if va_loss < best_val:
                best_val = va_loss
                torch.save(self._model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch:03d} — Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}")

        print("Treino concluído. Melhor Val Loss:", best_val)


    def train(self):

        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(self._seed)
    
        self._load_data()
        self._create_sequences()
        self._train_test_split(train_size=0.7)
        self._load_data_loader()
        self._train()




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

trainner = Train(hparams, 42)
trainner.train()