import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.train.hyperparamater import Hparams
from src.train.model import StockLSTM
import numpy as np
import pandas as pd
import os

class Train:

    DATA_PATH = './data'
    SAVING_WEIGHTS_PATH = './saved_weights'

    def __init__(
            self,
            hparams: Hparams,
            data: list[pd.DataFrame] = None
        ):

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
        self._device = torch.device(hparams.device)
        self._model = StockLSTM(hparams).to(self._device)
        self._loss_function = nn.MSELoss()

        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=self._hparams.learning_rate,
            weight_decay=self._hparams.weight_decay
        )


    # 1) CARREGAR DADOS
    def _load_data(self):
        self._data = []  # esvazia qualquer valor anterior
        for root, dirs, files in os.walk(Train.DATA_PATH):
            for fname in files:
                if fname.lower().endswith('.csv'):
                    path = os.path.join(root, fname)
                    df = pd.read_csv(path, parse_dates=['Date'])
                    df = df.sort_values('Date').reset_index(drop=True)
                    self._data.append(df)
        if not self._data:
            raise FileNotFoundError(f"Nenhum CSV encontrado em '{Train.DATA_PATH}'")


    # 2) GERAR SEQUÊNCIAS DE X E Y (para múltiplos DataFrames)
    def _create_sequences(self):
        """
        Para cada DataFrame em self._data:
          1) extrai as colunas em self._features → array de shape [n_rows, input_size]
          2) gera janelas de comprimento sequence_length para X e future_steps para y
        """
        seq_len = self._hparams.sequence_length
        fut     = self._hparams.future_steps

        all_X, all_y = [], []

        for df in self._data:
            # pega as colunas dinâmicas
            data = df[self._hparams.features].values.astype(np.float32)  # [n_rows, input_size]
            n_rows = data.shape[0]
            n_samples = n_rows - seq_len - fut + 1
            if n_samples <= 0:
                continue  # pula séries muito curtas

            for i in range(n_samples):
                # janela de entrada: [seq_len, input_size]
                seq_x = data[i : i + seq_len]
                # alvo multi‐step: flatten → [future_steps]
                seq_y = data[i + seq_len : i + seq_len + fut, 0].flatten() \
                        if self._hparams.future_steps == 1 else \
                        data[i + seq_len : i + seq_len + fut, 0].flatten()
                # se você quiser prever apenas a primeira feature (ex: Close) no y,
                # use data[..., 0], ou ajuste para prever múltiplas features.
                all_X.append(seq_x)
                all_y.append(seq_y)

        if not all_X:
            raise ValueError("Nenhuma sequência foi gerada; verifique self._data e parâmetros.")

        # empilha para formar [total_samples, seq_len, input_size] e [total_samples, future_steps]
        self._X = np.stack(all_X)
        self._y = np.stack(all_y)

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
                torch.save(
                    self._model.state_dict(),
                    f'{Train.SAVING_WEIGHTS_PATH}/best_model.pth'
                )
                
            print(f"Epoch {epoch:03d} — Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}")

        print("Treino concluído. Melhor Val Loss:", best_val)


    def train(self):

        torch.manual_seed(self._hparams.seed)
        np.random.seed(self._hparams.seed)

        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(self._hparams.seed)
    
        self._load_data()
        self._create_sequences()
        self._train_test_split(self._hparams.train_size)
        self._load_data_loader()
        self._train()





