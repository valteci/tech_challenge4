import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from src.train.hyperparamater import Hparams
from src.train.model import StockLSTM
import numpy as np
import pandas as pd
import os
import time
import mlflow

class Train:

    DATA_PATH = './data'
    SAVING_WEIGHTS_PATH = './saved_weights'

    def __init__(
            self,
            hparams : Hparams,
            data    : list[pd.DataFrame] = None
    ):

        self._hparams = hparams
        self._data          = data
        self._X             = []
        self._y             = []
        self._X_train       = []
        self._X_test        = []
        self._y_train       = []
        self._y_test        = []
        self._device        = torch.device(hparams.device)
        self._model         = StockLSTM(hparams).to(self._device)
        self._loss_function = nn.MSELoss()
        
        self._train_loader: DataLoader = None
        self._test_loader: DataLoader = None
        

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
            raise FileNotFoundError(
                f"Nenhum CSV encontrado em '{Train.DATA_PATH}'"
            )


    # 2) GERA UMA LISTA DE JANELAS A PARTIR DE UM BLOCO CONTÍNUO
    @staticmethod
    def _make_windows(data: np.ndarray, seq_len: int, fut_len: int):
        X, y = [], []
        n_samples = len(data) - seq_len - fut_len + 1
        for i in range(n_samples):
            X.append(data[i : i + seq_len]) # [seq_len, input_size]
            y.append(
                data[i + seq_len : i + seq_len + fut_len, 0].flatten()
            )
        
        return X, y   

    # 8) MÉTODO PARA AVALIAÇÃO FINAL
    def _evaluate(self):
        self._model.eval()
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in self._test_loader:
                xb, yb  = xb.to(self._device), yb.to(self._device)
                preds   = self._model(xb)
                all_preds.append(preds)
                all_targets.append(yb)
        
        # Concatena todas as previsões e targets
        all_preds   = torch.cat(all_preds,      dim=0)
        all_targets = torch.cat(all_targets,    dim=0)
        
        return all_preds.cpu().numpy(), all_targets.cpu().numpy()

    # 3) GERAR SEQUÊNCIAS JÁ SEPARADAS EM TREINO / VAL
    def _create_sequences(self):
        seq_len     = self._hparams.sequence_length
        fut_len     = self._hparams.future_steps
        train_sz    = self._hparams.train_size

        train_X, train_y, val_X, val_y = [], [], [], []

        for df in self._data:
            # (opcional) normalização por ativo
            values = df[self._hparams.features].values.astype(np.float32)

            split       = int(len(values) * train_sz)
            train_block = values[:split]
            val_block   = values[split:]

            # gera janelas em cada bloco
            X_t, y_t = self._make_windows(train_block, seq_len, fut_len)
            X_v, y_v = self._make_windows(val_block,   seq_len, fut_len)

            train_X.extend(X_t); train_y.extend(y_t)
            val_X.extend(X_v);   val_y.extend(y_v)

        if not train_X or not val_X:
            raise ValueError(
                "Não foram geradas janelas; verifique dados e parâmetros."
            )

        # converte para arrays
        self._X_train = np.stack(train_X) # [n_train, seq_len, input_size]
        self._y_train = np.stack(train_y) # [n_train, fut_len]
        self._X_test  = np.stack(val_X)
        self._y_test  = np.stack(val_y)

    
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

            # yb também [batch, future_steps]
            loss  = self._loss_function(preds, yb)
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
                xb, yb      = xb.to(self._device), yb.to(self._device)
                preds       = self._model(xb)
                loss        = self._loss_function(preds, yb)
                total_loss += loss.item() * xb.size(0)

        return total_loss / len(self._test_loader.dataset)


    # 7) TRAIN (ATUALIZADO)
    def _train(self):
        history = {"train_loss": [], "val_loss": []}
        best_val = float('inf')
        for epoch in range(1, self._hparams.n_epochs + 1):
            tr_loss = self._train_epoch()
            va_loss = self._eval_epoch()
            if va_loss < best_val:
                best_val = va_loss
                # Salvar apenas o melhor modelo
                torch.save(
                    self._model.state_dict(),
                    f'{Train.SAVING_WEIGHTS_PATH}/best_model.pth'
                )

            mlflow.log_metric("train_loss", tr_loss,    step=epoch)
            mlflow.log_metric("val_loss",   va_loss,    step=epoch)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
        
            print(f"Epoch {epoch:03d} — Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}")

        # ================================================================
        # AVALIAÇÃO FINAL COM MELHOR MODELO
        # ================================================================
        
        # Carregar o melhor modelo
        self._model.load_state_dict(torch.load(f'{Train.SAVING_WEIGHTS_PATH}/best_model.pth'))
        
        # Obter previsões finais
        preds, targets = self._evaluate()
        
        # Calcular RMSE e MAPE (considerando todos os passos de previsão)
        fut_len = self._hparams.future_steps
        
        # 1. RMSE por passo de tempo
        rmse_per_step = []
        for step in range(fut_len):
            rmse = np.sqrt(mean_squared_error(
                targets[:, step], 
                preds[:, step]
            ))
            rmse_per_step.append(rmse)
            mlflow.log_metric(f"rmse_step_{step+1}", rmse)
        
        # RMSE médio (todos os passos)
        final_rmse = np.mean(rmse_per_step)
        
        # 2. MAPE por passo de tempo
        mape_per_step = []
        epsilon = 1e-8  # evitar divisão por zero
        for step in range(fut_len):
            # Calcular MAPE com proteção contra valores zero
            ape = np.abs((targets[:, step] - preds[:, step]) / 
                        (np.abs(targets[:, step]) + epsilon))
            mape = np.mean(ape) * 100  # em percentual
            mape_per_step.append(mape)
            mlflow.log_metric(f"mape_step_{step+1}", mape)
        
        # MAPE médio (todos os passos)
        final_mape = np.mean(mape_per_step)

        # ================================================================
        # LOG DAS MÉTRICAS FINAIS
        # ================================================================
        mlflow.log_metric("final_rmse", final_rmse)
        mlflow.log_metric("final_mape", final_mape)
        mlflow.log_metric("best_val_loss",      best_val)
        
        # Log adicional das métricas de perda
        mlflow.log_metric(
            "history_train_loss",
            np.mean(history["train_loss"])
        )
        
        mlflow.log_metric(
            "history_val_loss",
            np.mean(history["val_loss"])
        )
        
        print("\n" + "="*50)
        print(f"Final RMSE: {final_rmse:.4f}")
        print(f"Final MAPE: {final_mape:.2f}%")
        print("="*50)
        print("Treino concluído. Melhor Val Loss:", best_val)


    def train(self):

        # Nome do experimento no mlflow
        now = time.strftime("%Y-%m-%d-%H:%M:%S")
        mlflow.set_experiment(now)

        # Tempo inicial em que o treinamento começou:
        epoch_start = time.perf_counter()

        with mlflow.start_run():
            mlflow.log_params({
                "hidden_size":    self._hparams.hidden_size,
                "num_layers":     self._hparams.num_layers,
                "dropout":        self._hparams.dropout,
                "sequence_length":self._hparams.sequence_length,
                "learning_rate":  self._hparams.learning_rate,
                "batch_size":     self._hparams.batch_size,
                "weight_decay":   self._hparams.weight_decay,
                "n_epochs":       self._hparams.n_epochs,
                "future_steps":   self._hparams.future_steps,
                "device":         self._hparams.device,
                "seed":           self._hparams.seed,
                "train_size":     self._hparams.train_size
            })

            torch.manual_seed(self._hparams.seed)
            np.random.seed(self._hparams.seed)

            if torch.cuda.is_available(): 
                torch.cuda.manual_seed_all(self._hparams.seed)
        
            self._load_data()
            self._create_sequences()
            self._load_data_loader()
            self._train()

            # Tempo final em que o treinamento terminou
            epoch_end = time.perf_counter()

            delta = epoch_end - epoch_start

            mlflow.log_metric("training_time", delta)



