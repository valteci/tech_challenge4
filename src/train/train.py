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
    """
    Gerencia todo o fluxo de treinamento de uma rede LSTM para previsão de preços de ações,
    desde o carregamento dos dados até a avaliação final e salvamento dos melhores pesos.

    Esta classe realiza:
      - Leitura dos CSVs de mercado em DATA_PATH.
      - Geração de janelas de sequência (treino/validação) a partir dos dados.
      - Criação de DataLoaders para batches de treino e validação.
      - Loop de treinamento por época, com otimização via Adam e cálculo de MSE.
      - Avaliação periódica e final do modelo, incluindo RMSE e MAPE por passo futuro.
      - Salvamento dos pesos do melhor modelo em SAVING_WEIGHTS_PATH.
      - Logging de parâmetros e métricas (train_loss, val_loss, rmse, mape, training_time) no MLflow.
      - Para treinar o modelo, basta chamar o método train dessa classe, apenas isso.

    Attributes:
        DATA_PATH (str): diretório onde estão os arquivos CSV de entrada.
        SAVING_WEIGHTS_PATH (str): diretório para salvar o arquivo best_model.pth.
        _hparams (Hparams): configurações de hiperparâmetros para o treinamento.
        _model (StockLSTM): instância do modelo LSTM configurado.
        _device (torch.device): dispositivo de execução (cpu, cuda ou mps).
        _loss_function (nn.MSELoss): função de perda utilizada.
        _optimizer (optim.Optimizer): otimizador Adam para atualização dos pesos.
        _train_loader (DataLoader): iterador de lote para dados de treino.
        _test_loader (DataLoader): iterador de lote para dados de validação.
    """

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

    # CARREGAR DADOS
    def _load_data(self):
        """
        Carrega todos os arquivos CSV de dados históricos de preços disponíveis em DATA_PATH.

        Este método varre o diretório TRAIN.DATA_PATH recursivamente procurando arquivos
        com extensão `.csv`. Para cada arquivo encontrado, ele:
          - Lê o conteúdo com pandas, parseando a coluna 'Date' como datetime.
          - Ordena as linhas pelo campo 'Date' e reseta o índice.
          - Adiciona o DataFrame resultante à lista `self._data`.

        Raises:
            FileNotFoundError: se nenhum arquivo CSV for encontrado em TRAIN.DATA_PATH.
        """
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


    # GERA UMA LISTA DE JANELAS A PARTIR DE UM BLOCO CONTÍNUO
    @staticmethod
    def _make_windows(data: np.ndarray, seq_len: int, fut_len: int):
        """
        Gera janelas deslizantes de comprimento `seq_len` e seus correspondentes alvos futuros de comprimento `fut_len`.

        A partir de um array 2D `data` (formato [n_samples, n_features]), este método:
          - Cria uma lista `X` de sequências de entrada, onde cada sequência tem forma [seq_len, n_features].
          - Cria uma lista `y` de valores-alvo, extraindo o primeiro recurso de cada uma das `fut_len` etapas seguintes à sequência de entrada.

        Args:
            data (np.ndarray): array contínuo de valores (e.g., preços), shape [n_samples, n_features].
            seq_len (int): número de passos no histórico a incluir em cada janela de entrada.
            fut_len (int): número de passos futuros a prever após cada janela de entrada.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: 
                - X: lista de arrays de entrada, cada um com shape [seq_len, n_features].
                - y: lista de arrays de alvos futuros, cada um com shape [fut_len].
        """
        X, y = [], []
        n_samples = len(data) - seq_len - fut_len + 1
        for i in range(n_samples):
            X.append(data[i : i + seq_len]) # [seq_len, input_size]
            y.append(
                data[i + seq_len : i + seq_len + fut_len, 0].flatten()
            )
        
        return X, y   


    # MÉTODO PARA AVALIAÇÃO FINAL
    def _evaluate(self):
        """
        Avalia o modelo no conjunto de validação e retorna todas as previsões e alvos reais.

        Executa um forward pass em modo avaliação (sem gradientes) sobre cada batch de
        `self._test_loader`, coleta as saídas do modelo e os valores reais, concatena
        os resultados em tensores e converte para NumPy.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - preds: array de previsões com shape [n_samples, future_steps].
                - targets: array de valores reais com shape [n_samples, future_steps].
        """
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


    # GERAR SEQUÊNCIAS JÁ SEPARADAS EM TREINO / VAL
    def _create_sequences(self):
        """
        Constrói conjuntos de sequências de entrada e seus alvos de validação a partir dos dados brutos.

        Para cada DataFrame em `self._data`, este método:
          1. Extrai os valores das features definidas em `self._hparams.features`.
          2. Separa o bloco de dados em treino e validação usando `self._hparams.train_size`.
          3. Gera janelas de entrada (`seq_len`) e alvos futuros (`fut_len`) em cada bloco,
             chamando `_make_windows`.
          4. Acumula todas as janelas de treino em `_X_train` e `_y_train`, e de validação em
             `_X_test` e `_y_test`.
        Caso não sejam geradas janelas em nenhum dos blocos, levanta um ValueError.

        Após a geração, converte as listas de sequências em arrays NumPy com shapes:
          - `_X_train`: [n_train, seq_len, input_size]
          - `_y_train`: [n_train, fut_len]
          - `_X_test` : [n_val, seq_len, input_size]
          - `_y_test` : [n_val, fut_len]
        """
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
   

    # CARREGAR OS DADOS NO DATALOADER
    def _load_data_loader(self):
        """
        Cria TensorDatasets e DataLoaders para os conjuntos de treino e validação.

        Converte os arrays NumPy `_X_train`, `_y_train`, `_X_test` e `_y_test` em tensores,
        encapsula-os em TensorDatasets e, em seguida, gera DataLoaders com o tamanho de lote
        definido por `self._hparams.batch_size`. Os DataLoaders resultantes são atribuídos a
        `self._train_loader` e `self._test_loader` sem embaralhar os dados.
        """
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
        

    # TREINANDO 1 EPOCA
    def _train_epoch(self):
        """
        Executa uma passagem de treinamento (uma época) sobre todo o conjunto de treino.

        Para cada batch em `self._train_loader`, este método:
          1. Ajusta o modelo para modo treinamento.
          2. Move os tensores de entrada (`xb`) e alvo (`yb`) para `self._device`.
          3. Zera os gradientes do otimizador.
          4. Faz o forward pass para obter as previsões.
          5. Calcula a perda usando `self._loss_function` (MSE).
          6. Realiza backpropagation e atualiza os pesos via `self._optimizer`.
          7. Acumula a perda total ponderada pelo tamanho do batch.

        Returns:
            float: perda média da época, calculada como a soma das perdas de cada batch 
                   multiplicada pelo tamanho do batch, dividida pelo número total de amostras.
        """
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


    # TESTANDO 1 EPOCA
    def _eval_epoch(self):
        """
        Executa uma passagem de avaliação (uma época) sobre todo o conjunto de validação.

        Este método:
          1. Ajusta o modelo para modo avaliação.
          2. Desativa o cálculo de gradientes.
          3. Para cada batch em `self._test_loader`:
             - Move os tensores para `self._device`.
             - Faz o forward pass e calcula a perda (MSE).
             - Acumula a perda ponderada pelo tamanho do batch.
          4. Retorna a perda média da época.

        Returns:
            float: perda média de validação, obtida pela soma das perdas de cada batch
                   multiplicada pelo tamanho do batch, dividida pelo número total de amostras.
        """
        self._model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in self._test_loader:
                xb, yb      = xb.to(self._device), yb.to(self._device)
                preds       = self._model(xb)
                loss        = self._loss_function(preds, yb)
                total_loss += loss.item() * xb.size(0)

        return total_loss / len(self._test_loader.dataset)


    # TREINA O MODELO
    def _train(self):
        """
        Executa o loop completo de treinamento, validação e logging de métricas no MLflow.

        O fluxo deste método inclui:
          1. Para cada época até `self._hparams.n_epochs`:
             - Chamar `_train_epoch` para atualizar pesos no conjunto de treino.
             - Chamar `_eval_epoch` para avaliar no conjunto de validação.
             - Salvar os pesos do modelo se a perda de validação melhorar.
             - Logar `train_loss` e `val_loss` no MLflow.
          2. Após todas as épocas, recarregar o melhor modelo salvo.
          3. Executar avaliação final via `_evaluate` para obter previsões e alvos.
          4. Calcular RMSE e MAPE por passo futuro e em média, logando cada métrica no MLflow.
          5. Logar métricas agregadas finais (final_rmse, final_mape, best_val_loss, average losses).
          6. Exibir no console o resumo de RMSE, MAPE e melhor `val_loss`.

        Returns:
            None
        """
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
        """
        Orquestra todo o pipeline de treinamento da LSTM, incluindo configuração do experimento no MLflow,
        logging de hiperparâmetros, inicialização de sementes para reprodutibilidade, carregamento de dados,
        criação de sequências, DataLoaders e execução do loop de treino/validação.

        Fluxo:
          1. Define o nome do experimento no MLflow usando timestamp atual.
          2. Inicia uma nova execução (`mlflow.start_run`) e registra todos os hiperparâmetros.
          3. Configura as sementes de PyTorch e NumPy (e CUDA, se disponível) para garantir resultados reproduzíveis.
          4. Chama internamente:
             - `_load_data()` para ler os CSVs.
             - `_create_sequences()` para gerar janelas de entrada e alvos.
             - `_load_data_loader()` para criar DataLoaders.
             - `_train()` para treinar o modelo, avaliar e salvar os melhores pesos.
          5. Ao final, calcula o tempo total de treinamento e faz `mlflow.log_metric("training_time", delta)`.

        Não retorna valor, mas salva o melhor modelo em `SAVING_WEIGHTS_PATH` e armazena todas as métricas no MLflow.
        """
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



