from src.services.downloader import Downloader
from src.train.train import Train
from src.train.hyperparamater import Hparams
from src.deploy.deploy import Deploy
from src.deploy.fetch import Fetch
import mlflow

class Pipeline:
    def __init__(self, stock=''):
        self._hparams = Hparams(
            features        = ['Close'],
            hidden_size     = 40,
            num_layers      = 2,
            dropout         = 0.5,
            sequence_length = 20,
            future_steps    = 10,
            batch_size      = 32,
            learning_rate   = 5e-4,
            weight_decay    = 1e-5,
            n_epochs        = 150,
            device          = 'cpu',    
            seed            = 6544,
            train_size      = 0.7
        )

        self._deploy: Deploy = None
        self.stock: str = stock

        mlflow.set_tracking_uri("file:///app/statistics")
        #mlflow.set_experiment("Deploy")


    # FAZ DOWNLOAD DOS DADOS DE TREINO
    def _download_data(self, ticker: (str | list[str]), start: str, end: str) -> None:

        Downloader._remove_files() # Remove os arquivos da pasta de dados

        if isinstance(ticker, str):
            down = Downloader(ticker, start, end)
            down.download()
        elif isinstance(ticker, list):
            for stock in ticker:
                down = Downloader(stock, start, end)
                down.download()


    # TREINA O MODELO
    def _train_model(self) -> None:
        trainner = Train(self._hparams)
        trainner.train()


    # CARREGA O MODELO E COLOCA EM PRODUÇÃO
    def deploy_model(self) -> None:
        self._hparams.device = 'cpu'
        self._deploy = Deploy(self._hparams)

    
    def predict(self) -> list[float]:
        fetch = Fetch(self.stock, self._hparams.sequence_length, 2)
        data = fetch.get_input()
        predicted = list(self._deploy.predict(data))
        predicted = [round(value, 2) for value in predicted]
        return predicted


    def get_statistics(self):
        pass


