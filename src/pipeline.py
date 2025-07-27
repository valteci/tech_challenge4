from src.services.downloader import Downloader
from src.train.train import Train
from src.train.hyperparamater import Hparams
from src.deploy.deploy import Deploy
from src.deploy.fetch import Fetch

class Pipeline:
    def __init__(self):
        self._hparams = Hparams(
            features        = ['Close'],
            hidden_size     = 50,
            num_layers      = 2,
            dropout         = 0.2,
            sequence_length = 60,
            future_steps    = 5,
            batch_size      = 32,
            learning_rate   = 1e-3,
            weight_decay    = 1e-5,
            n_epochs        = 100,
            device          = 'cpu',    
            seed            = 65424,
            train_size      = 0.7
        )

        self._deploy: Deploy = None

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
        fetch = Fetch('ITUB4.SA', self._hparams.sequence_length, 2)
        data = fetch.get_input()
        predicted = list(self._deploy.predict(data))
        return predicted


