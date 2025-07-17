from src.services.downloader import Downloader
from src.train.train import Train
from src.train.hyperparamater import Hparams

class Pipeline:
    def __init__(self):
        pass

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

        features = ['Close']

        hparams = Hparams(
            input_size      = len(features),
            hidden_size     = 50,
            num_layers      = 2,
            dropout         = 0.2,
            sequence_length = 60,
            future_steps    = 5,
            batch_size      = 32,
            learning_rate   = 1e-3,
            weight_decay    = 1e-5,
            n_epochs        = 100,
            device          = 'cpu'
        )
        
        trainner = Train(hparams, 65462, features)
        trainner.train()


    # CARREGA O MODELO E COLOCA EM PRODUÇÃO
    def _deploy_model(self) -> None:
        pass



