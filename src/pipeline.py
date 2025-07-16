from src.services.downloader import Downloader
from src.train.train import Train
from src.train.hyperparamater import Hparams

class Pipeline:
    def __init__(self):
        pass

    def _download_data(self, ticker: (str | list[str]), start: str, end: str) -> None:

        if isinstance(ticker, str):
            down = Downloader(ticker, start, end)
            down.download()
        elif isinstance(ticker, list):
            for stock in ticker:
                down = Downloader(stock, start, end)
                down.download()

    
    def _train_model(self) -> None:
        hparams = Hparams(
            input_size      = 5,
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
        features = ['Close','High','Low','Open','Volume']
        trainner = Train(hparams, 65462, features)
        trainner.train()


        