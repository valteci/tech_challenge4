from src.services.downloader import Downloader
from src.train.train import Train
from src.train.hyperparamater import Hparams
from src.deploy.deploy import Deploy
from src.deploy.fetch import Fetch
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

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
        self._client = MlflowClient()

        mlflow.set_tracking_uri("file:///app/statistics")
        

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

    
    # FAZ PREDICT
    def predict(self) -> list[float]:
        fetch = Fetch(self.stock, self._hparams.sequence_length, 2)
        data = fetch.get_input()
        predicted = list(self._deploy.predict(data))
        predicted = [round(value, 2) for value in predicted]
        return predicted


    # PEGA ESTATÍSTICAS DE TODOS OS EXPERIMENTOS
    def get_statistics(self):
        # 1) Pegar todos os experimentos (ativos + arquivados)
        exps = self._client.search_experiments(view_type=ViewType.ALL)
        
        experiments_data = []
        for exp in exps:
            exp_dict = {
                "experiment_id":   exp.experiment_id,
                "name":            exp.name,
                "artifact_uri":    exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "runs": []
            }

            # 2) Buscar todos os runs desse experimento
            runs = self._client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="",  # sem filtro p/ trazer tudo
                order_by=["attributes.start_time DESC"],
                max_results=500  # ajusta conforme seu volume
            )

            # 3) Extrair params, metrics e metadados de cada run
            for run in runs:
                info, data = run.info, run.data
                exp_dict["runs"].append({
                    "run_id":       info.run_id,
                    "status":       info.status,
                    "start_time":   info.start_time,
                    "end_time":     info.end_time,
                    "params":       dict(data.params),
                    "metrics":      dict(data.metrics),
                    "tags":         dict(data.tags),
                    "artifact_uri": info.artifact_uri
                })

            experiments_data.append(exp_dict)

        return experiments_data


