import yfinance as yf
import os
import shutil

class Downloader():

    SAVING_PATH='./data'

    def __init__(self, ticker: str = '', start: str = '', end: str = ''):
        self._ticker = ticker
        self._start = start
        self._end = end


    @property
    def ticker(self):
        """Getter method"""
        return self._ticker
    
    @property
    def start(self):
        """Getter method"""
        return self._start
    
    @property
    def end(self):
        """Getter method"""
        return self._end
    
    @ticker.setter
    def ticker(self, value):
        """Setter method"""
        self._ticker = value

    @start.setter
    def start(self, value):
        """Setter method"""
        self._start = value

    @end.setter
    def end(self, value):
        """Setter method"""
        self._end = value

    @staticmethod
    def _remove_files() -> None:
        """Remove todos os arquivos da pasta SAVING_PATH"""
        path = Downloader.SAVING_PATH # pasta que têm os arquivos

        if not os.path.isdir(path):
            raise ValueError(f'diretório {path} não existe!')
        

        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            try:
                if os.path.isfile(full_path) or os.path.islink(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
            except Exception as e:
                print(f"Erro removendo '{full_path}': {e}")



    def download(self):
        """Faz o download dos arquivos"""
        df = yf.download(
            self._ticker,
            start=self._start,
            end=self._end
        )

        df.columns = df.columns.droplevel(1)
        df.to_csv(f'{Downloader.SAVING_PATH}/{self._ticker}.csv', sep=',')



