import yfinance as yf

class Downloader():
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

    def download(self):
        """Faz o download dos arquivos"""
        df = yf.download(
            self._ticker,
            start=self._start,
            end=self._end
        )

        df.columns = df.columns.droplevel(1)
        df.to_csv('raw_data.csv', sep=',')


    

down = Downloader('WEGE3.SA', '1990-03-03', '2025-03-03')
down.download()

