import pandas as pd
import yfinance as yf
import time

class Fetch:
    def __init__(
            self,
            stock_name      : str,
            sequence_length : int,
            years_to_fetch  : int
    ):
        self._stock_name            = stock_name
        self._sequence_length       = sequence_length
        self._years_to_fetch        = years_to_fetch
        self._data: pd.DataFrame    = None


    def _fetch(self) -> None:

        today       = time.strftime("%Y-%m-%d")
        start_month = int(today.split('-')[1])
        start_day   = int(today.split('-')[2])
        start_year  = int(time.strftime("%Y")) - self._years_to_fetch

        # trata ano bisexto
        if start_month == 2 and start_day == 29:
            start_day = 28

        start = f'{start_year}-{start_month}-{start_day}'

        df = yf.download(
            self._stock_name,
            start=start,
            end=today
        )

        df.columns = df.columns.droplevel(1)
        self._data = df


    def get_input(self) -> pd.DataFrame:
        self._fetch() # busca os dados

        # verifica se tem dados suficientes para alimentar o modelo
        if self._sequence_length > len(self._data):
            raise ValueError(
                'Não há dados suficientes para prever o preço dessa ação'
            )

        return self._data.tail(self._sequence_length)






