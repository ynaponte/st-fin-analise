from dataclasses import dataclass, field
from typing import List, Union
import pandas as pd

@dataclass(frozen=True)
class Config:
    target_ticker: str
    predictor_tickers: List[str]
    horizons: List[int] = field(default_factory=lambda: [1, 5, 21])
    smoothing_window: int = 30
    smoothing_method: str = "DEMA"
    stl_period: int = 21
    alpha: float = 0.05
    n_permutations: int = 500
    knn_k: int = 5
    lag_max: int = 21
    force_continue: bool = False

    def fetch(self, tickers: Union[List[str], str], start: str, end: str) -> dict[str, pd.Series]:
        import yfinance as yf
        
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)
            
        df = yf.download(ticker_list, start=start, end=end)
        
        prices_dict = {}
        for ticker in ticker_list:
            if isinstance(df.columns, pd.MultiIndex):
                if 'Adj Close' in df.columns.levels[0] and ticker in df['Adj Close'].columns:
                    prices_dict[ticker] = df['Adj Close'][ticker].dropna()
                elif 'Close' in df.columns.levels[0] and ticker in df['Close'].columns:
                    prices_dict[ticker] = df['Close'][ticker].dropna()
                else:
                    prices_dict[ticker] = pd.Series(dtype='float64')
            else:
                if 'Adj Close' in df.columns:
                    prices_dict[ticker] = df['Adj Close'].dropna()
                elif 'Close' in df.columns:
                    prices_dict[ticker] = df['Close'].dropna()
                else:
                    prices_dict[ticker] = pd.Series(dtype='float64')
                    
        return prices_dict
