import concurrent.futures
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance
from tqdm import tqdm

from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pandas as pd
import time
import requests as re

"""////"""
tradeStatsColumn = [
    'tradedate','tradetime','secid','pr_open','pr_high','pr_low', \
    'pr_close','pr_std','vol','val','trades','pr_vwap','pr_change', \
    'trades_b','trades_s','val_b','val_s','vol_b','vol_s','disb', \
    'pr_vwap_b','pr_vwap_s','SYSTIME' \
]

def getSuperCandles(ticker,start_data,end_data,pandas_data=True):
    null = None
    tradeStatsUrl = f'https://iss.moex.com/iss/datashop/algopack/eq/tradestats/{ticker}.json?from={start_data}&till={end_data}'
    responseTradeStats = eval(re.get(tradeStatsUrl).text)
    result = dict()
    result['tradeStats'] = dict()
    for data in responseTradeStats['data']['data']:
        result['tradeStats'][data[0]+' '+data[1]] = [data[i] for i in range(3,len(data)-1)]
    timeKeys = []
    for key in result.keys():
        timeKeys = timeKeys + list(result[key].keys())
    timeKeys = list(set(timeKeys))
    timeKeys.sort()
    resultData = []
    for key in timeKeys:
        try:
            tradeStats = result['tradeStats'][key]
        except:
            tradeStats = [None for i in range(3,len(tradeStatsColumn)-1)]
        data = list()
        data = [key,ticker] + tradeStats
        resultData.append(data)
    ts = [tradeStatsColumn[i] for i in range(3,len(tradeStatsColumn)-1)]
    columns = ['Date','symbol'] + ts
    if pandas_data:
        df = pd.DataFrame(resultData,columns=columns)
        df = df.set_index('Date')
        df.index = pd.to_datetime(df.index)
        return  df , columns
    else:
        return resultData , columns
    
def getRangeSuperCandles(ticker,minData):
    delta = timedelta(days=7)
    delta2 = timedelta(days=1)
    current_dateTime = datetime.now()
    first_data = current_dateTime - delta
    superCandlesBach = []
    while first_data > minData:
        start_data_str = first_data.strftime('%Y-%m-%d')
        end_data_str = current_dateTime.strftime('%Y-%m-%d')
        newSuperCandles , _ = getSuperCandles(ticker,start_data_str,end_data_str)
        superCandlesBach.append(newSuperCandles)
        current_dateTime = current_dateTime - delta
        first_data = current_dateTime - delta
        current_dateTime = current_dateTime - delta2
        time.sleep(1)
    superCandlesBach.reverse()
    superCandles = pd.concat(superCandlesBach)
    return superCandles

def getHistory(ticker):
    delta = timedelta(days=14)
    minData = datetime.now() - delta
    df = getRangeSuperCandles('SBER',minData)
    df = df.rename(columns={"pr_open": "Open", "pr_high": "High","pr_low":"Low","pr_close":"Close","vol":"Volume"})
    df = df.iloc[::-1]
    df["Dividends"] = [0 for i in range(len(df))]
    df["Stock Splits"] = [0 for i in range(len(df))]
    return df[["Open","High","Low","Close","Volume","Dividends","Stock Splits"]]


class RawStockDataHolder:
    def __init__(self, ticker_symbols: list, period_years: int = 5, interval: int = 1):
        self.ticker_symbols = ticker_symbols
        self.period_years = period_years
        self.interval = interval

        max_values_per_stock = self.period_years * self.interval * 365
        nb_ticker_symbols = len(self.ticker_symbols)

        self.dates = np.zeros((nb_ticker_symbols, max_values_per_stock))
        self.values = np.zeros((nb_ticker_symbols, max_values_per_stock), dtype=np.float32)
        self.nb_of_valid_values = np.zeros(nb_ticker_symbols, dtype=np.int32)

        self.symbol_to_label = {symbol: label for label, symbol in enumerate(ticker_symbols)}
        self.label_to_symbol = {label: symbol for symbol, label in self.symbol_to_label.items()}

        self.is_filled = False

    def _download_stock_data(self, symbol: str) -> pd.DataFrame:
        ticker = yfinance.Ticker(symbol)
        period_str = f"{self.period_years}y"
        interval_str = f"{self.interval}d"
        ticker_df = getHistory(ticker)
        if ticker_df.empty or len(ticker_df) == 0:
            raise ValueError(f"{symbol} does not have enough data")
        return ticker_df

    def _get_stock_data_for_symbol(self, symbol: str) -> Tuple[np.ndarray, np.ndarray, int]:
        ticker_df = self._download_stock_data(symbol=symbol)
        close_values = ticker_df["Close"].values
        dates = ticker_df.index.values
        label = self.symbol_to_label[symbol]
        return close_values, dates, label

    def fill(self):
        """
        Fills the data holder with the defined stock data
        Returns:
            None
        """

        pbar = tqdm(desc="Symbol data download", total=len(self.ticker_symbols))

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future_to_symbol = {}
            for symbol in self.ticker_symbols:
                future = pool.submit(self._get_stock_data_for_symbol, symbol=symbol)
                future_to_symbol[future] = symbol

            for future in concurrent.futures.as_completed(future_to_symbol):
                completed_symbol = future_to_symbol[future]
                try:
                    close_values, dates, label = future.result()
                    self.values[label, :len(close_values)] = close_values
                    self.dates[label, :len(dates)] = dates
                    self.nb_of_valid_values[label] = len(dates)
                except ValueError as e:
                    print(f"ERROR with {completed_symbol}: {e}")
                    continue

                pbar.update(1)
        self.is_filled = True
        pbar.close()

    def create_filename_for_today(self) -> str:
        current_date = datetime.now().strftime("%Y_%m_%d")
        file_name = f"data_holder_{self.period_years}y_{self.interval}d_{current_date}.pk"
        return file_name

    def serialize(self) -> str:
        if not self.is_filled:
            raise ValueError("You need to fill the class with data first")

        file_name = self.create_filename_for_today()
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

        return file_name

    @staticmethod
    def load(file_name: str) -> "RawStockDataHolder":
        with open(file_name, "rb") as f:
            obj = pickle.load(f)
        return obj


def initialize_data_holder(tickers: list, period_years: int, force_update: bool = False):
    data_holder = RawStockDataHolder(ticker_symbols=tickers,
                                     period_years=period_years,
                                     interval=1)

    file_path = Path(data_holder.create_filename_for_today())

    if (not file_path.exists()) or force_update:
        data_holder.fill()
        data_holder.serialize()
    else:
        data_holder = RawStockDataHolder.load(str(file_path))
    return data_holder
